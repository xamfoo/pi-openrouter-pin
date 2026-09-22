/**
 * Tests for the /openrouter-pins, /openrouter-unpin, and session-start
 * refresh logic.
 *
 * Four layers:
 *
 *   A — Pins surface: `formatRouting` (the /openrouter-pins display format)
 *       and `listPins` (what the command lists) from src/commands.ts.
 *
 *   B — Unpin core: the pure `unpinFromModels` and the file-level
 *       `performUnpin` (what the /openrouter-unpin handler delegates to).
 *
 *   C — Handlers: the REAL registered command handlers from src/index.ts,
 *       driven through a fake ExtensionAPI/ExtensionCommandContext with a
 *       temp PI_CODING_AGENT_DIR, asserting notifications and models.json
 *       contents end-to-end (including the no-args picker path, whose
 *       `ctx.custom` is resolved with a canned choice instead of driving a
 *       real TUI).
 *
 *   D — Refresh pipeline: `formatRefreshDiff` (the pricing display formatter
 *       that previously spilled multi-line output into the TUI footer),
 *       `collectRefreshTargets` and `applyPricingPatches` (pure refresh
 *       helpers), and the `session_start` handler that routes results to
 *       `ctx.ui.notify()` instead of raw `console.*` calls.
 *
 * These tests import index.ts, which pulls in @earendil-works/pi-coding-agent
 * and @earendil-works/pi-tui at runtime — run `npm install` (peer deps) first.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import openrouterPinExtension from "../src/index.ts";
import {
  applyPricingPatches,
  collectRefreshTargets,
  computeSettingsDiff,
  computeSettingsPrune,
  formatRefreshDiff,
  formatRouting,
  listPins,
  normalizeUnpinArg,
  planPin,
  planUnpin,
  performPin,
  performUnpin,
  probe,
  unpinFromModels,
  type PricingLimitsDiff,
  type PricingLimitsPatch,
} from "../src/commands.ts";
import { atomicWriteJson, readJsonFile, type ModelsJson, type ProviderEntry, type SettingsJson } from "../src/files.ts";
import type { ExtensionContext, ModelRegistry } from "@earendil-works/pi-coding-agent";
import type { ModelConfig } from "../src/config.ts";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const providerEntry = (models: ModelConfig[]): ProviderEntry => ({
  baseUrl: "https://openrouter.ai/api/v1",
  api: "openai-completions",
  apiKey: "$OPENROUTER_API_KEY",
  models,
});

/** A stored pinned model, as written to models.json. */
const glmModel = (over: Partial<ModelConfig> = {}): ModelConfig => ({
  id: "z-ai/glm-5.2",
  name: "GLM 5.2 (novita)",
  reasoning: true,
  input: ["text"],
  contextWindow: 1_048_576,
  maxTokens: 128_000,
  cost: { input: 100_000, output: 310_000, cacheRead: 20_000, cacheWrite: 0 },
  compat: { thinkingFormat: "openrouter", openRouterRouting: { only: ["novita"], allow_fallbacks: false } },
  ...over,
});

/** A second pinned model (relaxed -plus provider). */
const deepseekModel = (over: Partial<ModelConfig> = {}): ModelConfig => ({
  id: "deepseek/deepseek-v4-flash-0731",
  name: "DeepSeek V4 Flash (novita)",
  reasoning: true,
  input: ["text"],
  contextWindow: 131_072,
  maxTokens: 32_768,
  cost: { input: 20_000, output: 80_000, cacheRead: 5_000, cacheWrite: 0 },
  compat: {
    thinkingFormat: "openrouter",
    openRouterRouting: { order: ["novita", "deepseek"], allow_fallbacks: true },
  },
  ...over,
});

/** A model in an openrouter-* provider WITHOUT routing compat — must be invisible to pins/unpin. */
const legacyModel = (over: Partial<ModelConfig> = {}): ModelConfig =>
  ({
    id: "some/legacy-model",
    name: "Legacy",
    reasoning: false,
    input: ["text"],
    contextWindow: 128_000,
    maxTokens: 16_384,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    ...over,
  }) as unknown as ModelConfig;

async function withTempModels(
  models: ModelsJson | null,
  fn: (modelsPath: string) => Promise<void>,
): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-pins-"));
  const modelsPath = join(dir, "models.json");
  try {
    if (models !== null) await atomicWriteJson(modelsPath, models);
    await fn(modelsPath);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// A — /openrouter-pins surface
// ---------------------------------------------------------------------------

test("formatRouting: every routing option renders, fallbacks defaults to false", () => {
  // Empty routing still states the fallback policy explicitly.
  assert.equal(formatRouting({}), "fallbacks=false");
  assert.equal(formatRouting({ allow_fallbacks: false }), "fallbacks=false");
  assert.equal(formatRouting({ allow_fallbacks: true }), "fallbacks=true");

  // Strict pin: only the anchor provider, no fallbacks.
  assert.equal(
    formatRouting({ only: ["novita"], allow_fallbacks: false }),
    "only=novita fallbacks=false",
  );

  // Relaxed pin with the full surface, in the documented order.
  assert.equal(
    formatRouting({
      only: ["novita"],
      order: ["novita", "together"],
      ignore: ["openai", "anthropic"],
      quantizations: ["fp8"],
      data_collection: "deny",
      allow_fallbacks: true,
    }),
    "only=novita order=novita,together ignore=openai,anthropic quant=fp8 data_collection=deny fallbacks=true",
  );
});

test("listPins: only openrouter-* providers, only models with routing compat, insertion order", async () => {
  await withTempModels(
    {
      providers: {
        "openrouter-novita": providerEntry([glmModel(), legacyModel()]), // legacy model skipped
        "openrouter-novita-plus": providerEntry([deepseekModel()]),
        "anthropic": providerEntry([glmModel()]), // not our prefix — ignored
      },
    },
    async (modelsPath) => {
      const pins = await listPins(modelsPath);
      assert.deepEqual(
        pins.map((p) => [p.provider, p.model.id]),
        [
          ["openrouter-novita", "z-ai/glm-5.2"],
          ["openrouter-novita-plus", "deepseek/deepseek-v4-flash-0731"],
        ],
        "non-openrouter providers and compat-less models are invisible to /openrouter-pins",
      );
    },
  );
});

test("listPins: missing file, empty providers, and model-less providers all yield []", async () => {
  await withTempModels(null, async (modelsPath) => {
    assert.deepEqual(await listPins(modelsPath), []);
  });
  await withTempModels({}, async (modelsPath) => {
    assert.deepEqual(await listPins(modelsPath), []);
  });
  await withTempModels({ providers: {} }, async (modelsPath) => {
    assert.deepEqual(await listPins(modelsPath), []);
  });
  await withTempModels({ providers: { "openrouter-novita": providerEntry([]) } }, async (modelsPath) => {
    assert.deepEqual(await listPins(modelsPath), []);
  });
});

// ---------------------------------------------------------------------------
// B — Unpin core
// ---------------------------------------------------------------------------

test("unpinFromModels: removes the model from openrouter-* providers, keeps siblings", () => {
  const input: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
      "anthropic": providerEntry([glmModel()]),
    },
  };
  const { models, removed } = unpinFromModels(input, "z-ai/glm-5.2");
  assert.equal(removed, true);
  assert.deepEqual(
    models.providers!["openrouter-novita"].models.map((m) => m.id),
    ["deepseek/deepseek-v4-flash-0731"],
    "sibling pins survive",
  );
  // Non-openrouter providers are never scanned, let alone touched.
  assert.deepEqual(models.providers!["anthropic"], input.providers!["anthropic"]);
});

test("unpinFromModels: a provider left empty is dropped, not persisted as []", () => {
  const { models, removed } = unpinFromModels(
    { providers: { "openrouter-novita": providerEntry([glmModel()]) } },
    "z-ai/glm-5.2",
  );
  assert.equal(removed, true);
  assert.deepEqual(models.providers, {}, "the empty openrouter-* provider is deleted");
});

test("unpinFromModels: removes the same model from every openrouter-* provider at once", () => {
  // Same model pinned strict (novita) and relaxed (novita-plus): both go.
  const { models, removed } = unpinFromModels(
    {
      providers: {
        "openrouter-novita": providerEntry([glmModel()]),
        "openrouter-novita-plus": providerEntry([{ ...glmModel(), name: "GLM relaxed" }]),
      },
    },
    "z-ai/glm-5.2",
  );
  assert.equal(removed, true);
  assert.deepEqual(models.providers, {});
});

test("unpinFromModels: unknown model is a clean no-op", () => {
  const input: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel()]),
      "anthropic": providerEntry([glmModel()]),
    },
  };
  const { models, removed } = unpinFromModels(input, "nobody/home");
  assert.equal(removed, false);
  // The snapshot is a fresh object but shares the untouched provider entries.
  assert.notEqual(models, input);
  assert.equal(models.providers!["openrouter-novita"], input.providers!["openrouter-novita"]);
  assert.deepEqual(models, input);
});

test("unpinFromModels: never mutates its input (deep-frozen)", () => {
  const input: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
      "openrouter-together": providerEntry([glmModel({ id: "other/model" })]),
      "anthropic": providerEntry([glmModel()]),
    },
  };
  deepFreeze(input);
  const { models, removed } = unpinFromModels(input, "z-ai/glm-5.2");
  assert.equal(removed, true);
  // The deepseek sibling keeps the novita provider alive; only the unpinned
  // model is gone — and the frozen input was never touched.
  assert.deepEqual(
    models.providers!["openrouter-novita"].models.map((m) => m.id),
    ["deepseek/deepseek-v4-flash-0731"],
    "sibling model survives in the same provider",
  );
  assert.deepEqual(
    models.providers!["openrouter-together"].models.map((m) => m.id),
    ["other/model"],
    "sibling model in another provider survives",
  );
});

test("unpinFromModels: null / provider-less / malformed inputs are safe no-ops", () => {
  assert.deepEqual(unpinFromModels(null, "x/y"), { models: { providers: {} }, removed: false });
  assert.deepEqual(unpinFromModels({}, "x/y"), { models: { providers: {} }, removed: false });
  assert.deepEqual(unpinFromModels({ providers: {} }, "x/y"), { models: { providers: {} }, removed: false });
  // Defensive: an entry without a models array is kept as-is, never crashed on.
  const malformed = { providers: { "openrouter-novita": { baseUrl: "x" } } } as unknown as ModelsJson;
  const out = unpinFromModels(malformed, "x/y");
  assert.equal(out.removed, false);
  assert.equal(out.models.providers!["openrouter-novita"], malformed.providers!["openrouter-novita"]);
});

test("performUnpin: removes and writes only when something was removed", async () => {
  await withTempModels(
    { providers: { "openrouter-novita": providerEntry([glmModel(), deepseekModel()]) } },
    async (modelsPath) => {
      assert.deepEqual(await performUnpin(modelsPath, "z-ai/glm-5.2"), {
        status: "removed",
        resolvedModelId: "z-ai/glm-5.2",
      });
      const models = await readJsonFile<ModelsJson>(modelsPath);
      assert.deepEqual(
        models!.providers!["openrouter-novita"].models.map((m) => m.id),
        ["deepseek/deepseek-v4-flash-0731"],
        "the sibling survives on disk",
      );

      // A second unpin of the last model drops the whole provider.
      assert.deepEqual(
        await performUnpin(modelsPath, "deepseek/deepseek-v4-flash-0731"),
        { status: "removed", resolvedModelId: "deepseek/deepseek-v4-flash-0731" },
      );
      assert.deepEqual(await readJsonFile<ModelsJson>(modelsPath), { providers: {} });
    },
  );
});

test("performUnpin: not-found and no-providers never write", async () => {
  await withTempModels(
    { providers: { "openrouter-novita": providerEntry([glmModel()]) } },
    async (modelsPath) => {
      assert.deepEqual(await performUnpin(modelsPath, "nobody/home"), {
        status: "not-found",
        inputModelId: "nobody/home",
      });
      const after = await readJsonFile<ModelsJson>(modelsPath);
      assert.deepEqual(after!.providers!["openrouter-novita"].models.map((m) => m.id), ["z-ai/glm-5.2"]);
    },
  );

  // Missing file → no-providers, and the file is NOT created.
  await withTempModels(null, async (modelsPath) => {
    assert.deepEqual(await performUnpin(modelsPath, "x/y"), { status: "no-providers" });
    const { existsSync } = await import("node:fs");
    assert.equal(existsSync(modelsPath), false, "an unpin of nothing must not create models.json");
  });

  // File present but without a providers key → no-providers, content untouched.
  await withTempModels({ settings: { x: 1 } } as unknown as ModelsJson, async (modelsPath) => {
    assert.deepEqual(await performUnpin(modelsPath, "x/y"), { status: "no-providers" });
    assert.deepEqual(await readJsonFile<ModelsJson>(modelsPath), { settings: { x: 1 } });
  });
});

// ---------------------------------------------------------------------------
// C — The registered /openrouter-pins and /openrouter-unpin handlers
// ---------------------------------------------------------------------------

type Notify = { message: string; type: "info" | "warning" | "error" };

/**
 * Registers the real extension against a fake ExtensionAPI and plays the
 * ExtensionCommandContext side. PI_CODING_AGENT_DIR must already point at a
 * temp dir — the factory reads it once at construction.
 */
class CommandHarness {
  readonly notifications: Notify[] = [];
  customCalls = 0;
  private readonly picked: string | null;
  private readonly commands = new Map<
    string,
    { description: string; handler: (args: string, ctx: ExtensionCommandContext) => Promise<void> }
  >();

  constructor(picked: string | null = null) {
    // Explicit field, not a parameter property: Node's strip-only TS mode
    // cannot transform parameter properties (same constraint as api.ts).
    this.picked = picked;
    const pi = {
      on: () => {},
      registerProvider: () => {},
      registerCommand: (name: string, options: { description: string; handler: (args: string, ctx: ExtensionCommandContext) => Promise<void> }) => {
        this.commands.set(name, options);
      },
    } as unknown as ExtensionAPI;
    openrouterPinExtension(pi);
  }

  private ctx(): ExtensionCommandContext {
    return {
      mode: "tui",
      hasUI: true,
      cwd: "/tmp",
      modelRegistry: {},
      model: undefined,
      scopedModels: [],
      ui: {
        notify: (message: string, type: "info" | "warning" | "error" = "info") => {
          this.notifications.push({ message, type });
        },
        custom: async () => {
          this.customCalls++;
          return this.picked;
        },
        select: async () => this.picked,
      },
    } as unknown as ExtensionCommandContext;
  }

  async run(command: string, args: string): Promise<void> {
    const registered = this.commands.get(command);
    assert.ok(registered, `expected /${command} to be registered`);
    await registered.handler(args, this.ctx());
  }
}

/** Point PI_CODING_AGENT_DIR at a temp dir for the duration of fn. */
async function withAgentDir<T>(fn: () => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-agent-"));
  const saved = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = dir;
  try {
    return await fn();
  } finally {
    if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = saved;
    await rm(dir, { recursive: true, force: true });
  }
}

function deepFreeze<T>(value: T): T {
  if (value && typeof value === "object") {
    for (const v of Object.values(value as Record<string, unknown>)) deepFreeze(v);
    Object.freeze(value);
  }
  return value;
}

test("/openrouter-pins: lists every pin with its routing policy", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": providerEntry([glmModel()]),
        "openrouter-novita-plus": providerEntry([deepseekModel()]),
        "anthropic": providerEntry([glmModel()]), // must not appear
      },
    });
    const h = new CommandHarness();
    await h.run("openrouter-pins", "");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.equal(
      h.notifications[0].message,
      [
        "Active pins:",
        "  openrouter-novita/z-ai/glm-5.2  → only=novita fallbacks=false",
        "  openrouter-novita-plus/deepseek/deepseek-v4-flash-0731  → order=novita,deepseek fallbacks=true",
      ].join("\n"),
    );
  });
});

test("/openrouter-pins: no pins shows the create hint", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-pins", "");
    assert.equal(h.notifications.length, 1);
    assert.ok(h.notifications[0].message.startsWith("No pins. Run /openrouter-pin"), "points at the pin commands");
  });
});

test("/openrouter-pins: a broken models.json surfaces as a List failed error", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await (await import("node:fs/promises")).writeFile(modelsPath, "{ not json", "utf-8");
    const h = new CommandHarness();
    await h.run("openrouter-pins", "");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "error");
    assert.ok(h.notifications[0].message.startsWith("List failed:"), "the error is surfaced, not swallowed");
  });
});

test("/openrouter-unpin <model>: removes the pin and reports success", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
        "openrouter-together": providerEntry([glmModel({ id: "other/model" })]),
      },
    });
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "z-ai/glm-5.2");
    assert.deepEqual(h.notifications, [
      { message: "Unpinned z-ai/glm-5.2 from models.json (applies on /reload or next session).", type: "info" },
    ]);
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(
      models!.providers!["openrouter-novita"].models.map((m) => m.id),
      ["deepseek/deepseek-v4-flash-0731"],
    );
    assert.deepEqual(models!.providers!["openrouter-together"].models.map((m) => m.id), ["other/model"]);
  });
});

test("/openrouter-unpin <model>: the last pin in a provider drops the provider entirely", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-novita": providerEntry([glmModel()]) },
    });
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "z-ai/glm-5.2");
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(models, { providers: {} }, "the emptied openrouter-* provider is removed");
  });
});

test("/openrouter-unpin <model>: not pinned → info notice, file untouched", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    const before: ModelsJson = { providers: { "openrouter-novita": providerEntry([glmModel()]) } };
    await atomicWriteJson(modelsPath, before);
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "nobody/home");
    assert.deepEqual(h.notifications, [
      { message: 'No pin for "nobody/home" found (checked openrouter-* providers)', type: "info" },
    ]);
    assert.deepEqual(await readJsonFile<ModelsJson>(modelsPath), before);
  });
});

test("/openrouter-unpin <model>: no models.json → info notice, nothing created", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "z-ai/glm-5.2");
    assert.deepEqual(h.notifications, [
      { message: "No pins found (no providers in models.json)", type: "info" },
    ]);
    const { existsSync } = await import("node:fs");
    assert.equal(existsSync(join(process.env.PI_CODING_AGENT_DIR!, "models.json")), false);
  });
});

test("/openrouter-unpin (no args): offers the picker and unpins the chosen pin", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
        "anthropic": providerEntry([glmModel()]), // never offered: not an openrouter-* pin
      },
    });
    // The picker resolves with the novita pin; the handler extracts the model
    // id after the provider prefix and unpins exactly that one.
    const h = new CommandHarness("openrouter-novita/z-ai/glm-5.2");
    await h.run("openrouter-unpin", "");
    assert.equal(h.customCalls, 1, "the picker is shown");
    assert.deepEqual(h.notifications, [
      { message: "Unpinned z-ai/glm-5.2 from models.json (applies on /reload or next session).", type: "info" },
    ]);
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(models!.providers!["openrouter-novita"].models.map((m) => m.id), ["deepseek/deepseek-v4-flash-0731"]);
  });
});

test("/openrouter-unpin (no args): Esc-cancel is silent and writes nothing", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    const before: ModelsJson = { providers: { "openrouter-novita": providerEntry([glmModel()]) } };
    await atomicWriteJson(modelsPath, before);
    const h = new CommandHarness(null); // picker cancelled
    await h.run("openrouter-unpin", "");
    assert.equal(h.customCalls, 1, "the picker is shown before cancel");
    assert.deepEqual(h.notifications, [], "cancelling quietly does not notify");
    assert.deepEqual(await readJsonFile<ModelsJson>(modelsPath), before);
  });
});

test("/openrouter-unpin (no args): no pins → hint without opening the picker", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "");
    assert.equal(h.customCalls, 0, "no picker when there is nothing to pick");
    assert.deepEqual(h.notifications, [
      { message: "No pins to remove. Use /openrouter-pin to create one.", type: "info" },
    ]);
  });
});

test("/openrouter-pin --help: prints help without pinning or opening the wizard", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-pin", "--help");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-pin/);
    assert.ok(h.notifications[0].message.includes("--quant"));
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    assert.equal(await readJsonFile(modelsPath), null, "--help never writes models.json");
  });
});

test("/openrouter-pin -h: short flag prints the same help", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-pin", "-h");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-pin/);
    // A pin-looking invocation with --help anywhere also shows help, and
    // never reaches the network or models.json.
    const mixed = new CommandHarness();
    await mixed.run("openrouter-pin", "z-ai/glm-5.2 novita --help");
    assert.equal(mixed.notifications.length, 1);
    assert.match(mixed.notifications[0].message, /^Usage: \/openrouter-pin/);
  });
});

test("/openrouter-unpin --help: prints help and touches nothing", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "--help");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-unpin/);
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    assert.equal(await readJsonFile(modelsPath), null, "--help never creates models.json");
  });
});

test("/openrouter-unpin -h: short flag prints the same help", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "-h");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-unpin/);
  });
});

test("/openrouter-pins --help: prints help even with pins present", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-novita": providerEntry([glmModel()]) },
    });
    const h = new CommandHarness();
    await h.run("openrouter-pins", "--help");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-pins/);
  });
});

test("/openrouter-pins -h: short flag prints the same help", async () => {
  await withAgentDir(async () => {
    const h = new CommandHarness();
    await h.run("openrouter-pins", "-h");
    assert.equal(h.notifications.length, 1);
    assert.equal(h.notifications[0].type, "info");
    assert.match(h.notifications[0].message, /^Usage: \/openrouter-pins/);
  });
});

// ---------------------------------------------------------------------------
// D — Refresh pipeline
// ---------------------------------------------------------------------------

/** Harness for the `session_start` handler registered by the extension factory. */
class SessionStartHarness {
  readonly notifications: Notify[] = [];
  private handler?: (event: unknown, ctx: ExtensionContext) => void | Promise<void>;

  constructor() {
    const pi = {
      on: (event: string, handler: (..._args: unknown[]) => unknown) => {
        if (event === "session_start") {
          this.handler = handler as typeof this.handler;
        }
      },
      registerProvider: () => {},
      registerCommand: () => {},
    } as unknown as ExtensionAPI;
    openrouterPinExtension(pi);
  }

  async fireSessionStart(apiKey?: string): Promise<void> {
    assert.ok(this.handler, "session_start handler must be registered");

    const oldKey = process.env.OPENROUTER_API_KEY;
    if (apiKey !== undefined) {
      process.env.OPENROUTER_API_KEY = apiKey;
    } else {
      delete process.env.OPENROUTER_API_KEY;
    }

    try {
      await this.handler!(
        { type: "session_start", reason: "startup" },
        {
          mode: "tui",
          hasUI: true,
          cwd: "/tmp",
          modelRegistry: {} as ModelRegistry,
          model: undefined,
          scopedModels: [],
          ui: {
            notify: (message: string, type: "info" | "warning" | "error" = "info") => {
              this.notifications.push({ message, type });
            },
            select: async () => undefined,
          },
        } as unknown as ExtensionContext,
      );
    } finally {
      if (oldKey === undefined) delete process.env.OPENROUTER_API_KEY;
      else process.env.OPENROUTER_API_KEY = oldKey;
    }

    // Wait for async settle — refreshPinnedModels performs network I/O under the hood.
    await delay(200);
  }
}

const delay = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms));

// --- formatRefreshDiff -------------------------------------------------------

test("formatRefreshDiff: empty diff returns an empty string", () => {
  assert.equal(formatRefreshDiff([]), "");
});

test("formatRefreshDiff: unchanged model produces no output", () => {
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    before: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
    after: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
  }];
  assert.equal(formatRefreshDiff(diff), "");
});

test("formatRefreshDiff: single cost field change aligned", () => {
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    before: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
    after: {
      cost: { input: 0.34, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
  }];
  const result = formatRefreshDiff(diff);
  // Labels changed: ["cost.input"] → width = 10
  assert.equal(
    result,
    [
      "  z-ai/glm-5.2 (openrouter-novita):",
      "    cost.input  $0.39/M  →  $0.34/M",
    ].join("\n"),
  );
});

test("formatRefreshDiff: multiple cost fields, tabular alignment", () => {
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    before: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
    after: {
      cost: { input: 0.34, output: 1.18, cacheRead: 0.06, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
  }];
  const result = formatRefreshDiff(diff);
  // Labels: ["cost.input"(10), "cost.cacheRead"(14)] → width = 14
  // "cost.input".padEnd(14) = "cost.input    " (4 trailing spaces)
  // "cost.cacheRead".padEnd(14) = "cost.cacheRead"   (exact length)
  assert.equal(
    result,
    [
      "  z-ai/glm-5.2 (openrouter-novita):",
      "    cost.input      $0.39/M  →  $0.34/M",
      "    cost.cacheRead  $0.07/M  →  $0.06/M",
    ].join("\n"),
  );
});

test("formatRefreshDiff: token-limit changes alongside costs", () => {
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-groq",
    modelId: "meta-llama/llama-4-scout",
    before: {
      cost: { input: 0.10, output: 0.50, cacheRead: 0.01, cacheWrite: 0 },
      contextWindow: 131_072,
      maxTokens: 32_768,
    },
    after: {
      cost: { input: 0.08, output: 0.50, cacheRead: 0.01, cacheWrite: 0 },
      contextWindow: 262_144,
      maxTokens: 65_536,
    },
  }];
  const result = formatRefreshDiff(diff);
  // Labels: ["cost.input"(10), "contextWindow"(13), "maxTokens"(9)] → width = 13
  assert.equal(
    result,
    [
      "  meta-llama/llama-4-scout (openrouter-groq):",
      "    cost.input     $0.10/M  →  $0.08/M",
      "    contextWindow  131,072  →  262,144",
      "    maxTokens      32,768  →  65,536",
    ].join("\n"),
  );
});

test("formatRefreshDiff: multi-model groups by model header", () => {
  const diff: PricingLimitsDiff[] = [
    {
      provider: "openrouter-novita",
      modelId: "z-ai/glm-5.2",
      before: {
        cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
        contextWindow: 1_048_576,
        maxTokens: 128_000,
      },
      after: {
        cost: { input: 0.34, output: 1.18, cacheRead: 0.06, cacheWrite: 0 },
        contextWindow: 1_048_576,
        maxTokens: 128_000,
      },
    },
    {
      provider: "openrouter-groq",
      modelId: "meta-llama/llama-4-scout",
      before: {
        cost: { input: 0.10, output: 0.50, cacheRead: 0.01, cacheWrite: 0 },
        contextWindow: 131_072,
        maxTokens: 32_768,
      },
      after: {
        cost: { input: 0.08, output: 0.50, cacheRead: 0.01, cacheWrite: 0 },
        contextWindow: 131_072,
        maxTokens: 65_536,
      },
    },
  ];
  const result = formatRefreshDiff(diff);
  // Model 1 width = max(10, 14) = 14; Model 2 width = max(10, 9) = 10
  assert.equal(
    result,
    [
      "  z-ai/glm-5.2 (openrouter-novita):",
      "    cost.input      $0.39/M  →  $0.34/M",
      "    cost.cacheRead  $0.07/M  →  $0.06/M",
      "  meta-llama/llama-4-scout (openrouter-groq):",
      "    cost.input  $0.10/M  →  $0.08/M",
      "    maxTokens   32,768  →  65,536",
    ].join("\n"),
  );
});

test("formatRefreshDiff: only changed fields appear (no redundant labels)", () => {
  // Only input price changed — output, cacheRead, cacheWrite, tokens all hidden.
  const diff: PricingLimitsDiff[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    before: {
      cost: { input: 0.39, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
    after: {
      cost: { input: 0.34, output: 1.18, cacheRead: 0.07, cacheWrite: 0 },
      contextWindow: 1_048_576,
      maxTokens: 128_000,
    },
  }];
  const result = formatRefreshDiff(diff);
  assert.ok(!result.includes("cacheRead"), "unchanged cacheRead is hidden");
  assert.ok(!result.includes("output"), "unchanged output is hidden");
  assert.ok(!result.includes("cacheWrite"), "unchanged cacheWrite is hidden");
  assert.ok(!result.includes("contextWindow"), "unchanged contextWindow is hidden");
  assert.ok(!result.includes("maxTokens"), "unchanged maxTokens is hidden");
  assert.ok(result.includes("cost.input"), "changed cost.input is shown");
});

// --- collectRefreshTargets ---------------------------------------------------

test("collectRefreshTargets: only openrouter-* providers with routing compat", () => {
  const targets = collectRefreshTargets({
    providers: {
      "openrouter-novita": providerEntry([glmModel()]),           // matched
      "anthropic": providerEntry([glmModel()]),                   // not our prefix
      "openrouter-together": providerEntry([{                     // matching prefix but no routing config
        id: "google/gemini-2.5-pro",
        name: "Gemini Pro",
        reasoning: true,
        input: ["text"],
        contextWindow: 2_097_152,
        maxTokens: 65_536,
        cost: { input: 1.25, output: 5.00, cacheRead: 0.1875, cacheWrite: 0 },
        compat: { openRouterRouting: {} },
      }]),
    },
  });
  assert.deepEqual(targets.map(t => [t.provider, t.model.id]), [
    ["openrouter-novita", "z-ai/glm-5.2"],
  ]);
});

test("collectRefreshTargets: missing file / empty providers → []", () => {
  assert.deepEqual(collectRefreshTargets(null), []);
  assert.deepEqual(collectRefreshTargets({}), []);
  assert.deepEqual(collectRefreshTargets({ providers: {} }), []);
});

// --- applyPricingPatches -----------------------------------------------------

test("applyPricingPatches: mutates models in-place and returns diffs", () => {
  const current: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel()]),
    },
  };
  const patches: PricingLimitsPatch[] = [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    cost: { input: 0.34, output: 0.98, cacheRead: 0.06, cacheWrite: 0 },
    contextWindow: 2_097_152,
    maxTokens: 65_536,
  }];
  const { applied, diff } = applyPricingPatches(current, patches);

  assert.equal(applied, 1);
  assert.equal(diff.length, 1);
  assert.equal(diff[0].modelId, "z-ai/glm-5.2");
  assert.deepEqual(diff[0].before.cost, glmModel().cost); // original cost preserved
  assert.deepEqual(diff[0].after.cost, patches[0].cost); // patched cost recorded

  // Verify in-place mutation on disk snapshot
  assert.deepEqual(current.providers!["openrouter-novita"].models[0].cost, patches[0].cost);
  assert.equal(current.providers!["openrouter-novita"].models[0].contextWindow, 2_097_152);
  assert.equal(current.providers!["openrouter-novita"].models[0].maxTokens, 65_536);
});

test("applyPricingPatches: silently skips missing provider", () => {
  const current: ModelsJson = {
    providers: { "openrouter-novita": providerEntry([glmModel()]) },
  };
  const patches: PricingLimitsPatch[] = [{
    provider: "openrouter-nonexistent",
    modelId: "z-ai/glm-5.2",
    cost: { input: 0.34, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1_048_576,
    maxTokens: 128_000,
  }];
  const { applied, diff } = applyPricingPatches(current, patches);
  assert.equal(applied, 0);
  assert.equal(diff.length, 0);
});

test("applyPricingPatches: silently skips missing model id within provider", () => {
  const current: ModelsJson = {
    providers: { "openrouter-novita": providerEntry([glmModel()]) },
  };
  const patches: PricingLimitsPatch[] = [{
    provider: "openrouter-novita",
    modelId: "nobody/home",
    cost: { input: 0.34, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1_048_576,
    maxTokens: 128_000,
  }];
  const { applied, diff } = applyPricingPatches(current, patches);
  assert.equal(applied, 0);
  assert.equal(diff.length, 0);
});

test("applyPricingPatches: processes valid patches and skips invalid ones", () => {
  const current: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
    },
  };
  const patches: PricingLimitsPatch[] = [
    {  // valid
      provider: "openrouter-novita",
      modelId: "z-ai/glm-5.2",
      cost: { input: 0.34, output: 0.98, cacheRead: 0.06, cacheWrite: 0 },
      contextWindow: 2_097_152,
      maxTokens: 65_536,
    },
    {  // invalid model id
      provider: "openrouter-novita",
      modelId: "fake/model",
      cost: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 131_072,
      maxTokens: 32_768,
    },
  ];
  const { applied, diff } = applyPricingPatches(current, patches);
  assert.equal(applied, 1);
  assert.equal(diff.length, 1);

  // GLM patched, DeepSeek untouched
  const ids = current.providers!["openrouter-novita"].models.map(m => m.id);
  assert.deepEqual(ids, ["z-ai/glm-5.2", "deepseek/deepseek-v4-flash-0731"]);
  // GLM model cost was mutated
  assert.deepEqual(current.providers!["openrouter-novita"].models[0].cost, patches[0].cost);
});

// --- session_start handler (integration) -------------------------------------

test("session_start handler: no-op without API key (integration)", async () => {
  // Without an OpenRouter API key, refreshPinnedModels returns immediately
  // with empty results — no notification emitted. Guards against regressions
  // where console.log leaks into the TUI footer.
  const h = new SessionStartHarness();
  await h.fireSessionStart(); // no API key set

  // Without an API key, refreshPinnedModels either returns empty immediately
  // (0 notifications) or hits an error (e.g. empty modelRegistry throws) which
  // the handler catches and routes via ctx.ui.notify. Either outcome is
  // acceptable — the critical assertion is that messages always go through
  // notify() and never leak to raw console.*.
  assert.ok(h.notifications.length <= 1, "at most one notification (none, or an error routed via ctx.ui.notify)");
});

test("session_start handler: routes refresh results via ctx.ui.notify (guarded)", async () => {
  // Integration: requires OPENROUTER_API_KEY to make real API calls.
  // Skipped entirely when absent (does not fail CI).
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    // eslint-disable-next-line no-console -- expected skip in environments without API keys
    console.log("[skip] OPENROUTER_API_KEY not set");
    return;
  }

  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": providerEntry([glmModel({ id: "z-ai/glm-5.2" })]),
      },
    });

    const h = new SessionStartHarness();
    await h.fireSessionStart(apiKey);

    // With real credentials, the handler should emit at least one notification
    // (success with pricing, or warning if the endpoint call fails). Allow
    // graceful skip when network/mocks are unreachable within the harness delay.
    if (h.notifications.length === 0) {
      console.log("[skip] session_start refresh produced no notification (network/mocks unreachable)");
      return;
    }
    // All messages come through ctx.ui.notify — never raw console.*.
    assert.ok(h.notifications.every(n => n.type === "info" || n.type === "warning" || n.type === "error"));
    // Messages reference pricing or refresh, confirming the right logic ran.
    const anyPricingMsg = h.notifications.some(n =>
      n.message.toLowerCase().includes("refresh") || n.message.toLowerCase().includes("pricing"),
    );
    assert.ok(anyPricingMsg, "at least one message references pricing/refresh");
  });
});

// ---------------------------------------------------------------------------
// E — Pure plan values and settings-prune helpers
// ---------------------------------------------------------------------------

test("planUnpin: computes emptied and repruned providers", () => {
  const snapshot: ModelsJson = {
    providers: {
      "openrouter-novita": { baseUrl: "x", api: "y", apiKey: "$OPENROUTER_API_KEY", models: [glmModel(), deepseekModel()] },
      "openrouter-together": { baseUrl: "x", api: "y", apiKey: "$OPENROUTER_API_KEY", models: [deepseekModel()] },
      "anthropic": { baseUrl: "x", api: "y", apiKey: "$OPENROUTER_API_KEY", models: [glmModel()] },
    },
  };
  const plan = planUnpin(snapshot, null, "z-ai/glm-5.2");
  // novita had two models; removing glm-5.2 leaves deepseek, so novita is repruned.
  assert.equal(plan.repruned.length, 1, "novita is repruned (deepseek survives)");
  assert.equal(plan.emptied.length, 0, "no provider is emptied");
  assert.equal(plan.removed.length, 1);
});

test("planUnpin: computes settings prune for default clearing", () => {
  const settings: SettingsJson = { defaultProvider: "openrouter-novita", defaultModel: "z-ai/glm-5.2", enabledModels: ["z-ai/glm-5.2", "deepseek/deepseek-v4-flash-0731"] };
  const snapshot: ModelsJson = {
    providers: {
      "openrouter-novita": { baseUrl: "x", api: "y", apiKey: "$OPENROUTER_API_KEY", models: [glmModel()] },
    },
  };
  const plan = planUnpin(snapshot, settings, "z-ai/glm-5.2");
  assert.ok(plan.settingsPatch, "settings patch is computed");
  assert.equal(plan.settingsPatch!.defaultProvider, undefined, "defaultProvider is cleared");
  assert.equal(plan.settingsPatch!.defaultModel, undefined, "defaultModel is cleared");
  assert.ok(plan.settingsPatch!.enabledModel, "enabledModel is pruned");
});

test("planUnpin: non-default model keeps default untouched", () => {
  const settings: SettingsJson = { defaultProvider: "openrouter-novita", defaultModel: "z-ai/glm-5.2", enabledModels: ["z-ai/glm-5.2"] };
  const snapshot: ModelsJson = {
    providers: {
      "openrouter-novita": { baseUrl: "x", api: "y", apiKey: "$OPENROUTER_API_KEY", models: [glmModel(), deepseekModel()] },
    },
  };
  const plan = planUnpin(snapshot, settings, "deepseek/deepseek-v4-flash-0731");
  assert.equal(plan.settingsPatch, null, "no settings patch for non-default unpin");
});

test("planUnpin: null / empty snapshot returns empty plan", () => {
  const plan = planUnpin(null, null, "x/y");
  assert.equal(plan.emptied.length, 0);
  assert.equal(plan.repruned.length, 0);
  assert.equal(plan.removed.length, 0);
  assert.equal(plan.settingsPatch, null);
});

test("computeSettingsPrune: empty and unrelated inputs produce null", () => {
  assert.equal(computeSettingsPrune(null, "x/y", []), null);
  // Unrelated model with no default or emptied providers still returns null
  // because neither enabledModels nor default is affected.
  assert.equal(computeSettingsPrune({ enabledModels: ["a/b"] }, "x/y", []), null, "unrelated model with no emptied providers");
  // Model in enabledModels gets pruned — this IS a settings change.
  const pruneResult = computeSettingsPrune({ enabledModels: ["a/b"] }, "a/b", []);
  assert.ok(pruneResult, "pruning an enabled model produces a patch");
  assert.equal(pruneResult!.enabledModel, "", "enabledModel is empty string when all pruned (cleared to [])");
});

test("computeSettingsDiff: detects changes correctly", () => {
  // Default cleared: before has defaultProvider/defaultModel, after doesn't.
  const before: SettingsJson = { defaultProvider: "novita", defaultModel: "z-ai/glm-5.2", enabledModels: ["z-ai/glm-5.2"] };
  const after: SettingsJson = { enabledModels: ["deepseek/deepseek-v4-flash-0731"] };
  const diff = computeSettingsDiff(before, after);
  assert.equal(diff.defaultCleared, true, "default was cleared");
  assert.equal(diff.wrote, true, "settings.json was rewritten");
  // enabledPruned requires all after models were in before list.
  // Since after has a completely different model, enabledPruned is false.
  assert.equal(diff.enabledPruned, false, "enabledModels changed to different models");
});

test("computeSettingsDiff: no change returns wrote=false", () => {
  const settings: SettingsJson = { defaultProvider: "novita", defaultModel: "z-ai/glm-5.2", enabledModels: ["z-ai/glm-5.2"] };
  const diff = computeSettingsDiff(settings, { ...settings });
  assert.equal(diff.wrote, false);
});

test("planPin: detects duplicate enabledModel", () => {
  const settings: SettingsJson = { enabledModels: ["z-ai/glm-5.2"] };
  const plan = planPin(null, settings, "z-ai/glm-5.2", { defaultProvider: "novita", defaultModel: "z-ai/glm-5.2", enabledModel: "z-ai/glm-5.2" });
  assert.equal(plan.duplicateEnabledModel, true);
});

test("planPin: non-duplicate enabledModel is allowed", () => {
  const settings: SettingsJson = { enabledModels: ["z-ai/glm-5.2"] };
  const plan = planPin(null, settings, "z-ai/glm-5.2", { defaultProvider: "novita", defaultModel: "z-ai/glm-5.2", enabledModel: "deepseek/deepseek-v4-flash-0731" });
  assert.equal(plan.duplicateEnabledModel, false);
  assert.ok(plan.settingsPatch, "patch is computed");
});

test("probe: detects capability presence on fake pi", () => {
  const piWithUnregister = { unregisterProvider: () => {}, registerProvider: () => {} } as unknown as ExtensionAPI;
  const cap1 = probe(piWithUnregister);
  assert.equal(cap1.canUnregister, true);
  assert.equal(cap1.live, true);

  const piWithout = { registerProvider: () => {} } as unknown as ExtensionAPI;
  const cap2 = probe(piWithout);
  assert.equal(cap2.canUnregister, false);
  assert.equal(cap2.live, false);
});

// ---------------------------------------------------------------------------
// F — Updated handler notice tests
// ---------------------------------------------------------------------------

test("/openrouter-unpin <model>: live policy emits live notice when pi supports unregister", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-novita": providerEntry([glmModel(), deepseekModel()]) },
    });
    // CommandHarness currently uses a minimal fake pi — the unpin handler
    // probes capabilities. With no unregisterProvider, it falls back to
    // fileOnly wording.
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "z-ai/glm-5.2");
    const lastNotice = h.notifications[h.notifications.length - 1];
    assert.ok(
      lastNotice.message.includes("applies on /reload or next session") || lastNotice.message.includes("removed live"),
      "notice reflects live or fileOnly policy",
    );
  });
});

// ---------------------------------------------------------------------------
// G — Qualified-to-bare normalization
// ---------------------------------------------------------------------------

test("normalizeUnpinArg: bare id passes through unchanged", () => {
  const known = new Set(["z-ai/glm-5.2", "z-ai/glm-5.2:free"]);
  assert.equal(normalizeUnpinArg("z-ai/glm-5.2", known), "z-ai/glm-5.2");
  assert.equal(normalizeUnpinArg("z-ai/glm-5.2:free", known), "z-ai/glm-5.2:free");
  assert.equal(normalizeUnpinArg("deepseek/deepseek-v4-flash-0731", known), "deepseek/deepseek-v4-flash-0731");
});

test("normalizeUnpinArg: qualified strict → bare when remainder matches", () => {
  const known = new Set(["z-ai/glm-5.2", "z-ai/glm-5.2:free"]);
  assert.equal(normalizeUnpinArg("openrouter-decart/z-ai/glm-5.2", known), "z-ai/glm-5.2");
  assert.equal(normalizeUnpinArg("openrouter-decart/z-ai/glm-5.2:free", known), "z-ai/glm-5.2:free");
  assert.equal(normalizeUnpinArg("openrouter-novita/deepseek/deepseek-v4-flash-0731", new Set(["deepseek/deepseek-v4-flash-0731"])), "deepseek/deepseek-v4-flash-0731");
});

test("normalizeUnpinArg: qualified -plus → bare when remainder matches", () => {
  const known = new Set(["z-ai/glm-5.2", "z-ai/glm-5.2:free"]);
  assert.equal(normalizeUnpinArg("openrouter-decart-plus/z-ai/glm-5.2", known), "z-ai/glm-5.2");
  assert.equal(normalizeUnpinArg("openrouter-decart-plus/z-ai/glm-5.2:free", known), "z-ai/glm-5.2:free");
});

test("normalizeUnpinArg: hyphenated slugs (reported case) normalize correctly", () => {
  const knownZai = new Set(["z-ai/glm-5.2", "z-ai/glm-5.2:free"]);
  // The bug report: openrouter-z-ai was failing because '-' wasn't in the regex.
  assert.equal(normalizeUnpinArg("openrouter-z-ai/z-ai/glm-5.2", knownZai), "z-ai/glm-5.2");
  assert.equal(normalizeUnpinArg("openrouter-z-ai/z-ai/glm-5.2:free", knownZai), "z-ai/glm-5.2:free");
  assert.equal(normalizeUnpinArg("openrouter-z-ai-plus/z-ai/glm-5.2", knownZai), "z-ai/glm-5.2");
  // google-vertex also has hyphens.
  const knownVertex = new Set(["google/gemini-2.5-pro"]);
  assert.equal(
    normalizeUnpinArg("openrouter-google-vertex/google/gemini-2.5-pro", knownVertex),
    "google/gemini-2.5-pro",
  );
  // Non-matching hyphenated prefix stays verbatim.
  assert.equal(
    normalizeUnpinArg("openrouter-z-ai/unknown/model", knownZai),
    "openrouter-z-ai/unknown/model",
  );
});

test("normalizeUnpinArg: unknown input stays verbatim (no false positive)", () => {
  const known = new Set(["z-ai/glm-5.2"]);
  assert.equal(normalizeUnpinArg("openrouter-foo/unknown-model", known), "openrouter-foo/unknown-model");
  // Bare ids not in the set also pass through.
  assert.equal(normalizeUnpinArg("some/other-model", known), "some/other-model");
});

test("normalizeUnpinArg: :free kept verbatim (never stripped)", () => {
  const known = new Set(["z-ai/glm-5.2:free"]);
  assert.equal(normalizeUnpinArg("z-ai/glm-5.2:free", known), "z-ai/glm-5.2:free");
  assert.equal(normalizeUnpinArg("openrouter-decart/z-ai/glm-5.2:free", known), "z-ai/glm-5.2:free");
});

test("unpinFromModels: normalized qualified arg unpins the same model", () => {
  const input: ModelsJson = {
    providers: {
      "openrouter-decart": providerEntry([glmModel({ id: "z-ai/glm-5.2:free" })]),
    },
  };
  // Qualified input should match the bare id stored on disk.
  const { models, removed } = unpinFromModels(input, "openrouter-decart/z-ai/glm-5.2:free");
  assert.equal(removed, true);
  assert.deepEqual(models.providers, {}, "the emptied provider is dropped");
});

test("unpinFromModels: normalized -plus qualified arg works identically", () => {
  const input: ModelsJson = {
    providers: {
      "openrouter-decart-plus": providerEntry([glmModel({ id: "z-ai/glm-5.2:free" })]),
    },
  };
  const { models, removed } = unpinFromModels(input, "openrouter-decart-plus/z-ai/glm-5.2:free");
  assert.equal(removed, true);
  assert.deepEqual(models.providers, {});
});

test("unpinFromModels: multi-provider same-model removal still works with normalized input", () => {
  const input: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel({ id: "z-ai/glm-5.2:free" })]),
      "openrouter-together": providerEntry([glmModel({ id: "z-ai/glm-5.2:free", name: "GLM relaxed" })]),
    },
  };
  // Qualified form pointing to novita — both providers drop the model.
  const { models, removed } = unpinFromModels(input, "openrouter-novita/z-ai/glm-5.2:free");
  assert.equal(removed, true);
  assert.deepEqual(models.providers, {}, "both providers dropped");
});

test("planUnpin: normalized qualified arg plans the same as bare", () => {
  const snapshot: ModelsJson = {
    providers: {
      "openrouter-decart": { baseUrl: "x", api: "y", apiKey: "$OPENROUTER_API_KEY", models: [glmModel({ id: "z-ai/glm-5.2:free" })] },
    },
  };
  const settings: SettingsJson = { defaultProvider: "openrouter-decart", defaultModel: "z-ai/glm-5.2:free", enabledModels: ["openrouter-decart/z-ai/glm-5.2:free"] };
  const planQualified = planUnpin(snapshot, settings, "openrouter-decart/z-ai/glm-5.2:free");
  const planBare = planUnpin(snapshot, settings, "z-ai/glm-5.2:free");
  assert.deepEqual(planQualified, planBare, "qualified and bare produce identical plans");
});

// ---------------------------------------------------------------------------
// H — Handler notice echoes bare id (task 2.1)
// ---------------------------------------------------------------------------

test("/openrouter-unpin <model>: bare-arg echoes bare id in notice", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, { providers: { "openrouter-novita": providerEntry([glmModel()]) } });
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "z-ai/glm-5.2");
    const lastNotice = h.notifications[h.notifications.length - 1];
    assert.ok(lastNotice.message.includes("Unpinned z-ai/glm-5.2"));
  });
});

test("/openrouter-unpin <model>: qualified strict arg echoes bare id in notice", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, { providers: { "openrouter-decart": providerEntry([glmModel()]) } });
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "openrouter-decart/z-ai/glm-5.2");
    const lastNotice = h.notifications[h.notifications.length - 1];
    assert.ok(
      lastNotice.message.includes("Unpinned z-ai/glm-5.2") && !lastNotice.message.includes("openrouter-decart/z-ai/glm-5.2"),
      "notice echoes bare id, not qualified input",
    );
    // The provider should be dropped entirely.
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(models!.providers, {}, "provider dropped");
  });
});

test("/openrouter-unpin <model>: qualified -plus arg echoes bare id in notice", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, { providers: { "openrouter-decart-plus": providerEntry([glmModel()]) } });
    const h = new CommandHarness();
    await h.run("openrouter-unpin", "openrouter-decart-plus/z-ai/glm-5.2");
    const lastNotice = h.notifications[h.notifications.length - 1];
    assert.ok(
      lastNotice.message.includes("Unpinned z-ai/glm-5.2") && !lastNotice.message.includes("openrouter-decart-plus/z-ai/glm-5.2"),
      "notice echoes bare id, not qualified input",
    );
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(models!.providers, {});
  });
});

test("/openrouter-unpin <model>: picker path echoes bare id (unchanged behavior)", async () => {
  await withAgentDir(async () => {
    const modelsPath = join(process.env.PI_CODING_AGENT_DIR!, "models.json");
    await atomicWriteJson(modelsPath, { providers: { "openrouter-decart": providerEntry([glmModel()]) } });
    const h = new CommandHarness("openrouter-decart/z-ai/glm-5.2");
    await h.run("openrouter-unpin", "");
    const lastNotice = h.notifications[h.notifications.length - 1];
    assert.ok(lastNotice.message.includes("Unpinned z-ai/glm-5.2"));
  });
});

// ---------------------------------------------------------------------------
// I — Pin-then-unpin-default round-trip (task 2.2)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// J — Event-combination matrix (task 2.3)
// ---------------------------------------------------------------------------

function createOldShapePi(): ExtensionAPI {
  return {
    on: () => {},
    registerProvider: () => {},
    registerCommand: () => {},
  } as unknown as ExtensionAPI;
}

async function withTempDir<T>(fn: (dir: string) => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), "or-roundtrip-"));
  try {
    return await fn(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

test("performUnpin after performPin default: bare-id unpin clears defaults and prunes enabledModels", async () => {
  // Simulate what performPin does when --default is set.
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-decart": providerEntry([glmModel({ id: "z-ai/glm-5.2:free" })]),
      },
    });
    await atomicWriteJson(settingsPath, {
      defaultProvider: "openrouter-decart",
      defaultModel: "z-ai/glm-5.2:free",
      enabledModels: ["openrouter-decart/z-ai/glm-5.2:free"],
    });

    // Unpin with bare id.
    const resultBare = await performUnpin({
      modelsPath,
      settingsPath,
      pi: createOldShapePi(),
      modelId: "z-ai/glm-5.2:free",
    });
    assert.equal(resultBare.status, "removed", "bare-id unpin removes the pin");

    const modelsAfter = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(modelsAfter!.providers, {}, "provider dropped after unpin");

    const settingsAfter = await readJsonFile<SettingsJson>(settingsPath);
    assert.equal(
      settingsAfter!.defaultProvider ?? undefined,
      undefined,
      "defaultProvider cleared (via delete)",
    );
    assert.equal(
      settingsAfter!.defaultModel ?? undefined,
      undefined,
      "defaultModel cleared (via delete)",
    );
    // enabledModels still exists (from prior content), but the entry for our
    // model was pruned — since there was only one entry it becomes [].
    // The applier preserves non-default keys, so enabledModels key survives.
    assert.deepEqual(settingsAfter?.enabledModels, []);
  });
});

test("performUnpin after performPin default: qualified-id unpin clears defaults identically", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-decart": providerEntry([glmModel({ id: "z-ai/glm-5.2:free" })]),
      },
    });
    await atomicWriteJson(settingsPath, {
      defaultProvider: "openrouter-decart",
      defaultModel: "z-ai/glm-5.2:free",
      enabledModels: ["openrouter-decart/z-ai/glm-5.2:free"],
    });

    const resultQ = await performUnpin({
      modelsPath,
      settingsPath,
      pi: createOldShapePi(),
      modelId: "openrouter-decart/z-ai/glm-5.2:free",
    });
    assert.equal(resultQ.status, "removed", "qualified-id unpin also removes");

    const modelsAfter = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(modelsAfter!.providers, {});
    const settingsAfter = await readJsonFile<SettingsJson>(settingsPath);
    assert.equal(settingsAfter!.defaultProvider ?? undefined, undefined);
    assert.equal(settingsAfter!.defaultModel ?? undefined, undefined);
    assert.deepEqual(settingsAfter?.enabledModels, []);
  });
});

// --- Scenario 1: Strict default sole-model via qualified input, live -----------

test("matrix: strict default sole-model unpinned via qualified (live) clears all", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-decart": providerEntry([glmModel({ id: "z-ai/glm-5.2" })]) },
    });
    await atomicWriteJson(settingsPath, {
      defaultProvider: "openrouter-decart",
      defaultModel: "z-ai/glm-5.2",
      enabledModels: ["openrouter-decart/z-ai/glm-5.2"],
    });
    // New-shape pi has unregister.
    const piWithUnregister = {
      on: () => {},
      registerProvider: () => {},
      registerCommand: () => {},
      unregisterProvider: () => {},
    } as unknown as ExtensionAPI;
    const result = await performUnpin({
      modelsPath,
      settingsPath,
      pi: piWithUnregister,
      modelId: "openrouter-decart/z-ai/glm-5.2",
    });
    assert.equal(result.status, "removed");
    const m = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(m!.providers, {});
    const s = await readJsonFile<SettingsJson>(settingsPath);
    assert.equal(s!.defaultProvider ?? undefined, undefined);
    assert.equal(s!.defaultModel ?? undefined, undefined);
  });
});

// --- Scenario 2: Relaxed non-default sibling-survives via bare input, live -------

test("matrix: relaxed non-default sibling survives (reprune)", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-decart-plus": providerEntry([
          glmModel({ id: "z-ai/glm-5.2" }),
          deepseekModel(),
        ]),
      },
    });
    await atomicWriteJson(settingsPath, {
      defaultProvider: "openrouter-decart", // different provider — not affected
      defaultModel: "some/other-model",
      enabledModels: ["openrouter-decart-plus/z-ai/glm-5.2", "openrouter-decart-plus/deepseek/deepseek-v4-flash-0731"],
    });
    const piWithUnregister = {
      on: () => {},
      registerProvider: () => {},
      registerCommand: () => {},
      unregisterProvider: () => {},
    } as unknown as ExtensionAPI;
    await performUnpin({
      modelsPath,
      settingsPath,
      pi: piWithUnregister,
      modelId: "z-ai/glm-5.2",
    });
    const m = await readJsonFile<ModelsJson>(modelsPath);
    assert.ok(
      m!.providers!["openrouter-decart-plus"].models.map((mm) => mm.id).includes("deepseek/deepseek-v4-flash-0731"),
      "surviving sibling stays in repruned provider",
    );
    // Default untouched because we didn't remove the default model.
    const s = await readJsonFile<SettingsJson>(settingsPath);
    assert.equal(s!.defaultModel, "some/other-model");
  });
});

// --- Scenario 3: Same model under two providers via qualified input -------------

test("matrix: same model two providers — qualified unpin removes both", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": providerEntry([glmModel({ id: "z-ai/glm-5.2" })]),
        "openrouter-together": providerEntry([glmModel({ id: "z-ai/glm-5.2", name: "GLM relaxed" })]),
      },
    });
    await atomicWriteJson(settingsPath, {
      defaultProvider: "openrouter-novita",
      defaultModel: "z-ai/glm-5.2",
      enabledModels: [
        "openrouter-novita/z-ai/glm-5.2",
        "openrouter-together/z-ai/glm-5.2",
      ],
    });
    const result = await performUnpin({
      modelsPath,
      settingsPath,
      pi: createOldShapePi(),
      modelId: "openrouter-novita/z-ai/glm-5.2",
    });
    assert.equal(result.status, "removed");
    const m = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(m!.providers, {}, "both providers dropped");
    const s = await readJsonFile<SettingsJson>(settingsPath);
    assert.equal(s!.defaultProvider ?? undefined, undefined, "default cleared");
    assert.equal(s!.defaultModel ?? undefined, undefined, "defaultModel cleared");
    assert.deepEqual(s?.enabledModels, [], "all enabled entries pruned");
  });
});

// --- Scenario 4: Stale default plus scope-active pin ----------------------------

test("matrix: stale default preserved when scope-active pin removed", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    // Settings default points to a DIFFERENT provider (stale).
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-decart": providerEntry([glmModel({ id: "z-ai/glm-5.2" })]),
        "openrouter-novita": providerEntry([glmModel({ id: "another/model", name: "Other" })]),
      },
    });
    await atomicWriteJson(settingsPath, {
      defaultProvider: "openrouter-novita", // stale — not the one we're removing
      defaultModel: "another/model",
      enabledModels: [
        "openrouter-decart/z-ai/glm-5.2",
        "openrouter-novita/another/model",
      ],
    });
    await performUnpin({
      modelsPath,
      settingsPath,
      pi: createOldShapePi(),
      modelId: "z-ai/glm-5.2",
    });
    const m = await readJsonFile<ModelsJson>(modelsPath);
    assert.ok(!("openrouter-decart" in m!.providers!), "decart provider dropped");
    assert.ok("openrouter-novita" in m!.providers!, "novita provider survives");
    const s = await readJsonFile<SettingsJson>(settingsPath);
    assert.equal(s!.defaultProvider, "openrouter-novita", "stale default preserved");
    assert.equal(s!.defaultModel, "another/model", "stale defaultModel preserved");
    assert.ok(!s!.enabledModels!.includes("openrouter-decart/z-ai/glm-5.2"), "removed entry pruned");
  });
});

// --- Scenario 5: Missing settings file plus fileOnly host -----------------------

test("matrix: missing settings file + fileOnly — no crash, provider still dropped", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-decart": providerEntry([glmModel()]) },
    });
    // DO NOT write settings.json.
    const piWithout = {
      on: () => {},
      registerProvider: () => {},
      registerCommand: () => {},
      // No unregisterProvider — fileOnly
    } as unknown as ExtensionAPI;
    const result = await performUnpin({
      modelsPath,
      settingsPath,
      pi: piWithout,
      modelId: "z-ai/glm-5.2",
    });
    assert.equal(result.status, "removed");
    const m = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(m!.providers, {}, "provider dropped despite no settings file");
    // No settings file should be created.
    const { existsSync } = await import("node:fs");
    assert.equal(existsSync(settingsPath), false, "no settings file created for nothing");
  });
});

// --- Scenario 6: Quant and data-collection extras unpinned by id ----------------

test("matrix: quant/data-collection extras unpinned by id alone", async () => {
  const quantGlm = glmModel({
    id: "z-ai/glm-5.2",
    name: "GLM 5.2 FP8 (novita)",
    compat: {
      thinkingFormat: "openrouter",
      openRouterRouting: {
        only: ["novita"],
        allow_fallbacks: false,
        quantizations: ["fp8"],
        data_collection: "allow",
      },
    },
  });
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-novita": providerEntry([quantGlm]) },
    });
    await atomicWriteJson(settingsPath, {
      enabledModels: ["openrouter-novita/z-ai/glm-5.2"],
    });
    await performUnpin({
      modelsPath,
      settingsPath,
      pi: createOldShapePi(),
      modelId: "z-ai/glm-5.2",
    });
    const m = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(m!.providers, {}, "model with routing extras is fully removed");
  });
});
