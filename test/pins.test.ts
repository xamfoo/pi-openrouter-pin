/**
 * Tests for the /openrouter-pins and /openrouter-unpin commands.
 *
 * Three layers:
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
import { formatRouting, listPins, performUnpin, unpinFromModels } from "../src/commands.ts";
import { atomicWriteJson, readJsonFile, type ModelsJson, type ProviderEntry } from "../src/files.ts";
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
      assert.deepEqual(await performUnpin(modelsPath, "z-ai/glm-5.2"), { status: "removed" });
      const models = await readJsonFile<ModelsJson>(modelsPath);
      assert.deepEqual(
        models!.providers!["openrouter-novita"].models.map((m) => m.id),
        ["deepseek/deepseek-v4-flash-0731"],
        "the sibling survives on disk",
      );

      // A second unpin of the last model drops the whole provider.
      assert.deepEqual(await performUnpin(modelsPath, "deepseek/deepseek-v4-flash-0731"), { status: "removed" });
      assert.deepEqual(await readJsonFile<ModelsJson>(modelsPath), { providers: {} });
    },
  );
});

test("performUnpin: not-found and no-providers never write", async () => {
  await withTempModels(
    { providers: { "openrouter-novita": providerEntry([glmModel()]) } },
    async (modelsPath) => {
      assert.deepEqual(await performUnpin(modelsPath, "nobody/home"), { status: "not-found" });
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
