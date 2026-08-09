/**
 * Tests for the interactive pinning wizard (src/wizard.ts).
 *
 * The wizard is driven exactly as pi would drive it: `runWizard` calls
 * `ctx.custom(factory)` and the factory hands back the TUI component. This
 * harness plays the TUI's role — it renders the component, feeds `handleInput`
 * key data ("tab", "enter", "escape", "down", plain characters), and captures
 * the `done()` result and every `ctx.notify` call. `performPin` runs for real
 * against temp models.json/settings.json files, so a full flow is an
 * end-to-end test of wizard → pin.
 *
 * These tests import wizard.ts, which pulls in @earendil-works/pi-coding-agent
 * and @earendil-works/pi-tui at runtime — run `npm install` (peer deps) first.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { initTheme } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI, ExtensionUIContext, ModelRegistry, Theme } from "@earendil-works/pi-coding-agent";
import { KeybindingsManager, TUI_KEYBINDINGS, type Component, type KeybindingsManager as TuiKeybindingsManager, type TUI } from "@earendil-works/pi-tui";
import type { EndpointValidation, OpenRouterClient } from "../src/api.ts";
import type { RawModelShape } from "../src/config.ts";
import { runWizard, type WizardResult } from "../src/wizard.ts";
import { readJsonFile, type ModelsJson, type SettingsJson } from "../src/files.ts";

// The wizard's list steps render SelectList items through pi's global theme
// (getSelectListTheme() reads module state) — initialize it like pi does at
// startup, so rendering works under plain `node --test`.
initTheme();

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const glmRaw = (over: Partial<RawModelShape> = {}): RawModelShape => ({
  id: "z-ai/glm-5.2",
  name: "Z.ai: GLM 5.2",
  context_length: 1_048_576,
  top_provider: { max_completion_tokens: 128_000 },
  pricing: { prompt: "0.1", completion: "0.31", input_cache_read: "0.02", input_cache_write: "0" },
  architecture: { input_modalities: ["text"] },
  supported_parameters: ["reasoning"],
  ...over,
});

const deepseekRaw = (): RawModelShape => ({
  id: "deepseek/deepseek-v4-flash-0731",
  name: "DeepSeek V4 Flash 0731",
});

// ---------------------------------------------------------------------------
// Harness: plays the TUI / extension-context side of runWizard
// ---------------------------------------------------------------------------

interface HarnessOptions {
  catalog?: RawModelShape[];
  catalogError?: string;
  endpoints?: Array<{ provider_name?: string; quantization?: string }>;
  endpointsMessage?: string;
  endpointsError?: string;
  userModels?: Set<string> | null;
  userModelsError?: string;
  apiKey?: string;
  fetchRawModel?: (id: string) => Promise<RawModelShape | null>;
  validateEndpoint?: (
    modelId: string,
    slug: string,
    quant: string | undefined,
    apiKey: string | undefined,
  ) => Promise<EndpointValidation>;
  /** When set, ctx.custom's factory throws — runWizard must report and not hang. */
  factoryError?: string;
}

/** Friendly key names → raw terminal data, as pi-tui's matchesKey() expects. */
const RAW_KEYS: Record<string, string> = {
  enter: "\r",
  tab: "\t",
  "shift+tab": "\x1b[Z",
  escape: "\x1b",
  backspace: "\x7f",
  up: "\x1b[A",
  down: "\x1b[B",
  "ctrl+k": "\x0b", // delete to line end (clears a prefill)
};

class Harness {
  readonly notifications: Array<{ message: string; type: "info" | "warning" | "error" }> = [];
  readonly doneValues: Array<WizardResult | null> = [];
  component: Component | null = null;
  started: Promise<void> | null = null;
  private readonly tui: TUI;
  private readonly client: OpenRouterClient;
  private readonly opts: HarnessOptions;

  constructor(opts: HarnessOptions = {}) {
    this.opts = opts;
    this.tui = { terminal: { rows: 40 }, requestRender: () => {} } as unknown as TUI;
    const theme = { fg: (_c: string, s: string) => s, bold: (s: string) => s } as unknown as Theme;
    // The real pi-tui manager with default bindings, exactly as pi injects it —
    // matches() expects raw terminal data (see RAW_KEYS).
    const keybindings = new KeybindingsManager(TUI_KEYBINDINGS) as unknown as TuiKeybindingsManager;

    this.client = {
      fetchCatalog: async () => {
        if (opts.catalogError) throw new Error(opts.catalogError);
        return opts.catalog ?? [];
      },
      fetchRawModel:
        opts.fetchRawModel ??
        (async (id: string) => (opts.catalog ?? []).find((m) => m.id === id) ?? null),
      fetchModelEndpoints: async (_id: string, apiKey?: string) => {
        if (opts.endpointsError) throw new Error(opts.endpointsError);
        if (!apiKey) return { endpoints: [], message: "no API key set — provider list not filtered" };
        return { endpoints: opts.endpoints ?? [], message: opts.endpointsMessage };
      },
      fetchUserModelIds: async () => {
        if (opts.userModelsError) throw new Error(opts.userModelsError);
        return opts.userModels ?? null;
      },
      validateEndpoint:
        opts.validateEndpoint ??
        // The real client echoes the (normalized) quant back; performPin builds
        // the pin from check.quant, so the fake must preserve it too.
        (async (_modelId: string, _slug: string, quant: string | undefined): Promise<EndpointValidation> => ({
          status: "ok",
          quant,
        })),
    } as unknown as OpenRouterClient;

    const modelRegistry = { getApiKeyForProvider: async () => opts.apiKey } as unknown as ModelRegistry;
    const pi = { registerProvider: () => {} } as unknown as ExtensionAPI;

    const ctx = {
      notify: (message: string, type: "info" | "warning" | "error" = "info") => {
        this.notifications.push({ message, type });
      },
      custom: <T>(
        factory: (
          tui: TUI,
          theme: Theme,
          keybindings: KeybindingsManager,
          done: (result: T) => void,
        ) => Component & { dispose?: () => void },
      ) => {
        return new Promise<T>((resolve, reject) => {
          if (opts.factoryError) {
            reject(new Error(opts.factoryError));
            return;
          }
          const ui = factory(this.tui, theme, keybindings, (result: T) => {
            this.doneValues.push(result as WizardResult | null);
            resolve(result);
          });
          this.component = ui;
        });
      },
    } as unknown as ExtensionUIContext;

    this.started = null;
    // keep a reference so start() can be called with per-test paths
    this.runWizardFn = (modelsPath, settingsPath) =>
      runWizard(modelsPath, settingsPath, pi, ctx, this.client, modelRegistry);
  }

  private runWizardFn: (modelsPath: string, settingsPath: string) => Promise<void>;

  start(modelsPath: string, settingsPath: string): Promise<void> {
    this.started = this.runWizardFn(modelsPath, settingsPath);
    return this.started;
  }

  press(data: string): void {
    this.component?.handleInput?.(RAW_KEYS[data] ?? data);
  }

  lines(width = 80): string[] {
    return this.component?.render(width) ?? [];
  }

  text(width = 80): string {
    return this.lines(width).join("\n");
  }
}

/** Flush microtask chains (catalog, endpoints, /models/user, raw model) plus a macrotask. */
async function settle(): Promise<void> {
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));
}

/** Pick the model tab's first match for a query, then let the provider checks settle. */
async function pickModel(h: Harness, query = "glm"): Promise<void> {
  h.press(query);
  h.press("enter");
  await settle();
}

/** Commit the current step by accepting the default (first) selection. */
async function pickFirst(h: Harness): Promise<void> {
  h.press("enter");
  await settle();
}

async function withTempDir(
  opts: HarnessOptions,
  fn: (modelsPath: string, settingsPath: string, h: Harness) => Promise<void>,
): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-wizard-"));
  const modelsPath = join(dir, "models.json");
  const settingsPath = join(dir, "settings.json");
  const h = new Harness(opts);
  try {
    await fn(modelsPath, settingsPath, h);
  } finally {
    // runWizard may still be mid-pin when an assertion above threw; give it a
    // bounded grace period before removing the dir, or the tmp-file rename
    // races rm(). A still-pending wizard (e.g. a test bug) must not hang the
    // whole suite, hence the race with a timeout.
    if (h.started) {
      await Promise.race([h.started.catch(() => {}), new Promise((r) => setTimeout(r, 2000))]);
    }
    for (let attempt = 0; ; attempt++) {
      try {
        await rm(dir, { recursive: true, force: true });
        break;
      } catch (err) {
        const code = (err as NodeJS.ErrnoException).code;
        if (attempt >= 5 || !code || !["ENOTEMPTY", "EBUSY", "EAGAIN"].includes(code)) throw err;
        await new Promise((r) => setTimeout(r, 25));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Cancel / navigation
// ---------------------------------------------------------------------------

test("Esc cancels the wizard without pinning or notifying", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      assert.ok(h.text().includes("Search & pick a model"), "model tab is shown first");
      h.press("escape");
      await started;
      assert.deepEqual(h.doneValues, [null]);
      assert.deepEqual(h.notifications, []);
      assert.equal(await readJsonFile<ModelsJson>(modelsPath), null, "nothing is written on cancel");
  });
});

test("Esc mid-flow cancels the wizard", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);
      await pickFirst(h); // provider → novita
      await pickFirst(h); // quant → none
      h.press("escape");
      await started;
      assert.deepEqual(h.doneValues, [null]);
      assert.equal(await readJsonFile<ModelsJson>(modelsPath), null);
  });
});

test("shift+tab clamps at the first step", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      h.press("shift+tab");
      await settle();
      assert.ok(h.text().includes("Search & pick a model"), "still on the first step");
      h.press("escape");
      await started;
  });
});

test("component invalidate() (theme change) rebuilds the active tab without crashing", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      h.press("glm");
      h.component!.invalidate();
      // The rebuilt model tab still renders its title and the ranked match.
      assert.ok(h.text().includes("Search & pick a model"));
      assert.ok(h.text().includes("z-ai/glm-5.2"));
      h.press("escape");
      await started;
  });
});

// ---------------------------------------------------------------------------
// Full flows: wizard → performPin → files
// ---------------------------------------------------------------------------

test("full flow: prefer routing + quant + default → -plus provider name, settings written", async () => {
  await withTempDir(
    {
      catalog: [glmRaw(), deepseekRaw()],
      endpoints: [{ provider_name: "novita", quantization: "fp8" }],
      userModels: new Set(["z-ai/glm-5.2"]),
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();

      // model tab: local ranking — "glm" surfaces the GLM entry, not deepseek
      assert.ok(h.text().includes("type to search OpenRouter models"), "empty query shows the hint");
      h.press("glm");
      assert.ok(h.text().includes("z-ai/glm-5.2"));
      assert.ok(!h.text().includes("deepseek-v4"));
      h.press("enter");
      await settle();

      // provider tab: breadcrumb shows the committed model; list step help
      assert.ok(h.text().includes("Model: z-ai/glm-5.2"), "breadcrumb reflects the chosen model");
      assert.ok(h.text().includes("type to filter • ↑↓ choose"), "list steps show the filter help");
      h.press("enter"); // novita
      await settle();

      h.press("fp8");
      h.press("enter"); // quant
      await settle();

      // name tab: prefilled from raw model + provider + quant; input-step help
      assert.ok(
        h.text(120).includes("Name: Z.ai: GLM 5.2 (novita fp8)"),
        "name is prefilled (breadcrumb reflects it)",
      );
      assert.ok(h.text().includes("tab/enter next • shift+tab back"), "input steps show the short help");
      h.press("enter"); // keep the prefill
      await settle();

      h.press("prefer");
      h.press("enter"); // routing → prefer
      await settle();
      h.press("enter"); // default → yes
      await settle();

      const result = h.doneValues[0];
      assert.deepEqual(result, {
        modelId: "z-ai/glm-5.2",
        slug: "novita",
        quant: "fp8",
        name: "Z.ai: GLM 5.2 (novita fp8)",
        isDefault: true,
        allowFallbacks: true,
        order: undefined,
        ignore: undefined,
        dataCollection: undefined,
    });
    // Post-finish input is inert: done() is called exactly once.
    h.press("enter");
    h.press("tab");
    assert.equal(h.doneValues.length, 1);

    await started;
    assert.ok(h.notifications.every((n) => n.type !== "error"), "no errors on the happy path");
    assert.deepEqual(
      h.notifications.map((n) => n.message),
      [
        "Pinning z-ai/glm-5.2 → openrouter-novita-plus…",
        "Pinned openrouter-novita-plus/z-ai/glm-5.2 and set as default.",
      ],
    );

    const models = await readJsonFile<ModelsJson>(modelsPath);
    const entry = models!.providers!["openrouter-novita-plus"];
    assert.ok(entry, "relaxed routing writes the -plus provider name");
    assert.equal(entry.models.length, 1);
    const m = entry.models[0];
    assert.equal(m.id, "z-ai/glm-5.2");
    assert.equal(m.name, "Z.ai: GLM 5.2 (novita fp8)");
    assert.equal(m.reasoning, true);
    assert.deepEqual(m.input, ["text"]);
    assert.equal(m.contextWindow, 1_048_576);
    assert.equal(m.maxTokens, 128_000);
    assert.deepEqual(m.cost, { input: 100_000, output: 310_000, cacheRead: 20_000, cacheWrite: 0 });
    assert.deepEqual(m.compat, {
      thinkingFormat: "openrouter",
      openRouterRouting: { allow_fallbacks: true, only: ["novita"], quantizations: ["fp8"] },
    });

    const settings = await readJsonFile<SettingsJson>(settingsPath);
    assert.deepEqual(settings, {
      defaultProvider: "openrouter-novita-plus",
      defaultModel: "z-ai/glm-5.2",
      enabledModels: ["openrouter-novita-plus/z-ai/glm-5.2"],
    });
  });
});

test("full flow: strict routing + custom name, no default → plain provider name, no settings file", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);
      await pickFirst(h); // provider → novita
      await pickFirst(h); // quant → none (provider default)

      assert.ok(
        h.text(120).includes("Name: Z.ai: GLM 5.2 (novita)"),
        "name prefill has no quant suffix",
      );
      // The prefill leaves the cursor at position 0; delete to line end, then type.
      h.press("ctrl+k");
      h.press("My Model");
      h.press("enter"); // name → custom value
      await settle();

      h.press("strict");
      h.press("enter"); // routing → strict
      await settle();
      h.press("down");
      h.press("enter"); // default → "No — pin only"
      await settle();

      assert.deepEqual(h.doneValues[0], {
        modelId: "z-ai/glm-5.2",
        slug: "novita",
        quant: undefined,
        name: "My Model",
        isDefault: false,
        allowFallbacks: false,
        order: undefined,
        ignore: undefined,
        dataCollection: undefined,
    });
    await started;

    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.ok(models!.providers!["openrouter-novita"], "strict pin uses the plain provider name");
    assert.equal(models!.providers!["openrouter-novita-plus"], undefined);
    assert.equal(models!.providers!["openrouter-novita"].models[0].name, "My Model");
    assert.deepEqual(models!.providers!["openrouter-novita"].models[0].compat.openRouterRouting, {
      allow_fallbacks: false,
      only: ["novita"],
    });
    assert.equal(await readJsonFile<SettingsJson>(settingsPath), null, "no default → no settings patch");
  });
});

test("custom routing inserts Order/Ignore/Data steps and builds the full result", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);
      await pickFirst(h); // provider
      await pickFirst(h); // quant → none
      await pickFirst(h); // name (prefill kept)

      h.press("custom");
      h.press("enter"); // routing → custom
      await settle();
      assert.ok(h.text().includes("Preferred order"), "custom routing inserts the Order step");
      h.press("Novita, Together");
      h.press("enter");
      await settle();
      assert.ok(h.text().includes("Exclude providers"), "Ignore step follows Order");
      h.press("openai, anthropic");
      h.press("enter");
      await settle();
      assert.ok(h.text().includes("Data collection"), "Data step follows Ignore");
      h.press("down");
      h.press("enter"); // dc → deny
      await settle();
      assert.ok(h.text().includes("Set as default?"), "Default step is last");
      h.press("down");
      h.press("enter"); // default → no
      await settle();

      assert.deepEqual(h.doneValues[0], {
        modelId: "z-ai/glm-5.2",
        slug: "novita",
        quant: undefined,
        name: "Z.ai: GLM 5.2 (novita)",
        isDefault: false,
        allowFallbacks: false,
        order: ["novita", "together"],
        ignore: ["openai", "anthropic"],
        dataCollection: "deny",
    });
    await started;
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.ok(models!.providers!["openrouter-novita-plus"], "order/ignore are relaxations → -plus name");
  });
});

// ---------------------------------------------------------------------------
// Provider step
// ---------------------------------------------------------------------------

test("provider step: loading state, deduped sorted list, breadcrumb", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [
        { provider_name: "together" },
        { provider_name: "novita" },
        { provider_name: "novita" }, // duplicate → deduped
      ],
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      h.press("glm");
      h.press("enter");
      // Synchronously after commit: endpoints have not resolved yet.
      assert.ok(h.text().includes("loading providers…"), "transient loading state");
      assert.ok(h.text().includes("Pick a provider for z-ai/glm-5.2"));
      await settle();

      assert.ok(h.text().includes("Pick a provider (2 serve z-ai/glm-5.2)"), "duplicates deduped");
      h.press("enter"); // first (sorted) provider → novita
      await settle();
      await pickFirst(h); // quant
      await pickFirst(h); // name
      h.press("strict");
      h.press("enter");
      await settle();
      h.press("enter"); // default → yes
      await settle();
      assert.equal(h.doneValues[0]?.slug, "novita", "sorted list commits the first provider");
      await started;
  });
});

test("provider step: custom… pick, invalid slug rejected, valid slug advances", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);

      h.press("custom");
      h.press("enter"); // pick "custom…" → custom input mode, does NOT advance
      await settle();
      assert.ok(h.text().includes("type a provider slug"), "custom mode shows its hint");
      assert.ok(h.text().includes("Provider for z-ai/glm-5.2"), "still on the provider step");

      h.press("Bad!");
      h.press("enter");
      await settle();
      assert.ok(h.text().includes('Invalid provider "Bad!"'), "invalid slug is reported");
      assert.ok(h.text().includes("Provider for z-ai/glm-5.2"), "invalid slug blocks the commit");

      for (let i = 0; i < 4; i++) h.press("backspace");
      h.press("myprov");
      h.press("enter");
      await settle();
      assert.ok(h.text().includes("Quantization (optional)"), "valid slug advances to the next step");

      await pickFirst(h); // quant
      await pickFirst(h); // name
      h.press("strict");
      h.press("enter");
      await settle();
      h.press("enter"); // default → yes
      await settle();
      assert.equal(h.doneValues[0]?.slug, "myprov");
      await started;
      const models = await readJsonFile<ModelsJson>(modelsPath);
      assert.ok(models!.providers!["openrouter-myprov"], "custom provider is pinned");
  });
});

test("provider step without an API key: forced custom input with the not-filtered note", async () => {
  const savedKey = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    await withTempDir(
      {
        catalog: [glmRaw()],
        endpoints: [{ provider_name: "novita" }],
        apiKey: undefined,
      },
      async (modelsPath, settingsPath, h) => {
        const started = h.start(modelsPath, settingsPath);
        await settle();
        await pickModel(h);
        // No key → endpoints empty + message → the wizard forces custom input.
        assert.ok(h.text().includes("Provider list not filtered"), "explains why the list is empty");
        assert.ok(h.text().includes("type a provider slug"));
        h.press("myprov");
        h.press("enter");
        await settle();
        assert.ok(h.text().includes("Quantization (optional)"), "typed slug commits");
        await pickFirst(h); // quant
        await pickFirst(h); // name
        h.press("strict");
        h.press("enter");
        await settle();
        h.press("enter"); // default → yes
        await settle();
        assert.equal(h.doneValues[0]?.slug, "myprov");
        await started;
        const models = await readJsonFile<ModelsJson>(modelsPath);
        assert.ok(models!.providers!["openrouter-myprov"], "pin is written without an API key");
    });
  } finally {
    if (savedKey !== undefined) process.env.OPENROUTER_API_KEY = savedKey;
    else delete process.env.OPENROUTER_API_KEY;
  }
});

test("provider custom mode resets to the list on re-entry when empty", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);

      h.press("custom");
      h.press("enter"); // custom mode, nothing typed
      await settle();
      assert.ok(h.text().includes("type a provider slug"));

      h.press("shift+tab"); // back to the model step
      await settle();
      h.press("enter"); // re-commit the same model (query is retained)
      await settle();
      assert.ok(h.text().includes("Pick a provider (1 serve z-ai/glm-5.2)"), "back to the provider list");
      assert.ok(!h.text().includes("type a provider slug"));
      h.press("escape");
      await started;
  });
});

test("provider custom mode with a typed slug stays custom across back-navigation", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);

      h.press("custom");
      h.press("enter"); // custom mode
      await settle();
      h.press("myprov"); // typed but not committed
      h.press("shift+tab"); // back to the model step
      await settle();
      h.press("enter"); // re-commit the model
      await settle();
      // The typed slug is still there, so custom mode is preserved.
      assert.ok(h.text().includes("type a provider slug"), "custom mode kept with a typed slug");
      h.press("enter"); // commit the typed slug
      await settle();
      assert.ok(h.text().includes("Quantization (optional)"), "typed slug commits on re-entry");
      h.press("escape");
      await started;
  });
});

// ---------------------------------------------------------------------------
// Model step
// ---------------------------------------------------------------------------

test("model step: no-match blocks commit, ranking is local and instant", async () => {
  await withTempDir(
    {
      catalog: [glmRaw(), deepseekRaw()],
      endpoints: [{ provider_name: "novita" }],
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();

      h.press("zzz");
      await settle();
      assert.ok(h.text().includes('No models match "zzz".'), "no-match message");
      h.press("enter");
      await settle();
      assert.ok(h.text().includes("Search & pick a model"), "commit is blocked without a selection");

      for (let i = 0; i < 3; i++) h.press("backspace");
      h.press("glm");
      await settle();
      assert.ok(h.text().includes("z-ai/glm-5.2"));
      assert.ok(!h.text().includes("deepseek-v4"), "ranking only returns matching models");
      h.press("enter");
      await settle();
      assert.ok(h.text().includes("Pick a provider"), "a match commits");
      h.press("escape");
      await started;
  });
});

test("model step: catalog failure surfaces inline and never crashes the wizard", async () => {
  await withTempDir(
    {
      catalog: [],
      catalogError: "network down",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      assert.ok(h.text().includes("model catalog unavailable (network down)"), "inline catalog note");
      h.press("glm");
      await settle();
      assert.ok(h.text().includes('No models match "glm".'));
      h.press("enter");
      await settle();
      assert.ok(h.text().includes("Search & pick a model"), "still on the model step");
      h.press("escape");
      await started;
      assert.deepEqual(h.notifications, []);
  });
});

// ---------------------------------------------------------------------------
// Name step
// ---------------------------------------------------------------------------

test("name step: user edits survive back-navigation (prefill never overwrites)", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);
      await pickFirst(h); // provider
      await pickFirst(h); // quant
      assert.ok(
        h.text(120).includes("Name: Z.ai: GLM 5.2 (novita)"),
        "name is prefilled (breadcrumb reflects it)",
      );

      // The prefill leaves the cursor at position 0; delete to line end, then type.
      h.press("ctrl+k");
      h.press("My Model");
      h.press("enter"); // → routing
      await settle();
      h.press("shift+tab"); // back to name
      await settle();
      assert.ok(h.text(120).includes("Name: My Model"), "the edited value is preserved");
      assert.ok(!h.text(120).includes("Name: Z.ai"), "prefill did not overwrite the edit");

      h.press("enter"); // → routing again
      await settle();
      h.press("strict");
      h.press("enter");
      await settle();
      h.press("enter"); // default → yes
      await settle();
      assert.equal(h.doneValues[0]?.name, "My Model");
      await started;
  });
});

// ---------------------------------------------------------------------------
// Background checks: /models/user heads-up
// ---------------------------------------------------------------------------

test("runWizard surfaces the /models/user heads-up before pinning", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
      userModels: new Set(["some/other-model"]), // GLM absent from the account
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);
      await pickFirst(h);
      await pickFirst(h);
      await pickFirst(h);
      h.press("prefer");
      h.press("enter");
      await settle();
      h.press("enter"); // default → yes
      await settle();
      await started;

      const messages = h.notifications.map((n) => n.message);
      assert.ok(messages[0]!.startsWith('Heads-up: "z-ai/glm-5.2" is absent'), "heads-up comes first");
      assert.equal(h.notifications[0]!.type, "warning");
      assert.ok(messages[1]!.startsWith("Pinning"), "the pin follows the heads-up");
      const models = await readJsonFile<ModelsJson>(modelsPath);
      assert.ok(models!.providers!["openrouter-novita-plus"], "the pin still lands");
  });
});

test("no heads-up when the model is present, or when /models/user is unavailable", async () => {
  for (const userModels of [new Set(["z-ai/glm-5.2"]), null]) {
    await withTempDir(
      {
        catalog: [glmRaw()],
        endpoints: [{ provider_name: "novita" }],
        userModels,
        apiKey: "test-key",
      },
      async (modelsPath, settingsPath, h) => {
        const started = h.start(modelsPath, settingsPath);
        await settle();
        await pickModel(h);
        await pickFirst(h);
        await pickFirst(h);
        await pickFirst(h);
        h.press("prefer");
        h.press("enter");
        await settle();
        h.press("enter");
        await settle();
        await started;
        assert.equal(
          h.notifications.some((n) => n.message.startsWith("Heads-up:")),
          false,
          `no heads-up for userModels=${userModels === null ? "null" : "present"}`,
        );
    });
  }
});

// ---------------------------------------------------------------------------
// Pin-time failures
// ---------------------------------------------------------------------------

test("pin-time failures surface as error notifications and write nothing", async () => {
  // Model vanished between the wizard and the pin: refused with a clear error.
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
      apiKey: "test-key",
      fetchRawModel: async () => null,
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);
      await pickFirst(h);
      await pickFirst(h);
      await pickFirst(h);
      h.press("prefer");
      h.press("enter");
      await settle();
      h.press("enter");
      await settle();
      await started;
      assert.ok(
        h.notifications.some((n) => n.type === "error" && n.message.includes('Model "z-ai/glm-5.2" not found')),
        "not-found error surfaces",
      );
      assert.equal(await readJsonFile<ModelsJson>(modelsPath), null, "nothing written");
  });

  // A throwing fetch is caught by runWizard and reported.
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpoints: [{ provider_name: "novita" }],
      apiKey: "test-key",
      fetchRawModel: async () => {
        throw new Error("fetch exploded");
      },
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);
      await pickFirst(h);
      await pickFirst(h);
      await pickFirst(h);
      h.press("prefer");
      h.press("enter");
      await settle();
      h.press("enter");
      await settle();
      await started;
      assert.ok(
        h.notifications.some((n) => n.type === "error" && n.message.includes("Pin failed: fetch exploded")),
        "pin failure is reported",
      );
  });
});

// ---------------------------------------------------------------------------
// Defensive error paths
// ---------------------------------------------------------------------------

test("endpoint and /models/user failures are swallowed — the flow still completes", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      endpointsError: "endpoints down",
      userModelsError: "user models down",
      apiKey: "test-key",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await settle();
      await pickModel(h);
      // Both background checks failed; the provider step still offers the
      // fallback custom entry and the flow is not blocked.
      assert.ok(h.text().includes("loading providers…"), "endpoint failure leaves the loading note");
      h.press("custom");
      h.press("enter"); // fall back to typing a provider
      await settle();
      h.press("myprov");
      h.press("enter");
      await settle();
      assert.ok(h.text().includes("Quantization (optional)"), "flow continues via custom provider");
      await pickFirst(h); // quant
      await pickFirst(h); // name
      h.press("strict");
      h.press("enter");
      await settle();
      h.press("enter"); // default → yes
      await settle();
      await started;
      assert.equal(h.doneValues[0]?.slug, "myprov");
      const models = await readJsonFile<ModelsJson>(modelsPath);
      assert.ok(models!.providers!["openrouter-myprov"], "pin lands despite background failures");
      assert.equal(
        h.notifications.some((n) => n.message.startsWith("Heads-up:")),
        false,
        "failed /models/user produces no heads-up",
      );
    },
  );
});

test("runWizard reports a component-construction failure instead of hanging", async () => {
  await withTempDir(
    {
      catalog: [glmRaw()],
      factoryError: "ui exploded",
    },
    async (modelsPath, settingsPath, h) => {
      const started = h.start(modelsPath, settingsPath);
      await started;
      assert.ok(
        h.notifications.some((n) => n.type === "error" && n.message.includes("Pin failed: ui exploded")),
        "construction failure surfaces as an error notification",
      );
      assert.equal(h.doneValues.length, 0);
      assert.equal(await readJsonFile<ModelsJson>(modelsPath), null, "nothing is written");
    },
  );
});
