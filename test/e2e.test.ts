/**
 * End-to-end tests against the PUBLISHED package, loaded through pi's real
 * extension loader (`discoverAndLoadExtensions` → package.json `pi.extensions`
 * → jiti), NOT re-imported from src. The whole point is exercising the actual
 * shipped artifact (`.ts` source, no dist) the way pi itself loads it.
 *
 * Gating: every test self-skips unless `CI_E2E=1` AND an `OPENROUTER_API_KEY`
 * is present — so plain `npm test` and regular CI never touch the network. The
 * only hardcoded network assumption is that OpenRouter's public catalog is
 * reachable when the gate is open; models/providers are discovered live so the
 * suite stays resilient to catalog churn.
 *
 * Each flow runs against a fresh temp `PI_CODING_AGENT_DIR` and a freshly
 * loaded extension instance (the factory calls `getAgentDir()` once at
 * construction), a fake `ExtensionCommandContext` (captured `ui.notify`, TUI
 * harness for `ui.custom` wired exactly like `wizard.test.ts`), and asserts on
 * the real persisted `models.json` / `settings.json`.
 *
 * The extension is loaded per flow — `discoverAndLoadExtensions` creates a
 * fresh jiti module graph per call (moduleCache: false), so the factory re-runs
 * with the current temp agent dir.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { createEventBus, discoverAndLoadExtensions, initTheme } from "@earendil-works/pi-coding-agent";
import type { ExtensionCommandContext, Theme } from "@earendil-works/pi-coding-agent";
import { KeybindingsManager, TUI_KEYBINDINGS, type Component, type KeybindingsManager as TuiKeybindingsManager, type TUI } from "@earendil-works/pi-tui";
import { CATALOG_CACHE_TTL_MS, ENDPOINT_CACHE_TTL_MS, OpenRouterClient } from "../src/api.ts";
import { findEndpoint, slugify, toPricingAndLimits, type RawEndpointShape, type RawModelShape } from "../src/config.ts";
import { readJsonFile, type ModelsJson, type SettingsJson } from "../src/files.ts";

// ---------------------------------------------------------------------------
// Gate: everything below is inert unless CI_E2E=1 + a key are present.
// ---------------------------------------------------------------------------

const E2E_ENABLED = process.env.CI_E2E === "1" && !!process.env.OPENROUTER_API_KEY;

// Restore proxy-aware HTTP dispatch. Importing @earendil-works/pi-coding-agent
// loads its nested undici (v8.9.0), whose lib/global.js installs a plain Agent
// into Symbol.for('undici.globalDispatcher.1') at import time — clobbering
// Node's proxy-aware EnvHttpProxyAgent. On hosts that require egress through an
// HTTP(S)_PROXY (e.g. sandboxed dev environments), fetch then connects directly
// and is denied (EPERM). Re-install a proxy-aware dispatcher so the suite honors
// the ambient proxy configuration. This is a no-op in CI (no proxy env vars).
if (process.env.HTTPS_PROXY || process.env.HTTP_PROXY) {
  try {
    const { setGlobalDispatcher, EnvHttpProxyAgent } = await import(
      new URL("../node_modules/@earendil-works/pi-coding-agent/node_modules/undici/index.js", import.meta.url).href,
    );
    setGlobalDispatcher(new EnvHttpProxyAgent({}));
  } catch {
    // If undici can't be resolved (different install layout), leave dispatch as
    // pi-coding-agent configured it; this only matters on proxy-gated hosts.
  }
}

/** The package under test. CI points this at the npm-installed artifact. */
function resolvePkgDir(): string {
  const fromEnv = process.env.ORPIN_PKG_DIR;
  if (fromEnv) return fromEnv;
  try {
    const require = createRequire(import.meta.url);
    return dirname(require.resolve("@xamfoo/pi-openrouter-pin/package.json"));
  } catch {
    // Local fallback (no installed copy): load the repo itself through pi's
    // loader — still the real loader path, just from the checkout instead of
    // node_modules. CI always sets ORPIN_PKG_DIR to the installed package.
    return process.cwd();
  }
}
const pkgDir = resolvePkgDir();
const repoRoot = process.cwd();

// ---------------------------------------------------------------------------
// Live discovery: a model + anchor provider + optional quant from the real
// catalog at test time, so the suite survives model/provider deprecation.
// ---------------------------------------------------------------------------

interface LiveTarget {
  modelId: string;
  /** Anchor provider slug (validated against the live endpoints list). */
  slug: string;
  /** A quantization that anchor provider actually serves, if any. */
  quant?: string;
  /** A second provider slug for relaxed routing; never equals `slug`. */
  otherSlug: string;
  raw: RawModelShape;
  endpoints: RawEndpointShape[];
}

const FALLBACK_OTHER_SLUGS = ["together", "deepinfra", "novita", "openai"];

let discoveryPromise: Promise<LiveTarget> | null = null;

async function discoverTarget(): Promise<LiveTarget> {
  if (discoveryPromise) return discoveryPromise;
  discoveryPromise = (async () => {
    const client = new OpenRouterClient(CATALOG_CACHE_TTL_MS, ENDPOINT_CACHE_TTL_MS);
    const catalog = await client.fetchCatalog();
    assert.ok(catalog.length > 0, "live OpenRouter catalog must not be empty (network or API change?)");
    let first: LiveTarget | null = null;
    let attempts = 0;
    for (const raw of catalog) {
      if (attempts++ >= 30) break;
      const { endpoints, message } = await client.fetchModelEndpoints(raw.id, process.env.OPENROUTER_API_KEY);
      if (message || endpoints.length === 0) continue;
      const slugs = [...new Set(endpoints.map((e) => slugify(e.provider_name ?? "")).filter(Boolean))];
      if (slugs.length === 0) continue;
      const slug = slugs[0];
      const quantEndpoint = endpoints.find((e) => slugify(e.provider_name ?? "") === slug && e.quantization);
      const target: LiveTarget = {
        modelId: raw.id,
        slug,
        quant: quantEndpoint?.quantization,
        otherSlug: slugs.find((s) => s !== slug) ?? FALLBACK_OTHER_SLUGS.find((s) => s !== slug) ?? "other-provider",
        raw,
        endpoints,
      };
      if (quantEndpoint) {
        console.log(`[openrouter-pin e2e] live pair: ${target.modelId} @ ${target.slug} (quant=${target.quant})`);
        return target; // prefer a provider with an explicit quantization
      }
      first ??= target;
    }
    assert.ok(first, "no model with a serving endpoint found in the live catalog");
    console.log(`[openrouter-pin e2e] live pair: ${first.modelId} @ ${first.slug} (no quantization endpoints)`);
    return first;
  })();
  return discoveryPromise;
}

// ---------------------------------------------------------------------------
// Loader helper: load the package under test through pi's real loader with a
// fresh temp PI_CODING_AGENT_DIR (the factory captures getAgentDir() once).
// ---------------------------------------------------------------------------

type CommandHandler = (args: string, ctx: ExtensionCommandContext) => Promise<void>;

interface LoadedExtension {
  pin: CommandHandler;
  unpin: CommandHandler;
  pins: CommandHandler;
  extensionPath: string;
}

async function loadExtension(agentDir: string): Promise<LoadedExtension> {
  const previous = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = agentDir;
  try {
    const result = await discoverAndLoadExtensions([pkgDir], repoRoot, agentDir, createEventBus());
    assert.deepEqual(result.errors, [], `package must load through pi's loader without errors (pkgDir=${pkgDir})`);
    const extension = result.extensions.find(
      (e) => e.path.endsWith("src/index.ts") || e.resolvedPath.includes("pi-openrouter-pin"),
    );
    assert.ok(extension, "expected the openrouter-pin extension to be loaded");
    const pin = extension.commands.get("openrouter-pin");
    const unpin = extension.commands.get("openrouter-unpin");
    const pins = extension.commands.get("openrouter-pins");
    assert.ok(pin && unpin && pins, "all three commands must be registered by the loaded extension");
    return { pin: pin.handler, unpin: unpin.handler, pins: pins.handler, extensionPath: extension.path };
  } finally {
    if (previous === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = previous;
  }
}

// ---------------------------------------------------------------------------
// Harness: plays the TUI / extension-context side, exactly like wizard.test.ts
// (real KeybindingsManager with TUI_KEYBINDINGS, fake TUI/theme, captured
// ui.notify, ui.custom resolved by the wizard's done() callback).
// ---------------------------------------------------------------------------

interface Notify {
  message: string;
  type: "info" | "warning" | "error";
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

class CommandHarness {
  readonly notifications: Notify[] = [];
  readonly doneValues: unknown[] = [];
  component: Component | null = null;
  private readonly tui: TUI;
  private readonly theme: Theme;
  private readonly keybindings: KeybindingsManager;

  constructor() {
    this.tui = { terminal: { rows: 40 }, requestRender: () => {} } as unknown as TUI;
    this.theme = { fg: (_c: string, s: string) => s, bold: (s: string) => s } as unknown as Theme;
    // The real pi-tui manager with default bindings, exactly as pi injects it.
    this.keybindings = new KeybindingsManager(TUI_KEYBINDINGS) as unknown as TuiKeybindingsManager;
  }

  makeCtx(): ExtensionCommandContext {
    return {
      mode: "tui",
      hasUI: true,
      cwd: repoRoot,
      modelRegistry: {
        // pi's auth machinery would resolve the env var; return it directly.
        getApiKeyForProvider: async () => process.env.OPENROUTER_API_KEY,
      },
      model: undefined,
      scopedModels: [],
      ui: {
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
          return new Promise<T>((resolve) => {
            const ui = factory(this.tui, this.theme, this.keybindings, (result: T) => {
              this.doneValues.push(result);
              resolve(result);
            });
            this.component = ui;
          });
        },
        select: async () => undefined,
        confirm: async () => false,
        input: async () => undefined,
      },
    } as unknown as ExtensionCommandContext;
  }

  press(data: string): void {
    this.component?.handleInput?.(RAW_KEYS[data] ?? data);
  }

  text(width = 80): string {
    return this.component?.render(width).join("\n") ?? "";
  }
}

/** Poll `cond` until true or `timeoutMs` elapses (network-backed UI states). */
async function waitFor(cond: () => boolean, what: string, timeoutMs = 60_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (cond()) return;
    if (Date.now() > deadline) throw new Error(`timed out waiting for: ${what}`);
    await new Promise((r) => setTimeout(r, 100));
  }
}

async function withAgentDir<T>(fn: (agentDir: string) => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-e2e-"));
  try {
    return await fn(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

const strictProvider = (slug: string) => `openrouter-${slugify(slug)}`;
const relaxedProvider = (slug: string) => `${strictProvider(slug)}-plus`;

const assertNoErrorNotifications = (h: CommandHarness, ctx: string): void => {
  assert.ok(
    h.notifications.every((n) => n.type !== "error"),
    `${ctx}: unexpected error notifications: ${JSON.stringify(h.notifications)}`,
  );
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test("e2e: live discovery picks a model with a serving endpoint from the catalog", { skip: !E2E_ENABLED, timeout: 120_000 }, async () => {
  const target = await discoverTarget();
  assert.ok(target.modelId, "a model id must be chosen");
  assert.ok(target.slug, "an anchor provider slug must be chosen");
  assert.ok(target.endpoints.length > 0, "the model must have serving endpoints");
  assert.ok(
    target.endpoints.some((e) => slugify(e.provider_name ?? "") === target.slug),
    `the anchor provider ${target.slug} must serve ${target.modelId}`,
  );
  assert.notEqual(target.otherSlug, target.slug, "the relaxed-routing second slug must differ from the anchor");
});

test("e2e: non-interactive strict pin with --default writes openrouter-<slug>, only=[slug], and default settings", { skip: !E2E_ENABLED, timeout: 180_000 }, async () => {
  const target = await discoverTarget();
  await withAgentDir(async (agentDir) => {
    const loaded = await loadExtension(agentDir);
    const h = new CommandHarness();
    await loaded.pin(`${target.modelId} ${target.slug} --default`, h.makeCtx());
    assertNoErrorNotifications(h, `strict pin (${target.modelId} @ ${target.slug})`);

    const providerName = strictProvider(target.slug);
    const models = await readJsonFile<ModelsJson>(join(agentDir, "models.json"));
    const entry = models?.providers?.[providerName];
    assert.ok(
      entry,
      `provider ${providerName} must exist (chosen pair: ${target.modelId} @ ${target.slug})`,
    );
    assert.equal(entry.models.length, 1);
    const m = entry.models[0];
    assert.equal(m.id, target.modelId);
    assert.deepEqual(m.compat.openRouterRouting, {
      allow_fallbacks: false,
      only: [slugify(target.slug)],
    });
    assert.equal(models?.providers?.[relaxedProvider(target.slug)], undefined, "strict pin must not create a -plus provider");

    const settings = await readJsonFile<SettingsJson>(join(agentDir, "settings.json"));
    assert.deepEqual(settings, {
      defaultProvider: providerName,
      defaultModel: target.modelId,
      enabledModels: [`${providerName}/${target.modelId}`],
    });
    assert.ok(
      h.notifications.some((n) => n.message.includes(`Pinned ${providerName}/${target.modelId}`)),
      "the success notification names the provider",
    );
  });
});

test("e2e: non-interactive relaxed pin (--quant --order) writes -plus, order, and endpoint-sourced pricing", { skip: !E2E_ENABLED, timeout: 180_000 }, async () => {
  const target = await discoverTarget();
  const quant = target.quant;
  const args = [`${target.modelId} ${target.slug}`];
  if (quant) args.push(`--quant ${quant}`);
  args.push(`--order ${target.otherSlug}`);
  await withAgentDir(async (agentDir) => {
    const loaded = await loadExtension(agentDir);
    const h = new CommandHarness();
    await loaded.pin(args.join(" "), h.makeCtx());
    assertNoErrorNotifications(h, `relaxed pin (${target.modelId} @ ${target.slug} quant=${quant ?? "none"})`);

    const providerName = relaxedProvider(target.slug);
    const models = await readJsonFile<ModelsJson>(join(agentDir, "models.json"));
    const entry = models?.providers?.[providerName];
    assert.ok(
      entry,
      `provider ${providerName} must exist (chosen pair: ${target.modelId} @ ${target.slug} quant=${quant ?? "none"})`,
    );
    assert.equal(models?.providers?.[strictProvider(target.slug)], undefined, "a relaxed pin must not create a plain provider");
    const m = entry.models[0];
    assert.equal(m.id, target.modelId);

    const routing = m.compat.openRouterRouting;
    assert.equal(routing.allow_fallbacks, true);
    assert.deepEqual(routing.order, [slugify(target.slug), slugify(target.otherSlug)]);
    if (quant) {
      assert.equal(
        routing.quantizations?.[0]?.toLowerCase(),
        quant.toLowerCase(),
        "the stored quantization matches the served one (case-insensitive)",
      );
    }

    // Endpoint-sourced pricing: the pin must record the anchor provider's
    // endpoint prices/limits, computed with the same conversion the package
    // uses (toPricingAndLimits), not the catalog aggregate.
    const expected = toPricingAndLimits(target.raw, findEndpoint(target.endpoints, target.slug, quant));
    assert.deepEqual(m.cost, expected.cost, "stored cost must equal endpoint-sourced cost");
    assert.equal(m.contextWindow, expected.contextWindow, "stored contextWindow must equal the endpoint's");
    assert.equal(m.maxTokens, expected.maxTokens, "stored maxTokens must equal the endpoint's");

    // Where applicable, the endpoint price differs from the catalog aggregate.
    const aggregate = toPricingAndLimits(target.raw, undefined);
    if (JSON.stringify(aggregate.cost) !== JSON.stringify(expected.cost)) {
      assert.notDeepEqual(m.cost, aggregate.cost, "endpoint pricing must not be the catalog aggregate when they differ");
    }

    assert.equal(await readJsonFile<SettingsJson>(join(agentDir, "settings.json")), null, "no --default → no settings file");
    assert.ok(
      h.notifications.some((n) => n.message.includes(`Pinned ${providerName}/${target.modelId}`)),
      "the success notification names the -plus provider",
    );
  });
});

test("e2e: interactive wizard strict pin with a custom name, not default → plain provider, only=[slug], no settings file", { skip: !E2E_ENABLED, timeout: 180_000 }, async () => {
  initTheme();
  const target = await discoverTarget();
  await withAgentDir(async (agentDir) => {
    const loaded = await loadExtension(agentDir);
    const h = new CommandHarness();
    const started = loaded.pin("", h.makeCtx());

    // Model tab: type the full id and wait for the live catalog match.
    h.press(target.modelId);
    await waitFor(
      () => h.text().includes(target.modelId) && !h.text().includes("No models match"),
      `model ${target.modelId} to appear as a ranked match`,
    );
    h.press("enter");

    // Provider tab: wait for the live endpoint list, then filter to the anchor.
    await waitFor(
      () => h.text().includes("Pick a provider") && !h.text().includes("loading providers"),
      "provider list to load",
    );
    h.press(target.slug);
    h.press("enter"); // commits the filtered (anchor) provider

    h.press("enter"); // quant → none (provider default)

    // Name tab: replace the prefill with a custom name.
    h.press("ctrl+k");
    h.press("E2E Custom Name");
    h.press("enter");

    h.press("strict");
    h.press("enter"); // routing → strict

    h.press("down");
    h.press("enter"); // default → "No — pin only"

    await started;
    assertNoErrorNotifications(h, `interactive strict pin (${target.modelId} @ ${target.slug})`);

    const providerName = strictProvider(target.slug);
    const models = await readJsonFile<ModelsJson>(join(agentDir, "models.json"));
    const entry = models?.providers?.[providerName];
    assert.ok(entry, `provider ${providerName} must exist (chosen pair: ${target.modelId} @ ${target.slug})`);
    const m = entry.models[0];
    assert.equal(m.id, target.modelId);
    assert.equal(m.name, "E2E Custom Name", "the custom display name is persisted");
    assert.deepEqual(m.compat.openRouterRouting, {
      allow_fallbacks: false,
      only: [slugify(target.slug)],
    });
    assert.equal(await readJsonFile<SettingsJson>(join(agentDir, "settings.json")), null, "no default → no settings file");
  });
});

test("e2e: interactive wizard custom routing (order/ignore/data) writes -plus with order/ignore/data_collection", { skip: !E2E_ENABLED, timeout: 180_000 }, async () => {
  initTheme();
  const target = await discoverTarget();
  await withAgentDir(async (agentDir) => {
    const loaded = await loadExtension(agentDir);
    const h = new CommandHarness();
    const started = loaded.pin("", h.makeCtx());

    h.press(target.modelId);
    await waitFor(
      () => h.text().includes(target.modelId) && !h.text().includes("No models match"),
      `model ${target.modelId} to appear as a ranked match`,
    );
    h.press("enter");

    await waitFor(
      () => h.text().includes("Pick a provider") && !h.text().includes("loading providers"),
      "provider list to load",
    );
    h.press(target.slug);
    h.press("enter"); // provider → anchor slug

    h.press("enter"); // quant → none
    h.press("enter"); // name → generated (prefill or derived)

    h.press("custom");
    h.press("enter"); // routing → custom (inserts Order/Ignore/Data steps)

    h.press(`${target.slug}, ${target.otherSlug}`);
    h.press("enter"); // order
    h.press("openai, anthropic");
    h.press("enter"); // ignore
    h.press("down");
    h.press("enter"); // data → deny
    h.press("down");
    h.press("enter"); // default → "No — pin only"

    await started;
    assertNoErrorNotifications(h, `interactive custom-routing pin (${target.modelId} @ ${target.slug})`);

    const providerName = relaxedProvider(target.slug);
    const models = await readJsonFile<ModelsJson>(join(agentDir, "models.json"));
    const entry = models?.providers?.[providerName];
    assert.ok(entry, `provider ${providerName} must exist (chosen pair: ${target.modelId} @ ${target.slug})`);
    const m = entry.models[0];
    assert.equal(m.id, target.modelId);
    assert.ok(m.name.endsWith(`(${slugify(target.slug)})`), `generated name ends with the anchor provider (got "${m.name}")`);

    const routing = m.compat.openRouterRouting;
    assert.equal(routing.allow_fallbacks, true);
    assert.deepEqual(routing.order, [slugify(target.slug), slugify(target.otherSlug)]);
    assert.deepEqual(routing.ignore, ["openai", "anthropic"]);
    assert.equal(routing.data_collection, "deny");
    assert.equal(await readJsonFile<SettingsJson>(join(agentDir, "settings.json")), null, "no default → no settings file");
  });
});

test("e2e: unpin round-trip — pin strict, list it, /openrouter-unpin prunes the provider, list reflects the change", { skip: !E2E_ENABLED, timeout: 180_000 }, async () => {
  const target = await discoverTarget();
  await withAgentDir(async (agentDir) => {
    const loaded = await loadExtension(agentDir);
    const modelsPath = join(agentDir, "models.json");
    const providerName = strictProvider(target.slug);

    const h = new CommandHarness();
    await loaded.pin(`${target.modelId} ${target.slug} --default`, h.makeCtx());
    assertNoErrorNotifications(h, `pre-unpin pin (${target.modelId} @ ${target.slug})`);
    const before = await readJsonFile<ModelsJson>(modelsPath);
    assert.ok(before?.providers?.[providerName], "pin landed before the unpin");

    // /openrouter-pins reflects the pin.
    const hList = new CommandHarness();
    await loaded.pins("", hList.makeCtx());
    assert.ok(
      hList.notifications[0]?.message.includes(providerName),
      `/openrouter-pins lists the pin (got: ${hList.notifications[0]?.message})`,
    );

    // Unpin.
    const hUnpin = new CommandHarness();
    await loaded.unpin(target.modelId, hUnpin.makeCtx());
    assert.ok(
      hUnpin.notifications.some((n) => n.message.includes("Unpinned")),
      "unpin reports success",
    );
    const after = await readJsonFile<ModelsJson>(modelsPath);
    assert.equal(after?.providers?.[providerName], undefined, "the provider is pruned from models.json");

    // /openrouter-pins reflects the removal.
    const hAfter = new CommandHarness();
    await loaded.pins("", hAfter.makeCtx());
    assert.ok(
      !hAfter.notifications[0]?.message.includes(providerName),
      `/openrouter-pins no longer lists the pin (got: ${hAfter.notifications[0]?.message})`,
    );
  });
});
