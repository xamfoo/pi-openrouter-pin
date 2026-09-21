/**
 * Real-`ModelRuntime` compatibility: pinned providers re-registered with the
 * effective key are selectable.
 *
 * This file drives the *real* installed pi `ModelRuntime` (not a spy) through
 * the real extension factory via `PI_CODING_AGENT_DIR`. It asserts durable
 * integration behavior: after the factory re-registers an openrouter-* pin
 * with the effective key (env → auth.json), the provider is registered, the
 * model exists, and the model shows up in the available snapshot so the
 * `/model` picker does not filter it out.
 *
 * For spy-level unit coverage of the same logic (layers A–D), see
 * `factory-auth.test.ts`. This file is the real-runtime counterpart and keeps
 * its distance from test-to-test imports on purpose.
 *
 * Run: node --test --test-concurrency=1 test/runtime-compat.test.ts
 */
import { describe, test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { atomicWriteJson, readJsonFile, type ModelsJson, type ProviderEntry } from "../src/files.ts";
import type { ModelConfig } from "../src/config.ts";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import openrouterPinExtension from "../src/index.ts";

// `ModelRuntime` is not exported via the package's `exports` map (only the
// ESM main entry is), so we resolve it off the installed package's `import`
// condition and load the core module by absolute file URL. This is portable
// across installs and fail-loud: if the module can't be found, or the API the
// test relies on drifts, this throws immediately instead of silently skipping.
// Structural contract. The hard fail-loud check is only for `create` and
// the two methods that have existed on every 0.84.x–0.86.x minor
// (`registerProvider`, `getProvider`); remaining methods are
// version-varying and are probed softly so the suite can fall back to the
// file-only contract on 0.84.x rather than erroring the whole file.
// Mirrors `probe()` in `src/commands.ts:638-646`.
interface ModelRuntimeClass {
  create(options: { authPath: string; modelsPath: string; refreshOnCreate: boolean }): Promise<any>;
  prototype: {
    registerProvider?(providerId: string, config: unknown): void;
    getProvider?(providerId: string): unknown;
    getModel?(providerId: string, modelId: string): unknown;
    getRegisteredProviderConfig?(providerId: string): { apiKey?: string };
    getProviderAuthStatus?(providerId: string): { configured?: boolean; label?: string };
    getAvailableSnapshot?(): unknown[];
    refresh?(options: { allowNetwork: boolean }): Promise<void>;
    unregisterProvider?(providerId: string): void;
  };
}

const runtimeEntryUrl = import.meta.resolve("@earendil-works/pi-coding-agent");
const runtimeEntryPath = fileURLToPath(runtimeEntryUrl);
const runtimePath = join(dirname(runtimeEntryPath), "core", "model-runtime.js");
const { ModelRuntime } = (await import(
  pathToFileURL(runtimePath).href
)) as { ModelRuntime?: ModelRuntimeClass };

assert.ok(ModelRuntime, "ModelRuntime must be resolvable from the installed pi package");
assert.equal(typeof ModelRuntime.create, "function", "create must exist on ModelRuntime");
// Hard assert only for methods known to exist on 0.84.1, 0.85.1 and 0.86.1.
// Verified via unpacked tgz inspection (see plan step 2): these three ship
// `registerProvider`/`getProvider` on every minor; the rest are probed softly.
for (const method of ["registerProvider", "getProvider"] as const) {
  assert.equal(typeof ModelRuntime.prototype[method], "function", `${method} must exist on ModelRuntime`);
}
// Soft probe for version-varying methods: collect missing, warn, and let
// individual tests fall back to the file-only contract instead of failing
// the whole suite.
const VERSION_VARYING_METHODS = [
  "getModel",
  "getRegisteredProviderConfig",
  "getProviderAuthStatus",
  "getAvailableSnapshot",
  "refresh",
] as const;
const missingRuntimeMethods = VERSION_VARYING_METHODS.filter(
  (m) => typeof (ModelRuntime.prototype as Record<string, unknown>)[m] !== "function",
);
const hasFullLiveRuntime = missingRuntimeMethods.length === 0;
if (!hasFullLiveRuntime) {
  console.warn(
    `[runtime-compat] missing ModelRuntime methods: ${missingRuntimeMethods.join(", ")} — live assertions will be skipped, file-only contract will be checked`,
  );
}
// Non-optional alias for use in the harness/tests; the asserts above guarantee
// it is set (they throw at load time otherwise).
const ModelRuntimeApi = ModelRuntime as ModelRuntimeClass;

const makeProviderEntry = (models: ModelConfig[]): ProviderEntry => ({
  baseUrl: "https://openrouter.ai/api/v1",
  api: "openai-completions",
  apiKey: "$OPENROUTER_API_KEY",
  models,
});

const glmModel: ModelConfig = {
  id: "z-ai/glm-5.2",
  name: "GLM 5.2 (novita)",
  reasoning: true,
  input: ["text"],
  contextWindow: 1_048_576,
  maxTokens: 128_000,
  cost: { input: 100_000, output: 310_000, cacheRead: 20_000, cacheWrite: 0 },
  compat: { thinkingFormat: "openrouter", openRouterRouting: { only: ["novita"], allow_fallbacks: false } },
};

/** Shape of the bounded subset of a snapshot entry this suite inspects. */
interface SnapshotModel {
  provider: string;
  id: string;
}

const PROVIDER = "openrouter-novita";
const MODEL_ID = "z-ai/glm-5.2";

/**
 * Poll the runtime until the async refresh kicked off by `registerProvider`
 * (fire-and-forget `void this.refresh(...)`) settles — i.e. two consecutive
 * reads of the provider's auth status produce the same value. Bounded: fails
 * loudly rather than sleeping on a magic constant.
 *
 * On runtimes that lack `getProviderAuthStatus` (pre-0.84 shape) the poll is
 * skipped and a bounded sleep is used instead so the file-only fallback can
 * still be checked.
 */
async function waitForRegisterRefreshSettle(
  runtime: any,
  providerId: string,
  maxTries = 40,
  stepMs = 20,
): Promise<void> {
  if (typeof runtime.getProviderAuthStatus !== "function") {
    await new Promise((r) => setTimeout(r, stepMs * 4));
    return;
  }
  let prev = "unset";
  for (let i = 0; i < maxTries; i++) {
    await new Promise((r) => setTimeout(r, stepMs));
    const status = runtime.getProviderAuthStatus(providerId) as { configured?: boolean };
    const cur = String(JSON.stringify(status));
    if (cur === prev) return;
    prev = cur;
  }
  throw new Error(`ModelRuntime state did not settle for provider "${providerId}"`);
}

/**
 * Create a ModelRuntime backed by a temp agent dir, then run the extension
 * factory which should re-register the pinned `providerId` with the effective
 * key. Returns the runtime for assertions.
 */
async function createRuntimeWithExtension(
  dir: string,
  opts: { effectiveKeyEnv?: string; providerId?: string } = {},
): Promise<any> {
  const { effectiveKeyEnv, providerId = PROVIDER } = opts;
  const modelsPath = join(dir, "models.json");
  const authPath = join(dir, "auth.json");
  // Ensure PI_CODING_AGENT_DIR is set so getAgentDir() inside the factory
  // resolves to this temp dir.
  const savedAgentDir = process.env.PI_CODING_AGENT_DIR;
  const savedEnv = process.env.OPENROUTER_API_KEY;
  process.env.PI_CODING_AGENT_DIR = dir;
  if (effectiveKeyEnv !== undefined) process.env.OPENROUTER_API_KEY = effectiveKeyEnv;
  else delete process.env.OPENROUTER_API_KEY;

  try {
    // ModelRuntime must be created first (as pi does), then the extension
    // registers on top of it. The extension's factory is sync, but
    // ModelRuntime's availability refresh is async.
    const runtime = await ModelRuntimeApi.create({
      authPath,
      modelsPath,
      refreshOnCreate: false,
    });

    // Fake ExtensionAPI that delegates to the real runtime (as pi's loader does
    // via pendingProviderRegistrations → modelRuntime.registerProvider).
    const pi = {
      on: () => {},
      registerProvider: (name: string, config: unknown) => {
        runtime.registerProvider(name, config);
      },
      registerCommand: () => {},
    } as unknown as ExtensionAPI;

    openrouterPinExtension(pi);

    // Let the fire-and-forget refresh in registerProvider settle, then run an
    // explicit refresh that recomputes availability deterministically (as
    // createAgentSessionServices does). Both are version-varying: old
    // ModelRuntime (≤0.84 sync shape) may lack getProviderAuthStatus/refresh,
    // so guard and fall back to file-only verification in the tests.
    await waitForRegisterRefreshSettle(runtime, providerId);
    if (typeof runtime.refresh === "function") {
      const r = runtime.refresh({ allowNetwork: false });
      if (r instanceof Promise) await r;
    }

    return runtime;
  } finally {
    if (savedAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = savedAgentDir;
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
}

/** Create a temp agent dir, run `fn`, always remove it afterwards. */
async function withTempDir<T>(fn: (dir: string) => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), "rt-compat-"));
  try {
    return await fn(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

const pinnedModels = { providers: { [PROVIDER]: makeProviderEntry([glmModel]) } };

// All tests in this suite touch global state (PI_CODING_AGENT_DIR and
// OPENROUTER_API_KEY), so they must run serially (same reason as
// factory-auth.test.ts). The harness saves/restores env around each call.
describe("real ModelRuntime compatibility", { concurrency: 1 }, () => {
  test("pin via auth.json (no env) is selectable", () =>
    withTempDir(async (dir) => {
      // Write a pinned provider with placeholder key (as pin does)
      await atomicWriteJson(join(dir, "models.json"), pinnedModels);
      // Auth via auth.json, not env – the factory must resolve it
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { key: "sk-auth-from-file" } }));

      const runtime = await createRuntimeWithExtension(dir);

      // The provider must be registered (hard method, always available)
      const provider = runtime.getProvider(PROVIDER);
      assert.ok(provider, "openrouter-novita provider should be registered");

      // Version-varying live checks: if any are missing, fall back to the
      // file-only contract (placeholder preserved, provider still in file).
      // Mirrors probe() live vs fileOnly branching in src/commands.ts.
      if (
        typeof runtime.getModel !== "function" ||
        typeof runtime.getProviderAuthStatus !== "function" ||
        typeof runtime.getAvailableSnapshot !== "function" ||
        typeof runtime.getRegisteredProviderConfig !== "function"
      ) {
        const models = await readJsonFile<ModelsJson>(join(dir, "models.json"));
        assert.ok(models?.providers?.[PROVIDER], "provider should remain in models.json (file-only fallback)");
        assert.equal(
          models.providers[PROVIDER].apiKey,
          "$OPENROUTER_API_KEY",
          "placeholder preserved in file when live runtime incomplete",
        );
        assert.ok(
          Array.isArray(models.providers[PROVIDER].models) && models.providers[PROVIDER].models.length === 1,
          "pinned model should remain in models.json (file-only fallback)",
        );
        return;
      }

      // The model must exist
      const model = runtime.getModel(PROVIDER, MODEL_ID);
      assert.ok(model, "pinned model should exist in runtime");
      assert.deepStrictEqual(model?.compat?.openRouterRouting, { only: ["novita"], allow_fallbacks: false });

      // Crucial: it must be *available* (auth configured), otherwise the /model
      // picker filters it out and the user sees "No models match pattern"
      const authStatus = runtime.getProviderAuthStatus(PROVIDER);
      assert.equal(authStatus.configured, true, "provider should be configured via auth.json key");

      const available = (runtime.getAvailableSnapshot() as SnapshotModel[]).filter(
        (m) => m.provider === PROVIDER && m.id === MODEL_ID,
      );
      assert.equal(available.length, 1, "pinned model should be in available snapshot");

      // Also verify the registered config actually carries the effective key
      const registered = runtime.getRegisteredProviderConfig(PROVIDER);
      assert.equal(registered?.apiKey, "sk-auth-from-file", "registered provider should carry resolved key literal");
    }));

  test("env wins over auth.json", () =>
    withTempDir(async (dir) => {
      await atomicWriteJson(join(dir, "models.json"), pinnedModels);
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { key: "sk-auth-file" } }));

      const runtime = await createRuntimeWithExtension(dir, { effectiveKeyEnv: "  sk-env  " });

      if (
        typeof runtime.getProviderAuthStatus !== "function" ||
        typeof runtime.getRegisteredProviderConfig !== "function" ||
        typeof runtime.getAvailableSnapshot !== "function"
      ) {
        const models = await readJsonFile<ModelsJson>(join(dir, "models.json"));
        assert.ok(models?.providers?.[PROVIDER], "provider should remain in models.json (file-only fallback)");
        assert.equal(models.providers[PROVIDER].apiKey, "$OPENROUTER_API_KEY", "placeholder preserved in file when live runtime incomplete");
        return;
      }

      const authStatus = runtime.getProviderAuthStatus(PROVIDER);
      assert.equal(authStatus.configured, true);

      const registered = runtime.getRegisteredProviderConfig(PROVIDER);
      assert.equal(registered?.apiKey, "sk-env", "env should win over auth.json (trimmed)");

      const available = (runtime.getAvailableSnapshot() as SnapshotModel[]).filter((m) => m.provider === PROVIDER);
      assert.equal(available.length, 1);
    }));

  test("missing key source registers but stays unconfigured", () =>
    withTempDir(async (dir) => {
      await atomicWriteJson(join(dir, "models.json"), pinnedModels);
      // No auth.json, no env. Durable contract: the pin stays registered and
      // the model exists, but the `$OPENROUTER_API_KEY` placeholder is preserved
      // (no literal injected), so after a network-free refresh the provider is
      // reported unconfigured and excluded from the available snapshot.
      const runtime = await createRuntimeWithExtension(dir);
      const provider = runtime.getProvider(PROVIDER);
      assert.ok(provider, "provider still registered via models.json");

      // This test is the canonical file-only contract; even on a full runtime
      // we check both live and file. On a reduced runtime we at least check file.
      const modelsFile = await readJsonFile<ModelsJson>(join(dir, "models.json"));
      assert.ok(modelsFile?.providers?.[PROVIDER], "provider still in models.json (file-only)");
      assert.equal(modelsFile.providers[PROVIDER].apiKey, "$OPENROUTER_API_KEY", "no key source => placeholder preserved in file");

      if (typeof runtime.getModel !== "function" || typeof runtime.getRegisteredProviderConfig !== "function") {
        // Live checks unavailable — file assertion above is the fallback; skip the rest.
        if (typeof runtime.getAvailableSnapshot === "function") {
          const avail = (runtime.getAvailableSnapshot() as SnapshotModel[]).filter((m) => m.provider === PROVIDER);
          assert.equal(avail.length, 0, "unconfigured pin must be excluded from available snapshot (if snapshot available)");
        }
        return;
      }

      const model = runtime.getModel(PROVIDER, MODEL_ID);
      assert.ok(model, "model still registered even with no key source");

      const registered = runtime.getRegisteredProviderConfig(PROVIDER);
      assert.equal(registered?.apiKey, "$OPENROUTER_API_KEY", "no key source => placeholder preserved, not resolved");

      // After the explicit refresh({ allowNetwork:false }) already ran inside
      // the harness, the pin is reported unconfigured when no key is resolvable.
      // When a global OPENROUTER_API_KEY is present outside the harness, the
      // restored env may make the provider appear configured after the harness
      // returns, so we only assert the durable file-level contract here.
      if (typeof runtime.getAvailableSnapshot === "function") {
        const available = (runtime.getAvailableSnapshot() as SnapshotModel[]).filter((m) => m.provider === PROVIDER);
        assert.equal(available.length, 0, "unconfigured pin must be excluded from the available snapshot");
      }
    }));

  test("BOM + JSONC models.json registers", () =>
    withTempDir(async (dir) => {
      // Write models.json with a UTF-8 BOM and JSONC (//) comments – a compat
      // case for pi's JSONC-tolerant loader; our factory must parse the same way.
      // Built from the real fixture rather than hardcoding a 1KB string so the
      // intent stays diffable.
      const modelsWithBomAndComments = `\uFEFF// pinned via openrouter-pin\n${JSON.stringify(pinnedModels, null, 2)}`;
      writeFileSync(join(dir, "models.json"), modelsWithBomAndComments);
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { key: "sk-bom-test" } }));

      const runtime = await createRuntimeWithExtension(dir);

      const provider = runtime.getProvider(PROVIDER);
      assert.ok(provider, "BOM+JSONC models.json should still register provider");

      if (typeof runtime.getModel !== "function" || typeof runtime.getAvailableSnapshot !== "function") {
        // File-only fallback: BOM file should still have parsed correctly.
        const models = await readJsonFile<ModelsJson>(join(dir, "models.json"));
        assert.ok(models?.providers?.[PROVIDER], "BOM+JSONC provider should be in file (file-only fallback)");
        return;
      }

      const model = runtime.getModel(PROVIDER, MODEL_ID);
      assert.ok(model, "model should be present despite BOM/comments");
      assert.deepStrictEqual(model?.compat?.openRouterRouting, { only: ["novita"], allow_fallbacks: false });

      const available = (runtime.getAvailableSnapshot() as SnapshotModel[]).filter((m) => m.provider === PROVIDER);
      assert.equal(available.length, 1, "BOM file pin should be selectable when auth present");
    }));
});