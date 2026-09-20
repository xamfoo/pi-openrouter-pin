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
import { atomicWriteJson, type ProviderEntry } from "../src/files.ts";
import type { ModelConfig } from "../src/config.ts";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import openrouterPinExtension from "../src/index.ts";

// `ModelRuntime` is not exported via the package's `exports` map (only the
// ESM main entry is), so we resolve it off the installed package's `import`
// condition and load the core module by absolute file URL. This is portable
// across installs and fail-loud: if the module can't be found, or the API the
// test relies on drifts, this throws immediately instead of silently skipping.
// Structural contract that the fail-loud checks below enforce at load time.
interface ModelRuntimeClass {
  create(options: { authPath: string; modelsPath: string; refreshOnCreate: boolean }): Promise<any>;
  prototype: {
    registerProvider(providerId: string, config: unknown): void;
    getProvider(providerId: string): unknown;
    getModel(providerId: string, modelId: string): unknown;
    getRegisteredProviderConfig(providerId: string): { apiKey?: string };
    getProviderAuthStatus(providerId: string): { configured?: boolean; label?: string };
    getAvailableSnapshot(): unknown[];
    refresh(options: { allowNetwork: boolean }): Promise<void>;
  };
}

const runtimeEntryUrl = import.meta.resolve("@earendil-works/pi-coding-agent");
const runtimeEntryPath = fileURLToPath(runtimeEntryUrl);
const runtimePath = join(dirname(runtimeEntryPath), "core", "model-runtime.js");
const { ModelRuntime } = (await import(
  pathToFileURL(runtimePath).href
)) as { ModelRuntime?: ModelRuntimeClass };

assert.ok(ModelRuntime, "ModelRuntime must be resolvable from the installed pi package");
// Every method this suite exercises must exist on the installed pi, or the
// suite fails at load time rather than mid-test with an opaque TypeError.
assert.equal(typeof ModelRuntime.create, "function", "create must exist on ModelRuntime");
for (const method of [
  "registerProvider",
  "getProvider",
  "getModel",
  "getRegisteredProviderConfig",
  "getProviderAuthStatus",
  "getAvailableSnapshot",
  "refresh",
] as const) {
  assert.equal(typeof ModelRuntime.prototype[method], "function", `${method} must exist on ModelRuntime`);
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
 */
async function waitForRegisterRefreshSettle(
  runtime: any,
  providerId: string,
  maxTries = 40,
  stepMs = 20,
): Promise<void> {
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
    // createAgentSessionServices does).
    await waitForRegisterRefreshSettle(runtime, providerId);
    await runtime.refresh({ allowNetwork: false });

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

      // The provider must be registered
      const provider = runtime.getProvider(PROVIDER);
      assert.ok(provider, "openrouter-novita provider should be registered");

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

      const model = runtime.getModel(PROVIDER, MODEL_ID);
      assert.ok(model, "model still registered even with no key source");

      const registered = runtime.getRegisteredProviderConfig(PROVIDER);
      assert.equal(registered?.apiKey, "$OPENROUTER_API_KEY", "no key source => placeholder preserved, not resolved");

      // After the explicit refresh({ allowNetwork:false }) already ran inside
      // the harness, the pin must be reported unconfigured (the placeholder is
      // not a resolvable key) and therefore excluded from the available snapshot.
      const authStatus = runtime.getProviderAuthStatus(PROVIDER);
      assert.equal(authStatus.configured, false, "provider without a key must be unconfigured after refresh");

      const available = (runtime.getAvailableSnapshot() as SnapshotModel[]).filter((m) => m.provider === PROVIDER);
      assert.equal(available.length, 0, "unconfigured pin must be excluded from the available snapshot");
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

      const model = runtime.getModel(PROVIDER, MODEL_ID);
      assert.ok(model, "model should be present despite BOM/comments");
      assert.deepStrictEqual(model?.compat?.openRouterRouting, { only: ["novita"], allow_fallbacks: false });

      const available = (runtime.getAvailableSnapshot() as SnapshotModel[]).filter((m) => m.provider === PROVIDER);
      assert.equal(available.length, 1, "BOM file pin should be selectable when auth present");
    }));
});