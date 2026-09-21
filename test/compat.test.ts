/**
 * Offline compatibility tests: old-shape vs new-shape pi/ctx.
 *
 * These tests verify that the plugin works correctly when pi's
 * extension API has different shapes:
 * - old pi (<=0.84): sync ModelRegistry.refresh, no unregisterProvider/setModel
 * - new pi (>=0.86): async refresh, has unregisterProvider/setModel
 *
 * All tests run under plain `npm test` without network or special env vars.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ModelRegistry } from "@earendil-works/pi-coding-agent";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import {
  planUnpin,
  computeSettingsPrune,
  probe,
  performUnpin,
  performPin,
  type CapabilityProbe,
} from "../src/commands.ts";
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

const deepseekModel = (over: Partial<ModelConfig> = {}): ModelConfig => ({
  id: "deepseek/deepseek-v4-flash-0731",
  name: "DeepSeek V4 Flash (novita)",
  reasoning: true,
  input: ["text"],
  contextWindow: 131_072,
  maxTokens: 32_768,
  cost: { input: 20_000, output: 80_000, cacheRead: 5_000, cacheWrite: 0 },
  compat: { thinkingFormat: "openrouter", openRouterRouting: { order: ["novita", "deepseek"], allow_fallbacks: true } },
  ...over,
});

async function withTempDir<T>(fn: (dir: string) => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), "compat-"));
  try {
    return await fn(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// Old-shape pi: sync refresh, no unregisterProvider/setModel
// ---------------------------------------------------------------------------

function createOldShapePi(): ExtensionAPI {
  return {
    on: () => {},
    registerProvider: () => {},
    registerCommand: () => {},
    // No unregisterProvider, no setModel — old pi
  } as unknown as ExtensionAPI;
}

// ---------------------------------------------------------------------------
// New-shape pi: async refresh, has unregisterProvider/setModel
// ---------------------------------------------------------------------------

function createNewShapePi(): ExtensionAPI {
  return {
    on: () => {},
    registerProvider: () => {},
    registerCommand: () => {},
    unregisterProvider: () => {},
    setModel: () => Promise.resolve(),
    // Also has modelRegistry.refresh (async)
  } as unknown as ExtensionAPI;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test("probe: old-shape pi reports canUnregister=false, canSetModel=false, live=false", () => {
  const cap = probe(createOldShapePi());
  assert.equal(cap.canUnregister, false);
  assert.equal(cap.canSetModel, false);
  assert.equal(cap.live, false);
});

test("probe: new-shape pi reports canUnregister=true, canSetModel=true, live=true", () => {
  const cap = probe(createNewShapePi());
  assert.equal(cap.canUnregister, true);
  assert.equal(cap.canSetModel, true);
  assert.equal(cap.live, true);
});

test("probe: mixed capability (only unregister) reports live=true", () => {
  const pi = {
    on: () => {},
    registerProvider: () => {},
    registerCommand: () => {},
    unregisterProvider: () => {},
  } as unknown as ExtensionAPI;
  const cap = probe(pi);
  assert.equal(cap.canUnregister, true);
  assert.equal(cap.canSetModel, false);
  assert.equal(cap.live, true);
});

test("planUnpin works with old-shape context (no live capabilities)", () => {
  const snapshot: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel(), deepseekModel()]),
    },
  };
  // Old pi: no settings changes needed for non-default model.
  const plan = planUnpin(snapshot, null, "z-ai/glm-5.2");
  // novita had two models; removing glm-5.2 leaves deepseek, so repruned.
  assert.equal(plan.repruned.length, 1);
  assert.equal(plan.emptied.length, 0);
  assert.equal(plan.removed.length, 1);
  assert.equal(plan.settingsPatch, null);
});

test("performUnpin works under old-shape pi (fileOnly path)", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-novita": providerEntry([glmModel(), deepseekModel()]) },
    });
    const outcome = await performUnpin(modelsPath, "z-ai/glm-5.2");
    assert.equal(outcome.status, "removed");
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(
      models!.providers!["openrouter-novita"].models.map((m) => m.id),
      ["deepseek/deepseek-v4-flash-0731"],
    );
  });
});

test("performUnpin with empty provider drops the provider (old pi fileOnly)", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-novita": providerEntry([glmModel()]) },
    });
    const outcome = await performUnpin(modelsPath, "z-ai/glm-5.2");
    assert.equal(outcome.status, "removed");
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(models, { providers: {} });
  });
});

test("performPin works under old-shape pi (no live setModel)", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, { providers: {} });
    const pi = createOldShapePi();
    const ctx = {
      mode: "tui" as const,
      hasUI: true,
      cwd: "/tmp",
      modelRegistry: {},
      model: undefined,
      scopedModels: [],
      ui: {
        notify: () => {},
        select: async () => undefined,
      },
    } as unknown as ExtensionCommandContext;
    // This should not throw even without setModel on pi.
    await assert.doesNotReject(
      performPin({
        modelsPath,
        settingsPath,
        pi,
        ctx: ctx.ui,
        client: {
          fetchRawModel: async () => ({ id: "z-ai/glm-5.2", name: "GLM 5.2" }),
          validateEndpoint: async () => ({ status: "ok" as const, endpoint: "https://example.com" }),
          fetchModelEndpoints: async () => ({ endpoints: [] }),
        } as any,
        resolveApiKey: async () => undefined,
        opts: { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false } as any,
      }),
      "performPin should not throw under old-shape pi",
    );
    // Also verify second call with same options-object works.
    await assert.doesNotReject(
      performPin({
        modelsPath,
        settingsPath,
        pi,
        ctx: ctx.ui,
        client: {
          fetchRawModel: async () => ({ id: "z-ai/glm-5.2", name: "GLM 5.2" }),
          validateEndpoint: async () => ({ status: "ok" as const, endpoint: "https://example.com" }),
          fetchModelEndpoints: async () => ({ endpoints: [] }),
        } as any,
        resolveApiKey: async () => undefined,
        opts: { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false } as any,
      }),
      "second performPin should not throw under old-shape pi",
    );
  });
});

test("planUnpin with settings computes correct patch for default clearing", () => {
  const settings = { defaultProvider: "openrouter-novita", defaultModel: "z-ai/glm-5.2", enabledModels: ["z-ai/glm-5.2"] };
  const snapshot: ModelsJson = {
    providers: { "openrouter-novita": providerEntry([glmModel()]) },
  };
  const plan = planUnpin(snapshot, settings, "z-ai/glm-5.2");
  assert.ok(plan.settingsPatch, "settingsPatch is computed");
  assert.equal(plan.settingsPatch!.defaultProvider, undefined);
  assert.equal(plan.settingsPatch!.defaultModel, undefined);
});

test("computeSettingsPrune handles stale defaults (model not in enabledModels)", () => {
  const settings = { defaultProvider: "openrouter-novita", defaultModel: "z-ai/glm-5.2", enabledModels: ["other/model"] };
  const result = computeSettingsPrune(settings, "z-ai/glm-5.2", []);
  assert.ok(result, "default match clears even without emptied providers");
  assert.equal(result!.defaultProvider, undefined);
  assert.equal(result!.defaultModel, undefined);
  // enabledModels does NOT contain the removed model, so no prune there.
  assert.equal(result!.enabledModel, undefined);
});

test("computeSettingsPrune: enabled model gets pruned", () => {
  // When the model is in enabledModels, pruning it IS a settings change.
  // Correct disk shape is `provider/model`, but bare id also matches for compat.
  const settings = { defaultProvider: "anthropic", defaultModel: "claude-3", enabledModels: ["z-ai/glm-5.2"] };
  const result = computeSettingsPrune(settings, "z-ai/glm-5.2", []);
  assert.ok(result, "pruning an enabled model produces a patch");
  assert.equal(result!.enabledModel, "", "all enabled models pruned → empty string (cleared to [])");
});

test("computeSettingsPrune: provider/model enabled entry gets pruned", () => {
  const settings = { defaultProvider: "anthropic", defaultModel: "claude-3", enabledModels: ["openrouter-novita/z-ai/glm-5.2", "other/model"] };
  const result = computeSettingsPrune(settings, "z-ai/glm-5.2", []);
  assert.ok(result, "provider/model entry pruned via suffix match");
  assert.equal(result!.enabledModel, "other/model");
});

test("computeSettingsPrune leaves truly unrelated defaults untouched", () => {
  // Model not in enabledModels and not the default → no changes.
  const settings = { defaultProvider: "anthropic", defaultModel: "claude-3", enabledModels: ["other/model"] };
  const result = computeSettingsPrune(settings, "z-ai/glm-5.2", []);
  assert.equal(result, null, "truly unrelated model does not trigger settings changes");
});

// ---------------------------------------------------------------------------
// Refresh boundary compatibility (await`-agnostic)
// ---------------------------------------------------------------------------

test("performUnpin handles null/empty snapshot gracefully", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    // Missing file — should return no-providers without error.
    const outcome = await performUnpin(modelsPath, "x/y");
    assert.equal(outcome.status, "no-providers");
  });
});

test("performPin with unbound setModel (loader stub) degrades to fileOnly", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    const settingsPath = join(dir, "settings.json");
    await atomicWriteJson(modelsPath, { providers: {} });
    const settingsBefore = { defaultProvider: "openrouter-novita", defaultModel: "z-ai/glm-5.2", enabledModels: ["z-ai/glm-5.2"] };
    await atomicWriteJson(settingsPath, settingsBefore);
    const pi = {
      on: () => {},
      registerProvider: () => {},
      registerCommand: () => {},
      unregisterProvider: () => {},
      // Loader-only stub: rejects with the same message as discoverAndLoadExtensions
      setModel: () => Promise.reject(new Error("Extension runtime not initialized")),
    } as unknown as ExtensionAPI;
    const notifications: { message: string; type: string }[] = [];
    const ctx = {
      mode: "tui" as const,
      hasUI: true,
      cwd: "/tmp",
      modelRegistry: {},
      model: undefined,
      scopedModels: [],
      ui: {
        notify: (message: string, type: string) => { notifications.push({ message, type }); },
        select: async () => undefined,
      },
    } as unknown as ExtensionCommandContext;
    await performPin({
      modelsPath,
      settingsPath,
      pi,
      ctx: ctx.ui,
      client: {
        fetchRawModel: async () => ({ id: "z-ai/glm-5.2", name: "GLM 5.2" }),
        validateEndpoint: async () => ({ status: "ok" as const, endpoint: "https://example.com" }),
        fetchModelEndpoints: async () => ({ endpoints: [] }),
      } as any,
      resolveApiKey: async () => undefined,
      opts: { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: true } as any,
    });
    // settings.json must persist (no rollback on loader-unbound).
    // performPin adds the enabledModel to enabledModels as part of
    // the settings patch, so include it in the expected snapshot.
    const expectedSettings = { ...settingsBefore, enabledModels: [...settingsBefore.enabledModels, "openrouter-novita/z-ai/glm-5.2"] };
    const settings = await readJsonFile<typeof expectedSettings>(settingsPath);
    assert.deepEqual(settings, expectedSettings, "settings.json must persist (fileOnly, no rollback)");
    // No error notifications from the loader-unbound path.
    assert.ok(notifications.every((n) => n.type !== "error"), `unexpected error notifications: ${JSON.stringify(notifications)}`);
    // Must have the fileOnly info notification.
    assert.ok(notifications.some((n) => n.type === "info" && n.message.includes("applies on /reload or next session")),
      "expected fileOnly notification");
  });
});

test("performUnpin not-found never writes files", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    await atomicWriteJson(modelsPath, { providers: { "openrouter-novita": providerEntry([glmModel()]) } });
    const outcome = await performUnpin(modelsPath, "nobody/home");
    assert.equal(outcome.status, "not-found");
    // File should be unchanged.
    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.equal(models!.providers!["openrouter-novita"].models.length, 1);
  });
});
