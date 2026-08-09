/**
 * Tests for the pricing/limits refresh (src/commands.ts).
 *
 * A — End-to-end: the real `refreshPinnedModels` against a real temp
 * models.json with a mocked OpenRouter client. The mock's fetchCatalog
 * mutates the file to simulate a concurrent pin/unpin landing between the
 * snapshot read and the apply-read — the exact interleaving the re-read-on-
 * write strategy protects against.
 *
 * B — Pure-path: the extracted phases (collectRefreshTargets →
 * computePricingPatches → applyPricingPatches) tested directly, covering the
 * edge cases the end-to-end tests can't reach.
 *
 * These tests import commands.ts, which pulls in @earendil-works/pi-tui at
 * runtime — run `npm install` (peer deps) first.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { OpenRouterClient } from "../src/api.ts";
import type { RawModelShape, ModelConfig } from "../src/config.ts";
import {
  applyPricingPatches,
  collectRefreshTargets,
  computePricingPatches,
  refreshPinnedModels,
} from "../src/commands.ts";
import { atomicWriteJson, readJsonFile, type ModelsJson, type ProviderEntry } from "../src/files.ts";

// ---------------------------------------------------------------------------
// Fixture helpers
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

/** A catalog entry, as served by OpenRouter's /models. */
const glmRaw = (over: Partial<RawModelShape> = {}): RawModelShape => ({
  id: "z-ai/glm-5.2",
  name: "Z.ai: GLM 5.2",
  context_length: 1_048_576,
  top_provider: { max_completion_tokens: 128_000 },
  per_request_limits: { max_tokens: 65_536 },
  pricing: { prompt: "0.1", completion: "0.31", input_cache_read: "0.02", input_cache_write: "0" },
  architecture: { input_modalities: ["text"] },
  supported_parameters: ["reasoning"],
  ...over,
});

/** A client whose fetchCatalog runs an arbitrary side effect before returning. */
const mockClient = (onCatalog: () => RawModelShape[] | Promise<RawModelShape[]>): OpenRouterClient =>
  ({ fetchCatalog: async () => onCatalog() }) as unknown as OpenRouterClient;

async function withTempModels(
  models: ModelsJson,
  fn: (modelsPath: string) => Promise<void>,
): Promise<void> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-refresh-"));
  const modelsPath = join(dir, "models.json");
  try {
    await atomicWriteJson(modelsPath, models);
    await fn(modelsPath);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// A — End-to-end
// ---------------------------------------------------------------------------

test("refreshPinnedModels: refreshes cost/contextWindow/maxTokens end-to-end", async () => {
  await withTempModels({ providers: { "openrouter-novita": providerEntry([glmModel()]) } }, async (modelsPath) => {
    const fresh = glmRaw({
      context_length: 2_000_000,
      top_provider: { max_completion_tokens: 200_000 },
      pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03", input_cache_write: "0" },
    });
    const result = await refreshPinnedModels(modelsPath, mockClient(() => [fresh]));
    assert.deepEqual(result, { refreshed: 1, failed: [] });

    const models = await readJsonFile<ModelsJson>(modelsPath);
    const updated = models!.providers!["openrouter-novita"].models[0];
    assert.deepEqual(updated.cost, { input: 200_000, output: 400_000, cacheRead: 30_000, cacheWrite: 0 });
    assert.equal(updated.contextWindow, 2_000_000);
    assert.equal(updated.maxTokens, 200_000);
    // Routing, name, and reasoning are untouched by a refresh.
    assert.equal(updated.name, "GLM 5.2 (novita)");
    assert.deepEqual(updated.compat.openRouterRouting, { only: ["novita"], allow_fallbacks: false });
  });
});

test("refreshPinnedModels: a pin written between snapshot read and apply-read survives", async () => {
  await withTempModels({ providers: { "openrouter-novita": providerEntry([glmModel()]) } }, async (modelsPath) => {
    const fresh = glmRaw({ pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03", input_cache_write: "0" } });
    // The concurrent pin lands inside fetchCatalog — i.e. after the snapshot
    // read, before the apply-read.
    const client = mockClient(async () => {
      const current = await readJsonFile<ModelsJson>(modelsPath);
      current!.providers!["openrouter-together"] = providerEntry([
        { ...glmModel(), id: "deepseek/deepseek-v4-flash-0731", name: "DeepSeek V4 Flash 0731" },
      ]);
      await atomicWriteJson(modelsPath, current);
      return [fresh];
    });

    const result = await refreshPinnedModels(modelsPath, client);
    assert.equal(result.refreshed, 1);

    const models = await readJsonFile<ModelsJson>(modelsPath);
    // The intervening pin survived the refresh write.
    assert.ok(models!.providers!["openrouter-together"], "concurrent pin must survive the refresh");
    assert.equal(models!.providers!["openrouter-together"].models[0].id, "deepseek/deepseek-v4-flash-0731");
    // And the original model's snapshot was refreshed.
    assert.deepEqual(models!.providers!["openrouter-novita"].models[0].cost, {
      input: 200_000,
      output: 400_000,
      cacheRead: 30_000,
      cacheWrite: 0,
    });
  });
});

test("refreshPinnedModels: an unpin between snapshot read and apply-read is not resurrected", async () => {
  await withTempModels({ providers: { "openrouter-novita": providerEntry([glmModel()]) } }, async (modelsPath) => {
    const fresh = glmRaw({ pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03", input_cache_write: "0" } });
    const client = mockClient(async () => {
      const current = await readJsonFile<ModelsJson>(modelsPath);
      delete current!.providers!["openrouter-novita"]; // concurrent unpin
      await atomicWriteJson(modelsPath, current);
      return [fresh];
    });

    const result = await refreshPinnedModels(modelsPath, client);
    assert.equal(result.refreshed, 0);

    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.equal(models!.providers!["openrouter-novita"], undefined, "unpinned provider must not be resurrected");
  });
});

// ---------------------------------------------------------------------------
// B — Pure phases
// ---------------------------------------------------------------------------

test("collectRefreshTargets: only openrouter-* providers with routing compat", () => {
  const snapshot: ModelsJson = {
    providers: {
      "openrouter-novita": providerEntry([glmModel()]),
      "openrouter-together": providerEntry([{ ...glmModel(), id: "deepseek/deepseek-v4-flash-0731" }]),
      "anthropic": providerEntry([glmModel()]), // not our prefix — ignored
    },
  };
  const targets = collectRefreshTargets(snapshot);
  assert.deepEqual(targets.map((t) => t.provider), ["openrouter-novita", "openrouter-together"]);
  assert.equal(collectRefreshTargets(null).length, 0);
  assert.equal(collectRefreshTargets({ providers: {} }).length, 0);
});

test("computePricingPatches: changed → patch, unchanged → skip, missing from catalog → failed", () => {
  const targets = collectRefreshTargets({ providers: { "openrouter-novita": providerEntry([glmModel()]) } });

  const changed = computePricingPatches(targets, [
    glmRaw({ pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03", input_cache_write: "0" } }),
  ]);
  assert.equal(changed.patches.length, 1);
  assert.deepEqual(changed.patches[0], {
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    cost: { input: 200_000, output: 400_000, cacheRead: 30_000, cacheWrite: 0 },
    contextWindow: 1_048_576,
    maxTokens: 128_000,
  });
  assert.deepEqual(changed.failed, []);

  // Same pricing as the stored snapshot → no patch.
  const unchanged = computePricingPatches(targets, [glmRaw()]);
  assert.equal(unchanged.patches.length, 0);
  assert.deepEqual(unchanged.failed, []);

  // Model gone from the catalog → reported as failed, not patched.
  const missing = computePricingPatches(targets, [glmRaw({ id: "some/other-model" })]);
  assert.equal(missing.patches.length, 0);
  assert.deepEqual(missing.failed, ["z-ai/glm-5.2"]);
});

test("applyPricingPatches: skips providers/models that vanished since the snapshot", () => {
  const patch = {
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    cost: { input: 200_000, output: 400_000, cacheRead: 30_000, cacheWrite: 0 },
    contextWindow: 2_000_000,
    maxTokens: 200_000,
  };
  // Provider gone since the snapshot: nothing applied, nothing resurrected.
  assert.equal(applyPricingPatches({ providers: {} }, [patch]), 0);
  // Model gone from a still-present provider: skipped.
  assert.equal(applyPricingPatches({ providers: { "openrouter-novita": providerEntry([]) } }, [patch]), 0);
  // Both present: applied, and only the three refresh fields change.
  const current: ModelsJson = { providers: { "openrouter-novita": providerEntry([glmModel()]) } };
  assert.equal(applyPricingPatches(current, [patch]), 1);
  const updated = current.providers!["openrouter-novita"].models[0];
  assert.equal(updated.cost.input, 200_000);
  assert.equal(updated.contextWindow, 2_000_000);
  assert.equal(updated.maxTokens, 200_000);
  assert.equal(updated.name, "GLM 5.2 (novita)");
  assert.equal(applyPricingPatches(null, [patch]), 0);
});
