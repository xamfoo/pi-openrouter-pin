/**
 * Tests for the pricing/limits refresh (src/commands.ts).
 *
 * A — End-to-end: the real `refreshPinnedModels` against a real temp
 * models.json with a mocked OpenRouter client. The mock's fetchModelEndpoints
 * mutates the file to simulate a concurrent pin/unpin landing between the
 * snapshot read and the apply-read — the exact interleaving the re-read-on-
 * write strategy protects against.
 *
 * B — Pure-path: the extracted phases (collectRefreshTargets →
 * computeEndpointPatch → applyPricingPatches) tested directly, covering the
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
import type { RawEndpointShape, RawModelShape, ModelConfig } from "../src/config.ts";
import {
  applyPricingPatches,
  collectRefreshTargets,
  computeEndpointPatch,
  formatRefreshDiff,
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

/** A stored pinned model, as written to models.json (catalog-aggregate pricing). */
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

/** A catalog entry, as served by OpenRouter's /models (model-level aggregate). */
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

/**
 * A novita endpoint, as served by /models/<id>/endpoints (provider-specific).
 * Its pricing differs from the catalog aggregate — the whole reason the
 * refresh must use endpoints, not the catalog.
 */
const novitaEndpoint = (over: Partial<RawEndpointShape> = {}): RawEndpointShape => ({
  provider_name: "Novita",
  quantization: "fp8",
  context_length: 1_048_576,
  max_completion_tokens: 128_000,
  pricing: { prompt: "0.0000007406", completion: "0.0000023276", input_cache_read: "0.00000013754" },
  ...over,
});

/** A client whose fetchModelEndpoints runs an arbitrary side effect before returning. */
const mockClient = (
  onEndpoints: (modelId: string) => Promise<RawEndpointShape[]> | RawEndpointShape[],
): OpenRouterClient =>
  ({
    fetchModelEndpoints: async (modelId: string) => ({ endpoints: await onEndpoints(modelId) }),
  }) as unknown as OpenRouterClient;

/** A client that reports the endpoints API unavailable for a model (404 / gone). */
const unavailableClient = (message: string): OpenRouterClient =>
  ({
    fetchModelEndpoints: async () => ({ endpoints: [], message }),
  }) as unknown as OpenRouterClient;

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

const noopKey = () => Promise.resolve("test-key" as string | undefined);

// ---------------------------------------------------------------------------
// A — End-to-end
// ---------------------------------------------------------------------------

test("refreshPinnedModels: refreshes cost/contextWindow/maxTokens from the pinned provider's endpoint", async () => {
  await withTempModels({ providers: { "openrouter-novita": providerEntry([glmModel()]) } }, async (modelsPath) => {
    // The novita endpoint carries different pricing + limits than the stored
    // catalog-aggregate snapshot — refresh must adopt the endpoint's values.
    const fresh = novitaEndpoint({
      context_length: 2_000_000,
      max_completion_tokens: 200_000,
      pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03" },
    });
    const result = await refreshPinnedModels(modelsPath, mockClient(() => [fresh]), noopKey);
    assert.deepEqual(result, {
      refreshed: 1,
      failed: [],
      diff: [{
        provider: "openrouter-novita",
        modelId: "z-ai/glm-5.2",
        before: { cost: { input: 100_000, output: 310_000, cacheRead: 20_000, cacheWrite: 0 }, contextWindow: 1_048_576, maxTokens: 128_000 },
        after: { cost: { input: 200_000, output: 400_000, cacheRead: 30_000, cacheWrite: 0 }, contextWindow: 2_000_000, maxTokens: 200_000 },
      }],
    });

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

test("refreshPinnedModels: endpoint pricing overrides the catalog aggregate (the glm-5.2/novita bug)", async () => {
  // Reproduces the reported issue: the stored pin carries the catalog's
  // model-level pricing (which for glm-5.2 matches SiliconFlow), but the pin
  // is to novita, so refresh must record novita's endpoint prices instead.
  const stored = glmModel({
    // Catalog-aggregate values (SiliconFlow's prices as $/M).
    cost: { input: 1_190_000, output: 3_740_000, cacheRead: 220_000, cacheWrite: 0 },
    maxTokens: 262_144, // catalog top_provider, not novita's 131072
    compat: {
      thinkingFormat: "openrouter",
      openRouterRouting: { only: ["novita"], allow_fallbacks: false, quantizations: ["fp8"] },
    },
  });
  await withTempModels({ providers: { "openrouter-novita": providerEntry([stored]) } }, async (modelsPath) => {
    const result = await refreshPinnedModels(modelsPath, mockClient(() => [novitaEndpoint({ max_completion_tokens: 131_072 })]), noopKey);
    assert.equal(result.refreshed, 1);

    const models = await readJsonFile<ModelsJson>(modelsPath);
    const updated = models!.providers!["openrouter-novita"].models[0];
    // Novita's real endpoint prices: 0.74 / 2.33 / 0.14 ($/M), maxTokens 131072.
    assert.deepEqual(updated.cost, { input: 0.74, output: 2.33, cacheRead: 0.14, cacheWrite: 0 });
    assert.equal(updated.maxTokens, 131_072);
  });
});

test("refreshPinnedModels: a pin written between snapshot read and apply-read survives", async () => {
  await withTempModels({ providers: { "openrouter-novita": providerEntry([glmModel()]) } }, async (modelsPath) => {
    const fresh = novitaEndpoint({
      pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03" },
    });
    // The concurrent pin lands inside fetchModelEndpoints — i.e. after the
    // snapshot read, before the apply-read.
    const client = mockClient(async () => {
      const current = await readJsonFile<ModelsJson>(modelsPath);
      current!.providers!["openrouter-together"] = providerEntry([
        { ...glmModel(), id: "deepseek/deepseek-v4-flash-0731", name: "DeepSeek V4 Flash 0731" },
      ]);
      await atomicWriteJson(modelsPath, current);
      return [fresh];
    });

    const result = await refreshPinnedModels(modelsPath, client, noopKey);
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
    const fresh = novitaEndpoint({
      pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03" },
    });
    const client = mockClient(async () => {
      const current = await readJsonFile<ModelsJson>(modelsPath);
      delete current!.providers!["openrouter-novita"]; // concurrent unpin
      await atomicWriteJson(modelsPath, current);
      return [fresh];
    });

    const result = await refreshPinnedModels(modelsPath, client, noopKey);
    assert.equal(result.refreshed, 0);

    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.equal(models!.providers!["openrouter-novita"], undefined, "unpinned provider must not be resurrected");
  });
});

test("refreshPinnedModels: without an API key, refresh is skipped (no clobbering with catalog aggregate)", async () => {
  await withTempModels({ providers: { "openrouter-novita": providerEntry([glmModel()]) } }, async (modelsPath) => {
    const client = mockClient(() => [novitaEndpoint()]);
    const result = await refreshPinnedModels(modelsPath, client, () => Promise.resolve(undefined));
    assert.deepEqual(result, { refreshed: 0, failed: [], diff: [] });

    const models = await readJsonFile<ModelsJson>(modelsPath);
    // Stored values untouched — refresh did not fall back to the catalog.
    assert.deepEqual(models!.providers!["openrouter-novita"].models[0].cost, {
      input: 100_000,
      output: 310_000,
      cacheRead: 20_000,
      cacheWrite: 0,
    });
  });
});

test("refreshPinnedModels: a model whose endpoints are unavailable is reported as failed, not thrown", async () => {
  await withTempModels({ providers: { "openrouter-novita": providerEntry([glmModel()]) } }, async (modelsPath) => {
    const result = await refreshPinnedModels(
      modelsPath,
      unavailableClient("OpenRouter lists no endpoints for z-ai/glm-5.2"),
      noopKey,
    );
    assert.deepEqual(result, { refreshed: 0, failed: ["z-ai/glm-5.2"], diff: [] });
  });
});

test("refreshPinnedModels: a stored quant the provider no longer serves is left untouched (not clobbered)", async () => {
  // The pin asks for fp8, but novita now only serves fp4 — there is no
  // matching endpoint, so refresh must not substitute another endpoint's
  // pricing (it would be wrong for the requested quant).
  const stored = glmModel({
    compat: {
      thinkingFormat: "openrouter",
      openRouterRouting: { only: ["novita"], allow_fallbacks: false, quantizations: ["fp8"] },
    },
  });
  await withTempModels({ providers: { "openrouter-novita": providerEntry([stored]) } }, async (modelsPath) => {
    const client = mockClient(() => [novitaEndpoint({ quantization: "fp4" })]);
    const result = await refreshPinnedModels(modelsPath, client, noopKey);
    assert.equal(result.refreshed, 0);

    const models = await readJsonFile<ModelsJson>(modelsPath);
    assert.deepEqual(models!.providers!["openrouter-novita"].models[0].cost, {
      input: 100_000,
      output: 310_000,
      cacheRead: 20_000,
      cacheWrite: 0,
    });
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

test("computeEndpointPatch: changed → patch; unchanged → null; missing endpoint → null", () => {
  const target = collectRefreshTargets({ providers: { "openrouter-novita": providerEntry([glmModel()]) } })[0];

  // Different pricing → patch with the endpoint's values.
  const changed = computeEndpointPatch(target, [
    novitaEndpoint({ pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03" } }),
  ]);
  assert.deepEqual(changed, {
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    cost: { input: 200_000, output: 400_000, cacheRead: 30_000, cacheWrite: 0 },
    contextWindow: 1_048_576,
    maxTokens: 128_000,
  });

  // Same pricing & limits as stored → no patch.
  const unchanged = computeEndpointPatch(target, [novitaEndpoint({
    context_length: 1_048_576,
    max_completion_tokens: 128_000,
    pricing: { prompt: "0.1", completion: "0.31", input_cache_read: "0.02" }, // matches stored cost
  })]);
  assert.equal(unchanged, null);

  // No endpoint for the anchor slug → null (no clobbering).
  assert.equal(computeEndpointPatch(target, [{ provider_name: "SiliconFlow" }]), null);
  // No endpoints at all → null.
  assert.equal(computeEndpointPatch(target, []), null);
});

test("computeEndpointPatch: honors the stored quant when selecting the endpoint", () => {
  const stored = glmModel({
    compat: {
      thinkingFormat: "openrouter",
      openRouterRouting: { only: ["novita"], allow_fallbacks: false, quantizations: ["fp8"] },
    },
  });
  const target = collectRefreshTargets({ providers: { "openrouter-novita": providerEntry([stored]) } })[0];
  const fp8 = novitaEndpoint({ quantization: "fp8", pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03" } });
  const fp4 = novitaEndpoint({ quantization: "fp4", pricing: { prompt: "0.0000005", completion: "0.000001", input_cache_read: "0.000000075" } });
  // fp8 pin selects the fp8 endpoint even though fp4 is also served.
  const patch = computeEndpointPatch(target, [fp4, fp8]);
  assert.deepEqual(patch!.cost, { input: 200_000, output: 400_000, cacheRead: 30_000, cacheWrite: 0 });
});

test("computeEndpointPatch: relaxed pin uses the order anchor (head of order)", () => {
  const stored = glmModel({
    compat: {
      thinkingFormat: "openrouter",
      openRouterRouting: { allow_fallbacks: true, order: ["novita", "together"] },
    },
  });
  const target = collectRefreshTargets({ providers: { "openrouter-novita-plus": providerEntry([stored]) } })[0];
  const patch = computeEndpointPatch(target, [
    novitaEndpoint({ pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03" } }),
  ]);
  assert.equal(patch!.provider, "openrouter-novita-plus");
  assert.deepEqual(patch!.cost, { input: 200_000, output: 400_000, cacheRead: 30_000, cacheWrite: 0 });
});

test("computeEndpointPatch: preserves input_cache_write from the stored snapshot when the endpoint omits it", () => {
  // Endpoints never carry input_cache_write; a stored pin that recorded a
  // nonzero cache-write (from the catalog) must keep it, not zero it.
  const stored = glmModel({
    cost: { input: 100_000, output: 310_000, cacheRead: 20_000, cacheWrite: 5_000 },
  });
  const target = collectRefreshTargets({ providers: { "openrouter-novita": providerEntry([stored]) } })[0];
  const patch = computeEndpointPatch(target, [
    novitaEndpoint({ pricing: { prompt: "0.2", completion: "0.4", input_cache_read: "0.03" } }),
  ]);
  assert.equal(patch!.cost.cacheWrite, 5_000, "stored input_cache_write is preserved");
  assert.equal(patch!.cost.input, 200_000, "endpoint input overrides stored");
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
  assert.deepEqual(applyPricingPatches({ providers: {} }, [patch]), { applied: 0, diff: [] });
  // Model gone from a still-present provider: skipped.
  assert.deepEqual(applyPricingPatches({ providers: { "openrouter-novita": providerEntry([]) } }, [patch]), { applied: 0, diff: [] });
  // Both present: applied, and only the three refresh fields change.
  const current: ModelsJson = { providers: { "openrouter-novita": providerEntry([glmModel()]) } };
  const result = applyPricingPatches(current, [patch]);
  assert.equal(result.applied, 1);
  assert.deepEqual(result.diff, [{
    provider: "openrouter-novita",
    modelId: "z-ai/glm-5.2",
    before: { cost: { input: 100_000, output: 310_000, cacheRead: 20_000, cacheWrite: 0 }, contextWindow: 1_048_576, maxTokens: 128_000 },
    after: { cost: { input: 200_000, output: 400_000, cacheRead: 30_000, cacheWrite: 0 }, contextWindow: 2_000_000, maxTokens: 200_000 },
  }]);
  const updated = current.providers!["openrouter-novita"].models[0];
  assert.equal(updated.cost.input, 200_000);
  assert.equal(updated.contextWindow, 2_000_000);
  assert.equal(updated.maxTokens, 200_000);
  assert.equal(updated.name, "GLM 5.2 (novita)");
  assert.deepEqual(applyPricingPatches(null, [patch]), { applied: 0, diff: [] });
});

// ---------------------------------------------------------------------------
// C — Reload-time diff rendering
// ---------------------------------------------------------------------------

test("formatRefreshDiff: empty input → empty string", () => {
  assert.equal(formatRefreshDiff([]), "");
});

test("formatRefreshDiff: emits only changed fields, grouped + aligned per model", () => {
  const out = formatRefreshDiff([
    {
      provider: "openrouter-novita",
      modelId: "z-ai/glm-5.2",
      before: { cost: { input: 100_000, output: 310_000, cacheRead: 20_000, cacheWrite: 0 }, contextWindow: 1_048_576, maxTokens: 128_000 },
      after: { cost: { input: 200_000, output: 310_000, cacheRead: 30_000, cacheWrite: 0 }, contextWindow: 1_048_576, maxTokens: 200_000 },
    },
  ]);
  // Changed: cost.input, cost.cacheRead, maxTokens.
  // Unchanged (must be absent): cost.output, cost.cacheWrite, contextWindow.
  const lines = out.split("\n");
  assert.equal(lines[0], "  z-ai/glm-5.2 (openrouter-novita):");
  assert.equal(lines.length, 4, "header + 3 changed fields");
  assert.ok(lines[1].startsWith("    cost.input"), "cost.input row first");
  assert.ok(lines[1].includes("$100000.00/M") && lines[1].includes("$200000.00/M"));
  assert.ok(lines[2].includes("cost.cacheRead"));
  assert.ok(lines[3].includes("maxTokens"));
  assert.ok(lines[3].includes("128,000") && lines[3].includes("200,000"));
  assert.ok(!out.includes("cost.output"), "unchanged cost.output must not appear");
  assert.ok(!out.includes("cost.cacheWrite"), "unchanged cost.cacheWrite must not appear");
  assert.ok(!out.includes("contextWindow"), "unchanged contextWindow must not appear");
});

test("formatRefreshDiff: multiple models render as separate blocks", () => {
  const out = formatRefreshDiff([
    {
      provider: "openrouter-novita",
      modelId: "z-ai/glm-5.2",
      before: { cost: { input: 100_000, output: 310_000, cacheRead: 20_000, cacheWrite: 0 }, contextWindow: 1_048_576, maxTokens: 128_000 },
      after: { cost: { input: 200_000, output: 310_000, cacheRead: 20_000, cacheWrite: 0 }, contextWindow: 1_048_576, maxTokens: 128_000 },
    },
    {
      provider: "openrouter-together",
      modelId: "deepseek/deepseek-v4-flash-0731",
      before: { cost: { input: 50_000, output: 150_000, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128_000, maxTokens: 16_384 },
      after: { cost: { input: 50_000, output: 150_000, cacheRead: 0, cacheWrite: 0 }, contextWindow: 256_000, maxTokens: 16_384 },
    },
  ]);
  const headers = out.match(/^  [^\s].*:$/gm) ?? [];
  assert.equal(headers.length, 2, "two model header blocks");
  assert.ok(out.includes("z-ai/glm-5.2 (openrouter-novita):"));
  assert.ok(out.includes("deepseek/deepseek-v4-flash-0731 (openrouter-together):"));
  // First model only changed cost.input; second only changed contextWindow.
  assert.ok(out.includes("cost.input"));
  assert.ok(out.includes("contextWindow"));
  assert.ok(!out.includes("maxTokens"), "unchanged maxTokens must not appear");
});
