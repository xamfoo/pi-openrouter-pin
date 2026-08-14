/**
 * Snapshot + behavior tests for the pure config builders (src/config.ts).
 *
 * Run: `node --test test/` (Node 22.6+ with type stripping, or Node 24+).
 * The snapshot file is the contract: when OpenRouter's catalog shape or the
 * pin's output shape drifts, these tests fail and the snapshot is updated
 * deliberately, not silently.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  buildPin,
  findEndpoint,
  mergeEndpointCost,
  pricingAndLimitsFromEndpoint,
  providerNameFor,
  slugify,
  stripLegacyAttribution,
  toCost,
  toModelConfig,
  toPricingAndLimits,
  type RawEndpointShape,
  type RawModelShape,
} from "../src/config.ts";

const here = dirname(fileURLToPath(import.meta.url));

function loadFixture(name: string): unknown {
  return JSON.parse(readFileSync(join(here, "fixtures", name), "utf-8"));
}

const raw = loadFixture("catalog-glm-5.2.json") as RawModelShape;
const expectedPin = loadFixture("pin-glm-5.2-novita.snapshot.json");

test("buildPin: fresh pin matches the checked-in snapshot", () => {
  const pin = buildPin(
    raw,
    { modelId: "z-ai/glm-5.2", slug: "novita", quant: "fp8", name: "GLM 5.2 (novita)", isDefault: true },
    [],
  );
  assert.deepEqual(pin, expectedPin);
});

test("buildPin: pinning a second model to the same provider keeps the first (no sibling drop)", () => {
  const first = buildPin(raw, { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false }, []);
  const secondRaw: RawModelShape = { ...raw, id: "deepseek/deepseek-v4-flash-0731", name: "DeepSeek V4 Flash 0731" };
  const second = buildPin(
    secondRaw,
    { modelId: "deepseek/deepseek-v4-flash-0731", slug: "novita", isDefault: false },
    first.providerEntry.models,
  );
  assert.equal(second.providerEntry.models.length, 2);
  assert.deepEqual(
    second.providerEntry.models.map((m) => m.id),
    ["z-ai/glm-5.2", "deepseek/deepseek-v4-flash-0731"],
  );
  // Re-pinning the same model replaces rather than duplicates.
  const again = buildPin(raw, { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false }, second.providerEntry.models);
  assert.equal(again.providerEntry.models.length, 2);
});

test("buildPin: settingsPatch only when isDefault is set", () => {
  const withDefault = buildPin(raw, { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: true }, []);
  assert.ok(withDefault.settingsPatch);
  assert.deepEqual(withDefault.settingsPatch, {
    defaultProvider: "openrouter-novita",
    defaultModel: "z-ai/glm-5.2",
    enabledModel: "openrouter-novita/z-ai/glm-5.2",
  });
  const withoutDefault = buildPin(raw, { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false }, []);
  assert.equal(withoutDefault.settingsPatch, undefined);
});

test("toModelConfig: strict pin keeps only + fallbacks off", () => {
  const cfg = toModelConfig(raw, { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false, quant: "fp8" });
  assert.deepEqual(cfg.compat.openRouterRouting, {
    only: ["novita"],
    allow_fallbacks: false,
    quantizations: ["fp8"],
  });
});

test("toModelConfig: order implies fallback, drops only, anchors the chosen provider first", () => {
  const cfg = toModelConfig(raw, {
    modelId: "z-ai/glm-5.2",
    slug: "novita",
    isDefault: false,
    order: ["Novita", "Together", "novita"], // anchor duplicated on purpose → deduped
  });
  assert.deepEqual(cfg.compat.openRouterRouting, {
    allow_fallbacks: true,
    order: ["novita", "together"],
  });
});

test("toModelConfig: ignore implies fallback (ignore is only meaningful when fallback is possible)", () => {
  const cfg = toModelConfig(raw, {
    modelId: "z-ai/glm-5.2",
    slug: "novita",
    isDefault: false,
    ignore: ["openai", "anthropic"],
  });
  assert.deepEqual(cfg.compat.openRouterRouting, {
    only: ["novita"],
    allow_fallbacks: true,
    ignore: ["openai", "anthropic"],
  });
});

test("toModelConfig: exposes the real routing vocabulary and thinkingFormat", () => {
  const cfg = toModelConfig(raw, {
    modelId: "z-ai/glm-5.2",
    slug: "novita",
    isDefault: false,
    allowFallbacks: true,
    order: ["Novita", "Together"],
    ignore: ["openai"],
    dataCollection: "deny",
  });
  assert.equal(cfg.compat.thinkingFormat, "openrouter");
  assert.deepEqual(cfg.compat.openRouterRouting, {
    allow_fallbacks: true,
    order: ["novita", "together"],
    ignore: ["openai"],
    data_collection: "deny",
  });
  assert.equal(cfg.reasoning, true);
  assert.deepEqual(cfg.input, ["text"]);
});

test("providerNameFor: strict pins name the provider; relaxed pins get -plus (no lying names)", () => {
  assert.equal(providerNameFor("novita"), "openrouter-novita");
  assert.equal(providerNameFor("novita", { allowFallbacks: true }), "openrouter-novita-plus");
  assert.equal(providerNameFor("novita", { order: ["together"] }), "openrouter-novita-plus");
  assert.equal(providerNameFor("novita", { ignore: ["openai"] }), "openrouter-novita-plus");
  assert.equal(providerNameFor("Novita", { allowFallbacks: true }), "openrouter-novita-plus");
});

test("buildPin: relaxed pins get a -plus provider name", () => {
  const pin = buildPin(raw, { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false, allowFallbacks: true }, []);
  assert.equal(pin.providerName, "openrouter-novita-plus");
  assert.equal(pin.providerEntry.models[0].compat.openRouterRouting.allow_fallbacks, true);
});

test("buildPin: an endpoint supplies provider-specific pricing & limits (the glm-5.2/novita fix)", () => {
  // Without an endpoint, the pin records the catalog aggregate (SiliconFlow's
  // prices for glm-5.2) — the bug.
  const noEndpoint = buildPin(raw, { modelId: "z-ai/glm-5.2", slug: "novita", quant: "fp8", isDefault: false }, []);
  assert.deepEqual(noEndpoint.modelConfig.cost, { input: 100000, output: 310000, cacheRead: 20000, cacheWrite: 0 });
  assert.equal(noEndpoint.modelConfig.maxTokens, 128000);

  // With the validated novita endpoint, the pin records novita's real prices.
  const novita: RawEndpointShape = {
    provider_name: "Novita",
    quantization: "fp8",
    context_length: 1_048_576,
    max_completion_tokens: 131_072,
    pricing: { prompt: "0.0000007406", completion: "0.0000023276", input_cache_read: "0.00000013754" },
  };
  const pin = buildPin(raw, { modelId: "z-ai/glm-5.2", slug: "novita", quant: "fp8", isDefault: false, endpoint: novita }, []);
  assert.deepEqual(pin.modelConfig.cost, { input: 0.74, output: 2.33, cacheRead: 0.14, cacheWrite: 0 });
  assert.equal(pin.modelConfig.maxTokens, 131_072);
  // Routing, name, reasoning, and input modalities are unaffected.
  assert.equal(pin.modelConfig.name, "Z.ai: GLM 5.2 (novita fp8)");
  assert.equal(pin.modelConfig.reasoning, true);
  assert.deepEqual(pin.modelConfig.compat.openRouterRouting, { only: ["novita"], allow_fallbacks: false, quantizations: ["fp8"] });
});

test("toModelConfig: empty/absent name falls back to the generated one, never blank", () => {
  const cfg = toModelConfig(raw, { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false });
  assert.equal(cfg.name, "Z.ai: GLM 5.2 (novita)");
  const withQuant = toModelConfig(raw, { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false, quant: "fp8" });
  assert.equal(withQuant.name, "Z.ai: GLM 5.2 (novita fp8)");
});

test("toCost: per-token pricing converts to per-million-token", () => {
  assert.deepEqual(toCost({ prompt: "0.1", completion: "0.31", input_cache_read: "0.02", input_cache_write: "0" }), {
    input: 100000,
    output: 310000,
    cacheRead: 20000,
    cacheWrite: 0,
  });
  assert.deepEqual(toCost(undefined), { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
});

test("toPricingAndLimits: endpoint overrides the catalog aggregate (provider-specific truth)", () => {
  // Catalog aggregate (matches SiliconFlow for glm-5.2, not the pinned novita).
  const snap = toPricingAndLimits(raw);
  assert.deepEqual(snap.cost, { input: 100000, output: 310000, cacheRead: 20000, cacheWrite: 0 });
  assert.equal(snap.maxTokens, 128000);

  // A novita endpoint carries its own pricing + limits; those must win.
  const novita: RawEndpointShape = {
    provider_name: "Novita",
    quantization: "fp8",
    context_length: 1_048_576,
    max_completion_tokens: 131_072,
    pricing: { prompt: "0.0000007406", completion: "0.0000023276", input_cache_read: "0.00000013754" },
  };
  const pinned = toPricingAndLimits(raw, novita);
  assert.deepEqual(pinned.cost, { input: 0.74, output: 2.33, cacheRead: 0.14, cacheWrite: 0 });
  assert.equal(pinned.maxTokens, 131_072, "endpoint max_completion_tokens overrides catalog top_provider");
  assert.equal(pinned.contextWindow, 1_048_576);
});

test("toPricingAndLimits: endpoint omits input_cache_write → catalog value is preserved, not zeroed", () => {
  // A model whose catalog records a nonzero cache-write keeps it when the
  // endpoint (which never carries input_cache_write) is supplied.
  const rawWithCacheWrite: RawModelShape = {
    ...raw,
    pricing: { prompt: "0.1", completion: "0.31", input_cache_read: "0.02", input_cache_write: "0.05" },
  };
  const endpoint: RawEndpointShape = {
    provider_name: "Novita",
    pricing: { prompt: "0.0000007406", completion: "0.0000023276", input_cache_read: "0.00000013754" },
  };
  assert.equal(toPricingAndLimits(rawWithCacheWrite, endpoint).cost.cacheWrite, 50000, "catalog cacheWrite preserved");
  // But a present endpoint "0" is honored (distinguished from absent).
  const endpointWithZero: RawEndpointShape = { ...endpoint, pricing: { ...endpoint.pricing!, input_cache_write: "0" } };
  assert.equal(toPricingAndLimits(rawWithCacheWrite, endpointWithZero).cost.cacheWrite, 0, "endpoint 0 honored");
});

test("toPricingAndLimits: endpoint with no pricing falls back to catalog cost", () => {
  const endpoint: RawEndpointShape = { provider_name: "Novita", context_length: 500_000 };
  const pinned = toPricingAndLimits(raw, endpoint);
  assert.deepEqual(pinned.cost, { input: 100000, output: 310000, cacheRead: 20000, cacheWrite: 0 });
  assert.equal(pinned.contextWindow, 500_000, "endpoint context_length still overrides");
});

test("mergeEndpointCost: present endpoint keys override, absent keys fall back", () => {
  const fallback = { input: 100000, output: 310000, cacheRead: 20000, cacheWrite: 50000 };
  assert.deepEqual(
    mergeEndpointCost({ pricing: { prompt: "0.2", completion: "0.4" } }, fallback),
    { input: 200000, output: 400000, cacheRead: 20000, cacheWrite: 50000 },
  );
  // No pricing at all → full fallback.
  assert.deepEqual(mergeEndpointCost({}, fallback), fallback);
});

test("pricingAndLimitsFromEndpoint: refresh fallback is the stored snapshot", () => {
  const stored = { contextWindow: 1_048_576, maxTokens: 128_000, cost: { input: 100000, output: 310000, cacheRead: 20000, cacheWrite: 0 } };
  const endpoint: RawEndpointShape = {
    provider_name: "Novita",
    max_completion_tokens: 131_072,
    pricing: { prompt: "0.0000007406", completion: "0.0000023276", input_cache_read: "0.00000013754" },
  };
  assert.deepEqual(pricingAndLimitsFromEndpoint(endpoint, stored), {
    contextWindow: 1_048_576,
    maxTokens: 131_072,
    cost: { input: 0.74, output: 2.33, cacheRead: 0.14, cacheWrite: 0 },
  });
  // Endpoint omitting a field keeps the stored value.
  const partial: RawEndpointShape = { provider_name: "Novita", pricing: { prompt: "0.2" } };
  assert.deepEqual(pricingAndLimitsFromEndpoint(partial, stored).cost, {
    input: 200000, output: 310000, cacheRead: 20000, cacheWrite: 0,
  });
});

test("findEndpoint: matches slug, then quant; never substitutes another quant's pricing", () => {
  const endpoints: RawEndpointShape[] = [
    { provider_name: "SiliconFlow", quantization: "fp8" },
    { provider_name: "Novita", quantization: "fp4" },
    { provider_name: "Novita", quantization: "fp8" },
  ];
  assert.equal(findEndpoint(endpoints, "novita", "fp8")!.quantization, "fp8");
  assert.equal(findEndpoint(endpoints, "novita", "fp4")!.quantization, "fp4");
  // No quant → first endpoint for the slug.
  assert.equal(findEndpoint(endpoints, "novita")!.quantization, "fp4");
  // Quant the provider no longer serves → undefined (no substitution).
  assert.equal(findEndpoint(endpoints, "novita", "bf16"), undefined);
  // Unknown slug → undefined.
  assert.equal(findEndpoint(endpoints, "baseten"), undefined);
  assert.equal(findEndpoint([], "novita"), undefined);
});

test("slugify: one normalizer for endpoint names and user input", () => {
  assert.equal(slugify("NVIDIA"), "nvidia");
  assert.equal(slugify("Z AI"), "z-ai");
  assert.equal(slugify("X.AI"), "x-ai");
  assert.equal(slugify("  Hyperbolic  "), "hyperbolic");
  assert.equal(providerNameFor("Novita"), "openrouter-novita");
});

test("stripLegacyAttribution: strips only the exact legacy pair this extension wrote", () => {
  const legacy = { "HTTP-Referer": "https://github.com/", "X-Title": "pi-openrouter-pin" };
  assert.equal(stripLegacyAttribution(legacy), undefined);
  const legacyPlusCustom = { ...legacy, "X-Custom": "keep" };
  assert.deepEqual(stripLegacyAttribution(legacyPlusCustom), { "X-Custom": "keep" });
  const userHeaders = { "HTTP-Referer": "https://example.com/", "X-Title": "my-app" };
  assert.deepEqual(stripLegacyAttribution(userHeaders), userHeaders);
  assert.equal(stripLegacyAttribution(undefined), undefined);
});
