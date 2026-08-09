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
  providerNameFor,
  slugify,
  stripLegacyAttribution,
  toCost,
  toModelConfig,
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
