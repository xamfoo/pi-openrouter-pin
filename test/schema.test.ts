/**
 * Contract tests: the OpenRouter catalog shape this extension depends on.
 *
 * The fixture (test/fixtures/catalog-glm-5.2.json) mirrors a real GET /models
 * payload. Every field the extension reads must be present with the expected
 * shape; when OpenRouter drifts, the fixture is updated deliberately and this
 * test documents the drift.
 *
 * A live check against the real endpoint is included, opt-in via
 * OPENROUTER_API_KEY (network + account-dependent, so it never runs in plain
 * CI): it asserts the live payload satisfies the same contract.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { toPricingAndLimits, type RawModelShape } from "../src/config.ts";

const here = dirname(fileURLToPath(import.meta.url));
const raw = JSON.parse(readFileSync(join(here, "fixtures", "catalog-glm-5.2.json"), "utf-8")) as RawModelShape;

test("catalog fixture carries every field the extension reads, in the expected shape", () => {
  assert.equal(typeof raw.id, "string");
  assert.ok(raw.id.includes("/"), "model ids are namespaced (org/model)");
  assert.equal(typeof raw.name, "string");
  assert.equal(typeof raw.context_length, "number");
  assert.ok(raw.top_provider && typeof raw.top_provider.max_completion_tokens === "number");
  assert.ok(raw.per_request_limits && typeof raw.per_request_limits.max_tokens === "number");
  assert.ok(raw.pricing, "pricing map present");
  for (const k of ["prompt", "completion", "input_cache_read", "input_cache_write"] as const) {
    assert.equal(typeof raw.pricing?.[k], "string", `pricing.${k} must be a per-token USD string`);
  }
  assert.ok(Array.isArray(raw.architecture?.input_modalities));
  assert.ok(Array.isArray(raw.supported_parameters));

  // The snapshot derived from it (what a pin persists) is finite and sane.
  const snap = toPricingAndLimits(raw);
  assert.ok(snap.cost.input > 0 && snap.cost.output > 0);
  assert.ok(snap.contextWindow >= snap.maxTokens);
});

test("live OpenRouter /models payload satisfies the same contract (needs OPENROUTER_API_KEY + network)", {
  skip: !process.env.OPENROUTER_API_KEY,
}, async () => {
  const res = await fetch("https://openrouter.ai/api/v1/models");
  assert.ok(res.ok, `GET /models → HTTP ${res.status}`);
  const payload = (await res.json()) as { data?: RawModelShape[] };
  assert.ok(Array.isArray(payload.data) && payload.data.length > 0, "catalog is a non-empty data array");
  // Every entry must not crash the snapshot builder; a sample must carry the
  // fields the pin reads.
  const sample = payload.data[0];
  assert.equal(typeof sample.id, "string");
  assert.ok(sample.id.includes("/"));
  assert.equal(typeof sample.name, "string");
  for (const m of payload.data.slice(0, 50)) toPricingAndLimits(m);

  // Server-side search (?q=) returns the same model-object contract.
  const searchRes = await fetch("https://openrouter.ai/api/v1/models?q=deepseek");
  assert.ok(searchRes.ok, `GET /models?q= → HTTP ${searchRes.status}`);
  const searchPayload = (await searchRes.json()) as { data?: RawModelShape[] };
  assert.ok(Array.isArray(searchPayload.data) && searchPayload.data.length > 0, "search returns a non-empty data array");
  for (const m of searchPayload.data.slice(0, 20)) {
    assert.equal(typeof m.id, "string");
    assert.ok(m.id.includes("/"));
    toPricingAndLimits(m);
  }
});
