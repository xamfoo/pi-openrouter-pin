/**
 * Tests for src/api.ts helpers (no pi packages at runtime — plain node --test).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pinnedProviderSlugs, OpenRouterClient } from "../src/api.ts";
import { atomicWriteJson, type ModelsJson } from "../src/files.ts";

test("pinnedProviderSlugs: strips the -plus relaxation suffix, never offers it as a provider", async () => {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-slugs-"));
  const modelsPath = join(dir, "models.json");
  try {
    const models: ModelsJson = {
      providers: {
        "openrouter-novita": { baseUrl: "https://openrouter.ai/api/v1", api: "openai-completions", apiKey: "$OPENROUTER_API_KEY", models: [] },
        "openrouter-novita-plus": { baseUrl: "https://openrouter.ai/api/v1", api: "openai-completions", apiKey: "$OPENROUTER_API_KEY", models: [] },
        "openrouter-together": { baseUrl: "https://openrouter.ai/api/v1", api: "openai-completions", apiKey: "$OPENROUTER_API_KEY", models: [] },
        "anthropic": { baseUrl: "https://anthropic.com", api: "anthropic", apiKey: "$ANTHROPIC_API_KEY", models: [] },
      },
    };
    await atomicWriteJson(modelsPath, models);
    // novita appears from both the strict and the relaxed pin, deduped to one;
    // the -plus suffix is this extension's, not the provider's.
    assert.deepEqual(await pinnedProviderSlugs(modelsPath), ["novita", "together"]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("searchModels: builds the ?q= URL and memoizes the last query", async () => {
  const calls: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    calls.push(String(input));
    return new Response(JSON.stringify({ data: [{ id: "z-ai/glm-5.2", name: "Z.ai: GLM 5.2" }] }));
  }) as typeof fetch;
  try {
    const client = new OpenRouterClient(60_000, 60_000);
    const first = await client.searchModels("glm");
    const memoized = await client.searchModels("glm");
    const other = await client.searchModels("glm-5");
    assert.deepEqual(
      first.map((m) => m.id),
      ["z-ai/glm-5.2"],
      "search returns the data array",
    );
    assert.equal(calls.length, 2, "the same query is memoized, a new query fetches");
    assert.ok(calls[0]?.includes("q=glm"), "query travels as the q parameter");
    assert.ok(memoized.length === 1 && other.length === 1);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("validateEndpoint: matches the tag's base routing slug after the Google rename", async () => {
  const calls: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    calls.push(String(input));
    // OpenRouter now reports provider_name "Google" (renamed from "Google
    // Vertex AI") and carries the routing slug in each endpoint's tag.
    return new Response(JSON.stringify({
      data: {
        endpoints: [
          { provider_name: "Google", tag: "google-vertex/global" },
          { provider_name: "Google", tag: "google-vertex/global/flex" },
          { provider_name: "Google AI Studio", tag: "google-ai-studio" },
        ],
      },
    }));
  }) as typeof fetch;
  try {
    const client = new OpenRouterClient(60_000, 60_000);
    const ok = await client.validateEndpoint("google/gemini-3.7-flash", "google-vertex", undefined, "test-key");
    assert.equal(ok.status, "ok", "google-vertex must validate from its tag");
    assert.equal(ok.endpoint?.tag, "google-vertex/global");
    assert.ok(calls[0]?.includes("google/gemini-3.7-flash/endpoints"), "endpoints URL uses the model path");

    // A slug OpenRouter does not serve is refused, and the message lists the
    // tag-derived slugs a user could actually pin.
    const err = await client.validateEndpoint("google/gemini-3.7-flash", "novita", undefined, "test-key");
    assert.equal(err.status, "error");
    assert.ok(err.message.includes("does not serve"), "error explains the refusal");
    assert.ok(err.message.includes("google-ai-studio"), "available list includes google-ai-studio");
    assert.ok(err.message.includes("google-vertex"), "available list includes google-vertex");
    assert.ok(!err.message.includes("google,"), "renamed provider_name must not leak as slug \"google\"");
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("validateEndpoint: without a key, pinning is unvalidated (not refused)", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    throw new Error("fetch must not be called without an API key");
  }) as typeof fetch;
  try {
    const client = new OpenRouterClient(60_000, 60_000);
    const result = await client.validateEndpoint("google/gemini-3.7-flash", "google-vertex", undefined, undefined);
    assert.equal(result.status, "unvalidated");
  } finally {
    globalThis.fetch = realFetch;
  }
});
