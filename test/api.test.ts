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
