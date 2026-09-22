/**
 * Factory auth inheritance tests for the auth-pin-repro change.
 *
 * Four layers:
 *
 *   A — Pure table tests: `resolveFactoryKey` and `collectPinnedProviders`
 *       (no fs, no pi).
 *   B — Thin fs wrappers: `readAuthJsonSync` and `registerPinnedProviders`
 *       with temp dirs and JSONC files.
 *   C — Factory repros: full extension registration with temp PI_CODING_AGENT_DIR,
 *       asserting `registerProvider` calls via ExtensionAPI spy.
 *   D — Live pin repros: `performPin` with mocked resolveApiKey.
 *
 * Run: `node --test test/factory-auth.test.ts`
 */
import { describe, test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  atomicWriteJson,
  collectPinnedProviders,
  readAuthJsonSync,
  readJsonFile,
  registerPinnedProviders,
  resolveFactoryKey,
  type ModelsJson,
  type ProviderEntry,
} from "../src/files.ts";
import openrouterPinExtension from "../src/index.ts";
import { performPin } from "../src/commands.ts";
import { type ExtensionAPI, type ExtensionUIContext, type ModelRegistry } from "@earendil-works/pi-coding-agent";
import { type ModelConfig, type PinOptions } from "../src/config.ts";
import { OpenRouterClient } from "../src/api.ts";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const makeProviderEntry = (models: ModelConfig[]): ProviderEntry => ({
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function withTempDir<T>(fn: (dir: string) => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-"));
  try {
    return await fn(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------
// A — Pure table tests
// ---------------------------------------------------------------------------

// --- Untagged leniency (backwards compat) ---

test("resolveFactoryKey: env wins over auth.json", () => {
  assert.strictEqual(
    resolveFactoryKey("  sk-env  ", { openrouter: { key: "sk-auth" } }),
    "sk-env",
  );
});

test("resolveFactoryKey: auth.json fallback when env absent", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { key: "  sk-auth  " } }),
    "sk-auth",
  );
});

test("resolveFactoryKey: both absent returns undefined", () => {
  assert.strictEqual(resolveFactoryKey(undefined, undefined), undefined);
  assert.strictEqual(resolveFactoryKey("", undefined), undefined);
  assert.strictEqual(resolveFactoryKey("   ", undefined), undefined);
});

test("resolveFactoryKey: whitespace-only env falls back to auth.json", () => {
  assert.strictEqual(
    resolveFactoryKey("   ", { openrouter: { key: "sk-auth" } }),
    "sk-auth",
  );
});

test("resolveFactoryKey: non-string auth.json key is ignored", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { key: 123 } }),
    undefined,
  );
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { key: "" } }),
    undefined,
  );
});

test("resolveFactoryKey: malformed auth.json does not throw", () => {
  assert.strictEqual(resolveFactoryKey(undefined, null), undefined);
});

test("resolveFactoryKey: untagged {key} leniency still works", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { key: "  sk-lenient  " } }),
    "sk-lenient",
  );
});

test("resolveFactoryKey: untagged key starting with ! is literal", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { key: "!op read secret" } }),
    "!op read secret",
  );
});

// --- OAuth credential ---

test("resolveFactoryKey: oauth returns trimmed access", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "oauth", access: "  tok-abc  ", refresh: "ref", expires: 9007199254740991 } }),
    "tok-abc",
  );
});

test("resolveFactoryKey: oauth with empty refresh still returns access", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "oauth", access: "tok-abc", refresh: "", expires: 9007199254740991 } }),
    "tok-abc",
  );
});

test("resolveFactoryKey: oauth with expired expires still returns access", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "oauth", access: "tok-abc", refresh: "ref", expires: 0 } }),
    "tok-abc",
  );
});

test("resolveFactoryKey: oauth with missing access returns undefined", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "oauth", access: "", refresh: "ref", expires: 9007199254740991 } }),
    undefined,
  );
});

// --- $VAR / ${VAR} / $$ resolution with tagged api_key ---

test("resolveFactoryKey: tagged api_key with $VAR resolves from credEnv", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: "$OPENROUTER_API_KEY", env: { OPENROUTER_API_KEY: "resolved-key" } } }),
    "resolved-key",
  );
});

test("resolveFactoryKey: tagged api_key with ${VAR} resolves from credEnv", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: "${OPENROUTER_API_KEY}", env: { OPENROUTER_API_KEY: "resolved-key" } } }),
    "resolved-key",
  );
});

test("resolveFactoryKey: tagged api_key $VAR resolves from process.env when credEnv absent", () => {
  const saved = process.env.TEST_RESOLVE_VAR;
  process.env.TEST_RESOLVE_VAR = "from-process-env";
  try {
    assert.strictEqual(
      resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: "$TEST_RESOLVE_VAR" } }),
      "from-process-env",
    );
  } finally {
    if (saved === undefined) delete process.env.TEST_RESOLVE_VAR;
    else process.env.TEST_RESOLVE_VAR = saved;
  }
});

test("resolveFactoryKey: tagged api_key unresolvable $VAR stays literal", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: "$UNRESOLVABLE_VAR", env: {} } }),
    "$UNRESOLVABLE_VAR",
  );
});

test("resolveFactoryKey: $$ escapes to literal dollar", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: "$${OPENROUTER_API_KEY}", env: { OPENROUTER_API_KEY: "x" } } }),
    "${OPENROUTER_API_KEY}",
  );
});

test("resolveFactoryKey: tagged api_key !command passes through literally", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: "!op read openrouter/api-key" } }),
    "!op read openrouter/api-key",
  );
});

test("resolveFactoryKey: tagged api_key whitespace-only key returns undefined", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: "   " } }),
    undefined,
  );
});

test("resolveFactoryKey: tagged api_key with env wins over process.env", () => {
  const saved = process.env.TEST_ENV_PRECEEDENCE;
  process.env.TEST_ENV_PRECEEDENCE = "process-val";
  try {
    assert.strictEqual(
      resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: "$TEST_ENV_PRECEEDENCE", env: { TEST_ENV_PRECEEDENCE: "cred-val" } } }),
      "cred-val",
    );
  } finally {
    if (saved === undefined) delete process.env.TEST_ENV_PRECEEDENCE;
    else process.env.TEST_ENV_PRECEEDENCE = saved;
  }
});

test("resolveFactoryKey: tagged api_key with non-string key is ignored", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: 123 } }),
    undefined,
  );
});

test("resolveFactoryKey: tagged oauth with non-string access is ignored", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "oauth", access: 123, refresh: "ref", expires: 1 } }),
    undefined,
  );
});

// --- :free id verbatim coverage ---

test("resolveFactoryKey: tagged api_key key is returned literally", () => {
  assert.strictEqual(
    resolveFactoryKey(undefined, { openrouter: { type: "api_key", key: "sk-free" } }),
    "sk-free",
  );
});

test("collectPinnedProviders: openrouter-decart with :free model id is valid", () => {
  const snapshot: ModelsJson = {
    providers: {
      "openrouter-decart": makeProviderEntry([glmModel({ id: "z-ai/glm-5.2:free" })]),
    },
  };
  const pinned = collectPinnedProviders(snapshot);
  assert.strictEqual(pinned.length, 1);
  assert.strictEqual(pinned[0][0], "openrouter-decart");
});

test("collectPinnedProviders: only openrouter-* with non-empty models", () => {
  const snapshot: ModelsJson = {
    providers: {
      "openrouter-novita": makeProviderEntry([glmModel()]),
      "openrouter-novita-plus": makeProviderEntry([deepseekModel()]),
      "openrouter-preset": makeProviderEntry([glmModel()]),
      "anthropic": makeProviderEntry([glmModel()]),
      "openrouter-empty": { ...makeProviderEntry([]), models: [] },
    },
  };
  const pinned = collectPinnedProviders(snapshot);
  const names = pinned.map(([n]) => n).sort();
  assert.deepEqual(names, ["openrouter-novita", "openrouter-novita-plus", "openrouter-preset"]);
});

test("collectPinnedProviders: null/empty snapshot yields []", () => {
  assert.deepEqual(collectPinnedProviders(null), []);
  assert.deepEqual(collectPinnedProviders({}), []);
  assert.deepEqual(collectPinnedProviders({ providers: {} }), []);
});

// ---------------------------------------------------------------------------
// B — Thin fs wrappers
// ---------------------------------------------------------------------------

test("readAuthJsonSync: missing auth.json returns undefined silently", async () => {
  await withTempDir(async (dir) => {
    assert.strictEqual(readAuthJsonSync(dir), undefined);
    assert.strictEqual(resolveFactoryKey(undefined, readAuthJsonSync(dir)), undefined);
  });
});

test("readAuthJsonSync: env arg wins over auth.json (no process.env mutation)", async () => {
  await withTempDir(async (dir) => {
    writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { key: "sk-auth" } }));
    assert.strictEqual(
      resolveFactoryKey("  sk-env  ", readAuthJsonSync(dir)),
      "sk-env",
    );
  });
});

test("readAuthJsonSync: auth.json fallback when env absent", async () => {
  await withTempDir(async (dir) => {
    writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { key: "  sk-auth  " } }));
    assert.strictEqual(
      resolveFactoryKey(undefined, readAuthJsonSync(dir)),
      "sk-auth",
    );
  });
});

test("readAuthJsonSync: malformed auth.json warns and returns undefined", async () => {
  await withTempDir(async (dir) => {
    writeFileSync(join(dir, "auth.json"), "{ not json");
    const warnSpy: string[] = [];
    const origWarn = console.warn;
    console.warn = (...args: unknown[]) => warnSpy.push(String(args.join(" ")));
    try {
      assert.strictEqual(readAuthJsonSync(dir), undefined);
      assert.ok(warnSpy.some((w) => w.includes("auth.json")));
    } finally {
      console.warn = origWarn;
    }
  });
});

test("registerPinnedProviders: re-registers openrouter-* pins with effective key", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: {
        "openrouter-novita": makeProviderEntry([glmModel()]),
        "openrouter-novita-plus": makeProviderEntry([deepseekModel()]),
      },
    });

    const registered = new Map<string, ProviderEntry>();
    const pi = {
      on: () => {},
      registerProvider: (name: string, entry: ProviderEntry) => { registered.set(name, entry); },
      registerCommand: () => {},
    } as unknown as ExtensionAPI;

    registerPinnedProviders(pi, modelsPath, "sk-auth");
    assert.strictEqual(registered.size, 2);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "sk-auth");
    assert.strictEqual(registered.get("openrouter-novita-plus")?.apiKey, "sk-auth");
  });
});

test("registerPinnedProviders: no effective key leaves entries unchanged", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    await atomicWriteJson(modelsPath, {
      providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
    });

    const registered = new Map<string, ProviderEntry>();
    const pi = {
      on: () => {},
      registerProvider: (name: string, entry: ProviderEntry) => { registered.set(name, entry); },
      registerCommand: () => {},
    } as unknown as ExtensionAPI;

    registerPinnedProviders(pi, modelsPath, undefined);
    assert.strictEqual(registered.size, 1);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "$OPENROUTER_API_KEY");
  });
});

test("registerPinnedProviders: missing models.json is silent", async () => {
  await withTempDir(async (dir) => {
    const registered: string[] = [];
    const pi = {
      on: () => {},
      registerProvider: (name: string) => { registered.push(name); },
      registerCommand: () => {},
    } as unknown as ExtensionAPI;

    const warnSpy: string[] = [];
    const origWarn = console.warn;
    console.warn = (...args: unknown[]) => warnSpy.push(String(args.join(" ")));
    try {
      registerPinnedProviders(pi, join(dir, "nonexistent.json"), "sk-auth");
      assert.strictEqual(registered.length, 0);
      assert.strictEqual(warnSpy.length, 0);
    } finally {
      console.warn = origWarn;
    }
  });
});

test("registerPinnedProviders: malformed models.json warns and returns", async () => {
  await withTempDir(async (dir) => {
    const modelsPath = join(dir, "models.json");
    writeFileSync(modelsPath, "{ not json");

    const registered: string[] = [];
    const pi = {
      on: () => {},
      registerProvider: (name: string) => { registered.push(name); },
      registerCommand: () => {},
    } as unknown as ExtensionAPI;

    const warnSpy: string[] = [];
    const origWarn = console.warn;
    console.warn = (...args: unknown[]) => warnSpy.push(String(args.join(" ")));
    try {
      registerPinnedProviders(pi, modelsPath, "sk-auth");
      assert.strictEqual(registered.length, 0);
      assert.ok(warnSpy.some((w) => w.includes("models.json")));
    } finally {
      console.warn = origWarn;
    }
  });
});

// ---------------------------------------------------------------------------
// C — Factory repros
// ---------------------------------------------------------------------------

/** Run the extension factory in a temp PI_CODING_AGENT_DIR and capture registerProvider calls. */
async function factoryHarness(setup: (dir: string) => Promise<void>): Promise<Map<string, ProviderEntry>> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-factory-"));
  const saved = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = dir;
  const registered = new Map<string, ProviderEntry>();
  try {
    await setup(dir);
    const pi = {
      on: () => {},
      registerProvider: (name: string, entry: ProviderEntry) => { registered.set(name, entry); },
      registerCommand: () => {},
    } as unknown as ExtensionAPI;
    openrouterPinExtension(pi);
    return registered;
  } finally {
    if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = saved;
    await rm(dir, { recursive: true, force: true });
  }
}

/** Run the extension factory with a spy on registerCommand. */
async function factoryCommandHarness(setup: (dir: string) => Promise<void>): Promise<string[]> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-factory-"));
  const saved = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = dir;
  const commands: string[] = [];
  try {
    await setup(dir);
    const pi = {
      on: () => {},
      registerProvider: () => {},
      registerCommand: (name: string) => { commands.push(name); },
    } as unknown as ExtensionAPI;
    openrouterPinExtension(pi);
    return commands;
  } finally {
    if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = saved;
    await rm(dir, { recursive: true, force: true });
  }
}

// Mirrors pins.test.ts:withAgentDir - factory captures getAgentDir() via
// ENV_AGENT_DIR (PI_CODING_AGENT_DIR). process.env is global, so these run
// serially to avoid races with parallel tests touching the same env var.
describe("factory repros (PI_CODING_AGENT_DIR serial)", { concurrency: 1 }, () => {
  test("4.1 Factory repro: pins re-registered with apiKey from auth.json without env", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api_key", key: "sk-auth" } }));
      await atomicWriteJson(join(dir, "models.json"), {
        providers: {
          "openrouter-novita": makeProviderEntry([glmModel()]),
          "openrouter-novita-plus": makeProviderEntry([deepseekModel()]),
        },
      });
    });

    assert.strictEqual(registered.size, 2, "both openrouter-* providers should be re-registered");
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "sk-auth");
    assert.strictEqual(registered.get("openrouter-novita-plus")?.apiKey, "sk-auth");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});

test("4.2 Factory repro: env wins over auth.json (trimmed)", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  process.env.OPENROUTER_API_KEY = "  sk-env  ";
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api_key", key: "sk-auth" } }));
      await atomicWriteJson(join(dir, "models.json"), {
        providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
      });
    });
    assert.strictEqual(registered.size, 1);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "sk-env");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});

test("4.2 Factory repro: whitespace-only env falls back to auth.json", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  process.env.OPENROUTER_API_KEY = "   ";
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api_key", key: "sk-auth" } }));
      await atomicWriteJson(join(dir, "models.json"), {
        providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
      });
    });
    assert.strictEqual(registered.size, 1);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "sk-auth");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});

test("4.2 Factory repro: untagged {key} leniency still works", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { key: "sk-auth" } }));
      await atomicWriteJson(join(dir, "models.json"), {
        providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
      });
    });
    assert.strictEqual(registered.size, 1);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "sk-auth");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});

test("4.3 Factory repro: missing files are silent, non-openrouter providers not re-registered", async () => {
  const registered = await factoryHarness(async (dir) => {
    await atomicWriteJson(join(dir, "models.json"), {
      providers: {
        "anthropic": makeProviderEntry([glmModel()]),
        "openrouter-empty": { ...makeProviderEntry([]), models: [] },
      },
    });
  });
  assert.strictEqual(registered.size, 0);
});

test("4.3 Factory repro: JSONC comments (// and /* */) in models.json still re-register", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api_key", key: "sk-auth" } }));
      // Write models.json with JSONC comments
      const modelsWithComments =
        '{\n  // This is a comment\n  "providers": {\n    /* comment */ "openrouter-novita": {\n      "baseUrl": "https://openrouter.ai/api/v1",\n      "api": "openai-completions",\n      "apiKey": "$OPENROUTER_API_KEY",\n      "models": [{"id": "z-ai/glm-5.2"}]\n    }\n  }\n}\n';
      writeFileSync(join(dir, "models.json"), modelsWithComments);
    });
    assert.strictEqual(registered.size, 1);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "sk-auth");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});

test("4.4 Factory repro: malformed files warn but do not crash", async () => {
  const warnSpy: string[] = [];
  const origWarn = console.warn;
  console.warn = (...args: unknown[]) => warnSpy.push(String(args.join(" ")));
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), "{ not json");
      writeFileSync(join(dir, "models.json"), "{ not json");
    });
    assert.ok(warnSpy.some((w) => w.includes("auth.json")));
    assert.ok(warnSpy.some((w) => w.includes("models.json")));
    assert.strictEqual(registered.size, 0);
  } finally {
    console.warn = origWarn;
  }
});

test("4.4 Factory repro: malformed files still allow registerCommand (no crash)", async () => {
  const commands = await factoryCommandHarness(async (dir) => {
    writeFileSync(join(dir, "auth.json"), "{ not json");
    writeFileSync(join(dir, "models.json"), "{ not json");
  });
  assert.ok(commands.includes("openrouter-pin"), "openrouter-pin command registered despite malformed files");
  assert.ok(commands.includes("openrouter-unpin"), "openrouter-unpin command registered despite malformed files");
  assert.ok(commands.includes("openrouter-pins"), "openrouter-pins command registered despite malformed files");
});

test("4.6 Factory repro: tagged api_key credential resolves $VAR", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api_key", key: "$OPENROUTER_API_KEY", env: { OPENROUTER_API_KEY: "sk-resolved-from-cred" } } }));
      await atomicWriteJson(join(dir, "models.json"), {
        providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
      });
    });
    assert.strictEqual(registered.size, 1);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "sk-resolved-from-cred");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});

test("4.7 Factory repro: !command passes through literally without exec", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api_key", key: "!op read openrouter/api-key" } }));
      await atomicWriteJson(join(dir, "models.json"), {
        providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
      });
    });
    assert.strictEqual(registered.size, 1);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "!op read openrouter/api-key");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});

test("4.8 Factory repro: oauth credential uses access token", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "oauth", access: "tok-oauth", refresh: "", expires: 9007199254740991 } }));
      await atomicWriteJson(join(dir, "models.json"), {
        providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
      });
    });
    assert.strictEqual(registered.size, 1);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "tok-oauth");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});

test("4.9 Factory repro: oauth with empty refresh still registers", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "oauth", access: "tok-oauth", refresh: "", expires: 9007199254740991 } }));
      await atomicWriteJson(join(dir, "models.json"), {
        providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
      });
    });
    assert.strictEqual(registered.size, 1);
    assert.strictEqual(registered.get("openrouter-novita")?.apiKey, "tok-oauth");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});

test("4.10 Factory repro: :free model id preserved verbatim in registerProvider", async () => {
  const savedEnv = process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_API_KEY;
  try {
    const registered = await factoryHarness(async (dir) => {
      writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api_key", key: "sk-free-key" } }));
      await atomicWriteJson(join(dir, "models.json"), {
        providers: {
          "openrouter-decart": makeProviderEntry([glmModel({ id: "z-ai/glm-5.2:free" })]),
        },
      });
    });
    assert.strictEqual(registered.size, 1);
    assert.ok(registered.has("openrouter-decart"));
    assert.strictEqual(registered.get("openrouter-decart")?.apiKey, "sk-free-key");
  } finally {
    if (savedEnv === undefined) delete process.env.OPENROUTER_API_KEY;
    else process.env.OPENROUTER_API_KEY = savedEnv;
  }
});
}); // end factory repros (PI_CODING_AGENT_DIR serial)

// ---------------------------------------------------------------------------
// D — Live pin repros (performPin)
// ---------------------------------------------------------------------------

/** Create a mock OpenRouterClient for performPin tests. */
function mockClient(fetchRawModelResult: any, validateEndpointResult: any): OpenRouterClient {
  return {
    fetchCatalog: async () => [],
    fetchRawModel: async () => fetchRawModelResult,
    fetchModelEndpoints: async () => ({ endpoints: [] }),
    validateEndpoint: async () => validateEndpointResult,
    searchModels: async () => [],
    fetchUserModelIds: async () => null,
  } as unknown as OpenRouterClient;
}

async function performPinTest(
  modelsPath: string,
  resolveApiKey: () => Promise<string | undefined>,
  opts: PinOptions,
): Promise<Map<string, ProviderEntry>> {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-perform-"));
  const registered = new Map<string, ProviderEntry>();
  const settingsPath = join(dir, "settings.json");
  const client = mockClient(
    { id: opts.modelId, name: opts.modelId, context_length: 128_000, top_provider: { max_completion_tokens: 128_000 }, pricing: {} },
    { status: "ok" as const, endpoint: {} },
  );
  const pi = {
    on: () => {},
    registerProvider: (name: string, entry: ProviderEntry) => { registered.set(name, entry); },
    registerCommand: () => {},
  } as unknown as ExtensionAPI;
  const ctx = {
    mode: "tui" as const,
    hasUI: true,
    cwd: "/tmp",
    modelRegistry: {} as ModelRegistry,
    model: undefined,
    scopedModels: [],
    notify: () => {},
    ui: {
      notify: () => {},
      custom: async () => "",
      select: async () => "",
    },
  } as unknown as ExtensionUIContext;

  await performPin({ modelsPath, settingsPath, pi, ctx, client, resolveApiKey, opts });
  await rm(dir, { recursive: true, force: true });
  return registered;
}

test("4.5 Live pin repro: resolveApiKey injects apiKey into live registerProvider", async () => {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-perform-"));
  const modelsPath = join(dir, "models.json");
  await atomicWriteJson(modelsPath, {
    providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
  });

  const registered = await performPinTest(
    modelsPath,
    async () => "sk-live",
    { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false },
  );

  assert.strictEqual(registered.size, 1);
  const entry = registered.get("openrouter-novita");
  assert.ok(entry, "provider should be registered");
  assert.strictEqual(entry!.apiKey, "sk-live");
});

test("4.5 Live pin repro: undefined resolveApiKey leaves provider unchanged", async () => {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-perform-"));
  const modelsPath = join(dir, "models.json");
  await atomicWriteJson(modelsPath, {
    providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
  });

  const registered = await performPinTest(
    modelsPath,
    async () => undefined,
    { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false },
  );

  assert.strictEqual(registered.size, 1);
  const entry = registered.get("openrouter-novita");
  assert.ok(entry, "provider should be registered");
  assert.strictEqual(entry!.apiKey, "$OPENROUTER_API_KEY");
});

test("performPin: persisted models.json keeps $OPENROUTER_API_KEY placeholder (not resolved key)", async () => {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-perform-"));
  const modelsPath = join(dir, "models.json");
  await atomicWriteJson(modelsPath, {
    providers: { "openrouter-novita": makeProviderEntry([glmModel()]) },
  });

  const registered = await performPinTest(
    modelsPath,
    async () => "sk-live",
    { modelId: "z-ai/glm-5.2", slug: "novita", isDefault: false },
  );

  assert.strictEqual(registered.size, 1);
  const entry = registered.get("openrouter-novita");
  assert.ok(entry, "provider should be registered");
  assert.strictEqual(entry!.apiKey, "sk-live", "live registerProvider gets resolved key");

  // Verify persisted file has the placeholder
  const disk = await readJsonFile<ModelsJson>(modelsPath);
  assert.strictEqual(
    disk?.providers?.["openrouter-novita"]?.apiKey,
    "$OPENROUTER_API_KEY",
    "persisted models.json must NOT contain the resolved literal key",
  );
});

// ---------------------------------------------------------------------------
// 5.x — Stretch tests
// ---------------------------------------------------------------------------
// 5.1-5.3 touch PI_CODING_AGENT_DIR (via factoryHarness or directly) - run
// serially for same reason as C.
describe("stretch factory repros (PI_CODING_AGENT_DIR serial)", { concurrency: 1 }, () => {
test("5.1 openrouter-preset is re-registered via prefix rule", async () => {
  const registered = await factoryHarness(async (dir) => {
    writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api_key", key: "sk-auth" } }));
    await atomicWriteJson(join(dir, "models.json"), {
      providers: {
        "openrouter-preset": makeProviderEntry([glmModel()]),
      },
    });
  });
  assert.strictEqual(registered.size, 1);
  assert.ok(registered.has("openrouter-preset"));
});

test("5.2 Startup perf: factory with 20 pins completes in <50ms", async () => {
  const dir = await mkdtemp(join(tmpdir(), "or-pin-perf-"));
  const saved = process.env.PI_CODING_AGENT_DIR;
  process.env.PI_CODING_AGENT_DIR = dir;
  try {
    // Build a models.json with 20 openrouter-* pins
    const providers: Record<string, ProviderEntry> = {};
    for (let i = 0; i < 20; i++) {
      providers[`openrouter-provider-${i}`] = makeProviderEntry([glmModel()]);
    }
    writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { key: "sk-auth" } }));
    await atomicWriteJson(join(dir, "models.json"), { providers });

    const pi = {
      on: () => {},
      registerProvider: (name: string, entry: ProviderEntry) => {},
      registerCommand: () => {},
    } as unknown as ExtensionAPI;

    const start = performance.now();
    openrouterPinExtension(pi);
    const elapsed = performance.now() - start;
    assert.ok(elapsed < 50, `factory with 20 pins should complete in <50ms, got ${elapsed.toFixed(2)}ms`);
  } finally {
    if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = saved;
    await rm(dir, { recursive: true, force: true });
  }
});

test("5.3 Multi-pin mixed-validity: only openrouter-* with models re-registered", async () => {
  const registered = await factoryHarness(async (dir) => {
    writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api_key", key: "sk-auth" } }));
    await atomicWriteJson(join(dir, "models.json"), {
      providers: {
        "anthropic": makeProviderEntry([glmModel()]),
        "openrouter-novita": makeProviderEntry([glmModel()]),
        "openrouter-empty": { ...makeProviderEntry([]), models: [] },
        "openrouter-decart": makeProviderEntry([glmModel({ id: "z-ai/glm-5.2:free" })]),
      },
    });
  });
  assert.strictEqual(registered.size, 2);
  assert.ok(registered.has("openrouter-novita"));
  assert.ok(registered.has("openrouter-decart"));
});
}); // end stretch factory repros (PI_CODING_AGENT_DIR serial)

// ---------------------------------------------------------------------------
// 6.x — Validation
// ---------------------------------------------------------------------------

test("resolveFactoryKey: out-of-scope $ENV interpolation treated as literal", () => {
  assert.strictEqual(resolveFactoryKey("$OTHER_ENV", undefined), "$OTHER_ENV");
});

test("collectPinnedProviders: matches pinnedProviderSlugs conventions", async () => {
  const snapshot: ModelsJson = {
    providers: {
      "openrouter-novita": makeProviderEntry([glmModel()]),
      "openrouter-novita-plus": makeProviderEntry([deepseekModel()]),
      "openrouter-preset": makeProviderEntry([glmModel()]),
      "openrouter-empty": { ...makeProviderEntry([]), models: [] },
    },
  };
  const pinned = collectPinnedProviders(snapshot);
  const names = pinned.map(([n]) => n);
  assert.ok(names.includes("openrouter-novita"));
  assert.ok(names.includes("openrouter-novita-plus"));
  assert.ok(names.includes("openrouter-preset"));
  assert.ok(!names.includes("openrouter-empty"));
});
