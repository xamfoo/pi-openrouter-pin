/**
 * JSONC-tolerant file read/write helpers for models.json, settings.json, and
 * auth.json — plus the pure auth-key resolution and pinned-provider
 * collection/registration that derive from those files.
 *
 * This is the single owner of the file→data seam. `api.ts` is the network
 * client; anything that reads or writes pi's on-disk state lives here so a
 * test can exercise it without touching the network or mutating `process.env`
 * (env is passed in as an argument, never captured).
 */
import { existsSync, readFileSync } from "node:fs";
import { readFile, writeFile, rename, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { PROVIDER_PREFIX, type ModelConfig } from "./config.ts";

// ---------------------------------------------------------------------------
// Auth key resolution
// ---------------------------------------------------------------------------

/**
 * Resolve the effective OpenRouter API key from env and a parsed auth object.
 *
 * Priority:
 *   1. trim(envVal) if non-empty after trim
 *   2. trim(auth.openrouter.key) if it is a non-empty string
 *   3. undefined
 *
 * Pure: `envVal` is passed in, never read from `process.env`, so this is
 * testable without mutating the environment.
 *
 * Does NOT interpret $ENV interpolation or !command — those are
 * pin-time ModelRegistry concerns. Whitespace-only values are
 * treated as absent. Non-string values from auth.json are ignored.
 */
export function resolveFactoryKey(
  envVal: string | undefined,
  auth: unknown,
): string | undefined {
  const env = envVal?.trim();
  if (env) return env;

  if (auth && typeof auth === "object") {
    const authObj = auth as Record<string, unknown>;
    const openrouter = authObj.openrouter as Record<string, unknown> | undefined;
    const key = openrouter?.key;
    if (typeof key === "string") {
      const trimmed = key.trim();
      if (trimmed) return trimmed;
    }
  }

  return undefined;
}

/**
 * Read and parse auth.json (JSONC-tolerant). Missing file → `undefined`.
 * Malformed JSON → warns once via `warnFile` and returns `undefined`.
 * The resolved key is the caller's job: `resolveFactoryKey(env, this)`.
 */
export function readAuthJsonSync(agentDir: string): unknown {
  const authPath = join(agentDir, "auth.json");
  if (!existsSync(authPath)) return undefined;

  try {
    const raw = readFileSync(authPath, "utf-8");
    return parseJsonc<Record<string, unknown>>(raw) as unknown;
  } catch (err) {
    warnFile(authPath, err);
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// JSONC parsing
// ---------------------------------------------------------------------------

/** Strip JSONC comments and parse as JSON. Throws on malformed input. */
function parseJsonc<T>(raw: string): T {
  const withoutBom = raw.charCodeAt(0) === 0xfeff ? raw.slice(1) : raw;
  const stripped = stripJsonComments(withoutBom);
  return JSON.parse(stripped) as T;
}

/**
 * Synchronously read a models.json file with JSONC tolerance.
 * Returns `null` if the file is missing. Throws on malformed JSON.
 */
export function readModelsJsonSync(path: string): ModelsJson | null {
  if (!existsSync(path)) return null;
  const raw = readFileSync(path, "utf-8");
  return parseJsonc<ModelsJson>(raw);
}

export interface ProviderEntry {
  baseUrl: string;
  api: string;
  apiKey: string;
  headers?: Record<string, string>;
  models: ModelConfig[];
}

export interface ModelsJson {
  providers?: Record<string, ProviderEntry>;
}

export interface SettingsJson {
  defaultProvider?: string;
  defaultModel?: string;
  enabledModels?: string[];
  [key: string]: unknown;
}

/** Remove // and /* * / comments outside strings (JSONC tolerance). */
export function stripJsonComments(input: string): string {
  let out = "";
  let inString = false;
  let i = 0;
  while (i < input.length) {
    const c = input[i];
    const next = input[i + 1];
    if (inString) {
      out += c;
      if (c === "\\") {
        out += next ?? "";
        i += 2;
        continue;
      }
      if (c === '"') inString = false;
      i++;
      continue;
    }
    if (c === '"') {
      inString = true;
      out += c;
      i++;
      continue;
    }
    if (c === "/" && next === "/") {
      while (i < input.length && input[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && next === "*") {
      i += 2;
      while (i < input.length && !(input[i] === "*" && input[i + 1] === "/")) i++;
      i += 2;
      continue;
    }
    out += c;
    i++;
  }
  return out;
}

export async function readJsonFile<T>(path: string): Promise<T | null> {
  try {
    if (!existsSync(path)) return null;
    const raw = await readFile(path, "utf-8");
    return parseJsonc<T>(raw);
  } catch (err) {
    throw new Error(`Cannot read ${path}: ${err instanceof Error ? err.message : String(err)}`);
  }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

/**
 * Warn about a malformed/unreadable file — once per process per path, even if
 * a later read of the same path fails again (a dotfile error is worth exactly
 * one startup message; re-reading the same broken file doesn't deserve more).
 *
 * `warnedPaths` is deliberately never cleared: the dedupe key is the absolute
 * path, and each process reads only a small fixed set (models.json, auth.json),
 * so the set stays bounded — no realistic unbounded-growth concern.
 */
const warnedPaths = new Set<string>();

export function warnFile(path: string, err: unknown): void {
  if (warnedPaths.has(path)) return;
  warnedPaths.add(path);
  const msg = err instanceof Error ? err.message : String(err);
  console.warn(`[pi-openrouter-pin] cannot read ${path}: ${msg}`);
}

let tmpSeq = 0;

export async function atomicWriteJson(path: string, data: unknown): Promise<void> {
  const dir = dirname(path);
  await mkdir(dir, { recursive: true });
  // pid + ms can collide for two writes in the same process in the same
  // millisecond (a pin landing during the fire-and-forget session_start
  // refresh) — the monotonic counter keeps tmp names unique.
  const tmp = `${path}.tmp-${process.pid}-${Date.now()}-${tmpSeq++}`;
  await writeFile(tmp, JSON.stringify(data, null, 2) + "\n", "utf-8");
  await rename(tmp, path);
}

// ---------------------------------------------------------------------------
// Pinned-provider collection & registration
// ---------------------------------------------------------------------------

/**
 * Collect pinned OpenRouter providers from a models.json snapshot.
 *
 * A provider is a "pin" iff its name starts with `openrouter-` and
 * its `models` is a non-empty array. Returns entries as
 * `[name, ProviderEntry]` tuples. Covers strict (`openrouter-<slug>`),
 * relaxed (`openrouter-<slug>-plus`), and `openrouter-preset`.
 */
export function collectPinnedProviders(
  snapshot: ModelsJson | null,
): Array<[string, ProviderEntry]> {
  const result: Array<[string, ProviderEntry]> = [];
  if (!snapshot?.providers) return result;

  for (const [name, provider] of Object.entries(snapshot.providers)) {
    if (!name.startsWith(PROVIDER_PREFIX)) continue;
    if (!Array.isArray(provider.models) || provider.models.length === 0) continue;
    result.push([name, provider]);
  }

  return result;
}

/**
 * Register pinned OpenRouter providers with the effective API key.
 *
 * Reads models.json via `readModelsJsonSync` (JSONC-tolerant sync read),
 * collects `openrouter-*` providers with non-empty models, and calls
 * `pi.registerProvider(name, effectiveKey ? {...entry, apiKey: effectiveKey} : entry)`.
 * Malformed models.json warns via `warnFile` and returns silently.
 */
export function registerPinnedProviders(
  pi: ExtensionAPI,
  modelsPath: string,
  effectiveKey: string | undefined,
): void {
  let snapshot: ModelsJson | null;
  try {
    snapshot = readModelsJsonSync(modelsPath);
  } catch (err) {
    warnFile(modelsPath, err);
    return;
  }

  for (const [name, entry] of collectPinnedProviders(snapshot)) {
    pi.registerProvider(name, effectiveKey ? { ...entry, apiKey: effectiveKey } : entry);
  }
}
