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

/** Shape of a pi-valid api_key credential from auth.json. */
interface ApiKeyCredential {
  type: "api_key";
  key?: string;
  env?: Record<string, string>;
}

/** Shape of a pi-valid oauth credential from auth.json. */
interface OAuthCredential {
  type: "oauth";
  access: string;
  refresh: string;
  expires: number;
}

/**
 * Synchronously resolve a `$VAR` or `${VAR}` template using `credEnv` first,
 * then `process.env`. `$$` escapes a literal dollar. Unresolvable names stay
 * literal so pi reports `configured:false` instead of receiving a wrong key.
 */
function resolveTemplate(input: string, credEnv: Record<string, string> | undefined): string {
  let out = "";
  let i = 0;
  while (i < input.length) {
    const c = input[i];
    if (c === "$" && i + 1 < input.length) {
      const next = input[i + 1];
      if (next === "$") {
        out += "$";
        i += 2;
        continue;
      }
      if (next === "{") {
        const close = input.indexOf("}", i + 2);
        if (close !== -1) {
          const varName = input.slice(i + 2, close);
          const resolved =
            credEnv?.[varName] ?? process.env[varName];
          if (resolved !== undefined) {
            out += resolved;
          } else {
            // Unresolvable: keep literal
            out += input.slice(i, close + 1);
          }
          i = close + 1;
          continue;
        }
      }
      // `$VAR` (no braces)
      const rest = input.slice(i + 1);
      const match = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/);
      if (match) {
        const varName = match[0];
        const resolved = credEnv?.[varName] ?? process.env[varName];
        if (resolved !== undefined) {
          out += resolved;
        } else {
          // Unresolvable: keep literal
          out += "$" + varName;
        }
        i += 1 + varName.length;
        continue;
      }
    }
    out += c;
    i++;
  }
  return out;
}

/**
 * Resolve the effective OpenRouter API key from env and a parsed auth object.
 *
 * Priority:
 *   1. trim(envVal) if non-empty after trim
 *   2. Tagged credential:
 *      - `type:"oauth"` → trimmed `access` (regardless of `expires`/`refresh`)
 *      - `type:"api_key"` → `$VAR`/`${VAR}` resolved via `credential.env` then
 *        `process.env`; `!command` passed through literally; `$$` escapes `$`
 *   3. Untagged leniency: `{key}` string (backwards compat). Untagged `$VAR`
 *      values are returned literally — the untagged path does not resolve
 *      templates, matching legacy behavior. Use tagged `api_key` for $VAR support.
 *   4. undefined
 *
 * Pure: `envVal` is passed in, never read from `process.env`, so this is
 * testable without mutating the environment.
 *
 * Whitespace-only values are treated as absent. Non-string values from
 * auth.json are ignored. Command credentials are never executed in the
 * factory — they are passed through literally.
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
    if (!openrouter) return undefined;

    // Tagged credential shapes
    const credType = openrouter.type;
    if (credType === "oauth") {
      const access = openrouter.access as string | undefined;
      if (typeof access === "string") {
        const trimmed = access.trim();
        if (trimmed) return trimmed;
      }
      return undefined;
    }

    if (credType === "api_key") {
      const key = openrouter.key as string | undefined;
      const credEnv = openrouter.env as Record<string, string> | undefined;
      // Validate env is a plain string-keyed record before passing to resolveTemplate.
      const validCredEnv = credEnv !== undefined && isRecordOfStrings(credEnv) ? credEnv : undefined;
      if (typeof key === "string") {
        const trimmed = key.trim();
        if (!trimmed) return undefined;
        if (trimmed.startsWith("!")) {
          // Pass through literally — never execute in the factory
          return trimmed;
        }
        return resolveTemplate(trimmed, validCredEnv);
      }
      return undefined;
    }

    // Untagged leniency: legacy {key: ...} shape
    const legacyKey = openrouter.key;
    if (typeof legacyKey === "string") {
      const trimmed = legacyKey.trim();
      if (trimmed) return trimmed;
    }
  }

  return undefined;
}

/** Check that a value is a plain string-keyed record (Record<string, string>). */
function isRecordOfStrings(value: unknown): value is Record<string, string> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  for (const v of Object.values(value as Record<string, unknown>)) {
    if (typeof v !== "string") return false;
  }
  return true;
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
