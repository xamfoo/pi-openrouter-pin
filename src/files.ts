/**
 * JSONC-tolerant file read/write helpers for models.json and settings.json.
 */
import { existsSync } from "node:fs";
import { readFile, writeFile, rename, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import type { ModelConfig } from "./config.ts";

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
    return JSON.parse(stripJsonComments(raw)) as T;
  } catch (err) {
    throw new Error(`Cannot read ${path}: ${err instanceof Error ? err.message : String(err)}`);
  }
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
