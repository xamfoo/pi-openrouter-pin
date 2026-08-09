/**
 * OpenRouter API client and API-key resolution.
 *
 * The client holds per-extension-instance caches (catalog, user models,
 * endpoints) — never module scope — so reloads, RPC + TUI sharing one
 * process, and tests each get independent state.
 */
import type { ModelRegistry } from "@earendil-works/pi-coding-agent";
import { slugify, type RawModelShape } from "./config.ts";
import { readJsonFile, type ModelsJson } from "./files.ts";

export const OR_BASE_URL = "https://openrouter.ai/api/v1";
export const OR_MODELS_URL = `${OR_BASE_URL}/models`;
export const PROVIDER_PREFIX = "openrouter-";
export const FETCH_TIMEOUT_MS = 15000;
export const CATALOG_CACHE_TTL_MS = 60_000;
export const ENDPOINT_CACHE_TTL_MS = 60_000;
/** How long a server-side ?q= search result stays memoized (back-and-forth typing). */
export const SEARCH_CACHE_TTL_MS = 30_000;

export type RawModel = RawModelShape;

export interface RawEndpoint {
  name?: string;
  provider_name?: string;
  quantization?: string;
}

export interface EndpointsResult {
  endpoints: RawEndpoint[];
  message?: string;
}

/**
 * Result of validating that a provider serves a model (and quant, if any).
 * Three explicit arms — never a truthy "ok" standing in for unknown:
 *
 *   - `ok`          — validated; `quant` may be normalized to the endpoint's
 *                     canonical spelling, `note` may carry a caveat.
 *   - `unvalidated` — proceeding without validation (no API key, or the
 *                     endpoints API was unavailable); `note` says why.
 *   - `error`       — refused: provider or quant is not served.
 *
 * Unknown state is always an explicit arm here, just as
 * `fetchUserModelIds` returns `null` for unknown (see item 6 convention).
 */
export type EndpointValidation =
  | { status: "ok"; quant?: string; note?: string }
  | { status: "unvalidated"; quant?: string; note: string }
  | { status: "error"; message: string };

async function fetchJson(url: string, apiKey?: string): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(url, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} for ${url}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

export class OpenRouterClient {
  private readonly catalogTtlMs: number;
  private readonly endpointTtlMs: number;
  private readonly searchTtlMs: number;
  private catalog: RawModel[] | null = null;
  private catalogAt = 0;
  private userModels: Set<string> | null = null;
  private userModelsAt = 0;
  private endpoints = new Map<string, { at: number; value: EndpointsResult }>();
  private lastSearch: { q: string; at: number; models: RawModel[] } | null = null;

  constructor(catalogTtlMs: number, endpointTtlMs: number, searchTtlMs: number = SEARCH_CACHE_TTL_MS) {
    // Explicit fields (not parameter properties): Node's strip-only TS mode
    // cannot transform parameter properties, and this module is imported by
    // tests that run under plain `node --test`.
    this.catalogTtlMs = catalogTtlMs;
    this.endpointTtlMs = endpointTtlMs;
    this.searchTtlMs = searchTtlMs;
  }

  async fetchCatalog(force = false): Promise<RawModel[]> {
    const now = Date.now();
    if (!force && this.catalog && now - this.catalogAt < this.catalogTtlMs) {
      return this.catalog;
    }
    const payload = (await fetchJson(OR_MODELS_URL)) as { data?: RawModel[] };
    this.catalog = payload.data ?? [];
    this.catalogAt = now;
    return this.catalog;
  }

  /**
   * Free-text search by model name or slug (GET /models?q=…). The catalog
   * endpoint is public — no API key needed. Returns full model objects, so
   * search results are drop-in RawModelShape values for the picker, and the
   * same shape the schema tests pin the extension to.
   *
   * OpenRouter documents no minimum query length, so callers decide their
   * own threshold. The wizard currently ranks the full locally-fetched
   * catalog instead (the catalog endpoint returns the complete model list,
   * so local matching is instant and complete); this method remains as
   * server-side search API surface for freshness/relevance when needed. The
   * most recent query is memoized briefly so back-and-forth typing doesn't
   * re-fetch the same page.
   */
  async searchModels(query: string): Promise<RawModel[]> {
    const trimmed = query.trim();
    if (!trimmed) return [];
    const now = Date.now();
    if (this.lastSearch && this.lastSearch.q === trimmed && now - this.lastSearch.at < this.searchTtlMs) {
      return this.lastSearch.models;
    }
    const payload = (await fetchJson(`${OR_MODELS_URL}?q=${encodeURIComponent(trimmed)}`)) as { data?: RawModel[] };
    const models = payload.data ?? [];
    this.lastSearch = { q: trimmed, at: now, models };
    return models;
  }

  async fetchRawModel(modelId: string): Promise<RawModel | null> {
    const catalog = await this.fetchCatalog();
    return catalog.find((m) => m.id === modelId) ?? null;
  }

  /**
   * Model ids this account can use, as filtered by its provider preferences,
   * privacy settings, and guardrails (GET /models/user). Returns null when
   * unknown (no API key or request failure) so callers can skip the check.
   * Unknown is an explicit `null` — matching the `status: "unvalidated"`
   * arm of validateEndpoint — never a truthy stand-in.
   */
  async fetchUserModelIds(apiKey: string | undefined): Promise<Set<string> | null> {
    if (!apiKey) return null;
    const now = Date.now();
    if (this.userModels && now - this.userModelsAt < this.catalogTtlMs) return this.userModels;
    try {
      const payload = (await fetchJson(`${OR_BASE_URL}/models/user`, apiKey)) as { data?: Array<{ id: string }> };
      this.userModels = new Set((payload.data ?? []).map((m) => m.id));
      this.userModelsAt = now;
      return this.userModels;
    } catch (err) {
      console.warn(`[openrouter-pin] /models/user unavailable: ${err instanceof Error ? err.message : String(err)}`);
      return null;
    }
  }

  /**
   * Fetch the endpoints a model is served from. Without an API key, or on an
   * endpoints API failure, returns an empty list plus a `message` explaining
   * why, so callers can fall back to typing a provider.
   */
  async fetchModelEndpoints(modelId: string, apiKey?: string): Promise<EndpointsResult> {
    if (!apiKey) return { endpoints: [], message: "no API key set — provider list not filtered" };
    const cached = this.endpoints.get(modelId);
    if (cached && Date.now() - cached.at < this.endpointTtlMs) return cached.value;
    let payload: { data?: { endpoints?: RawEndpoint[] } };
    try {
      // OpenRouter's endpoints route is slash-separated: encoding the "/" in the
      // model id (e.g. openai/gpt-5.2) as %2F makes the route 404. Encode each
      // segment separately instead. The response is the model object with the
      // endpoint list at data.endpoints.
      const endpointsUrl = `${OR_BASE_URL}/models/${modelId.split("/").map(encodeURIComponent).join("/")}/endpoints`;
      payload = (await fetchJson(endpointsUrl, apiKey)) as { data?: { endpoints?: RawEndpoint[] } };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const reason = /HTTP 404/.test(msg)
        ? `OpenRouter lists no endpoints for ${modelId}`
        : msg;
      const value: EndpointsResult = {
        endpoints: [],
        message: `endpoint list unavailable (${reason})`,
      };
      this.endpoints.set(modelId, { at: Date.now(), value });
      return value;
    }
    const value: EndpointsResult = { endpoints: payload.data?.endpoints ?? [] };
    this.endpoints.set(modelId, { at: Date.now(), value });
    return value;
  }

  /** Endpoint validation: best-effort with a key, unvalidated without one. */
  async validateEndpoint(
    modelId: string,
    slug: string,
    quant?: string,
    apiKey?: string,
  ): Promise<EndpointValidation> {
    if (!apiKey) {
      return { status: "unvalidated", quant, note: "no API key set — pinning unvalidated" };
    }
    const { endpoints, message } = await this.fetchModelEndpoints(modelId, apiKey);
    if (message) {
      return {
        status: "unvalidated",
        quant,
        note: `${message} — routing applied unvalidated; a wrong provider will surface as an OpenRouter 400 on first request`,
      };
    }
    const bySlug = endpoints.filter((e) => slugify(e.provider_name ?? "") === slug);
    const availableSlugs = [...new Set(endpoints.map((e) => slugify(e.provider_name ?? "")))].filter(Boolean).sort();
    if (bySlug.length === 0) {
      return {
        status: "error",
        message: `provider "${slug}" does not serve ${modelId}. Available: ${availableSlugs.join(", ") || "none listed"}`,
      };
    }
    if (quant) {
      const exact = bySlug.find((e) => (e.quantization ?? "").toLowerCase() === quant.toLowerCase());
      if (!exact) {
        const availableQuants = [...new Set(bySlug.map((e) => e.quantization ?? "?")).values()].sort();
        return {
          status: "error",
          message: `provider "${slug}" for ${modelId} has no "${quant}" endpoint. Available quantizations: ${availableQuants.join(", ") || "unspecified"}`,
        };
      }
      return { status: "ok", quant: exact.quantization ?? quant };
    }
    return { status: "ok" };
  }
}

/**
 * Account-declared providers OpenRouter must not route to (comma-separated
 * slugs in OPENROUTER_NON_ROUTABLE_PROVIDERS, e.g. "openai,anthropic"). The
 * endpoints API does not expose account-level provider eligibility, so this
 * is the reliable signal for the "(non-routable)" suffix.
 */
const NON_ROUTABLE_PROVIDERS = new Set(
  (process.env.OPENROUTER_NON_ROUTABLE_PROVIDERS ?? "")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter((s) => /^[a-z0-9-]+$/.test(s)),
);

/** Picker label for a provider: its slug, or "slug (non-routable)" when declared for this account. */
export function providerLabel(slug: string): string {
  return NON_ROUTABLE_PROVIDERS.has(slug) ? `${slug} (non-routable)` : slug;
}

/** Slugs of providers the user has already pinned (from models.json). */
export async function pinnedProviderSlugs(modelsPath: string): Promise<string[]> {
  const models = await readJsonFile<ModelsJson>(modelsPath);
  const slugs = new Set<string>();
  for (const name of Object.keys(models?.providers ?? {})) {
    if (!name.startsWith(PROVIDER_PREFIX)) continue;
    // A relaxed pin is openrouter-<slug>-plus: the -plus suffix is this
    // extension's marker, not the provider's name — strip it so the slug can
    // be offered as a completion candidate for a NEW pin. Without this, a
    // strict re-pin of "novita-plus" would be written unvalidated (no key)
    // and 400 on its first request.
    const slug = name.slice(PROVIDER_PREFIX.length).replace(/-plus$/, "");
    if (/^[a-z0-9-]+$/.test(slug)) slugs.add(slug);
  }
  return [...slugs].sort();
}

/**
 * Resolve the OpenRouter API key through pi's own auth machinery
 * (getApiKeyForProvider handles env vars, $ENV interpolation, !command,
 * auth.json credentials, and OAuth), so a key configured any way other than
 * a plain env var still validates. Falls back to the env var last.
 * `pinProviderName` is the fully computed provider name of the pin being
 * validated (or undefined before a pin exists).
 */
export async function resolveOpenRouterApiKey(
  modelRegistry: ModelRegistry,
  pinProviderName?: string,
): Promise<string | undefined> {
  const candidates = [pinProviderName, "openrouter"].filter((x): x is string => Boolean(x));
  for (const provider of candidates) {
    try {
      const key = await modelRegistry.getApiKeyForProvider(provider);
      if (key) return key;
    } catch {
      // try the next candidate
    }
  }
  return process.env.OPENROUTER_API_KEY;
}
