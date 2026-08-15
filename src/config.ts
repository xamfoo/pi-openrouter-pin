/**
 * Pure config builders for pi-openrouter-pin.
 *
 * This module has ZERO imports (no node, no pi packages) so it can be
 * unit-tested in isolation with a plain `node --test` run against a catalog
 * fixture. Everything here is a pure function of its inputs.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** The subset of an OpenRouter /models catalog entry the pin uses. */
export interface RawModelShape {
  id: string;
  name?: string;
  context_length?: number;
  top_provider?: { max_completion_tokens?: number };
  per_request_limits?: { max_tokens?: number };
  pricing?: Record<string, string>;
  architecture?: { input_modalities?: string[] };
  supported_parameters?: string[];
}

/**
 * The subset of an OpenRouter /models/<id>/endpoints entry the pin uses.
 * Endpoint pricing & limits are provider-specific (the catalog's model-level
 * `pricing` is an aggregate that often belongs to a different provider), so
 * a pinned provider's recorded cost/limits must come from its endpoint.
 */
export interface RawEndpointShape {
  name?: string;
  /**
   * OpenRouter's routing slug for this endpoint (e.g. "google-vertex/global",
   * "novita/fp8"). Since OpenRouter renamed display names ("Google Vertex AI"
   * → "Google"), provider_name is no longer a reliable slug source; the tag's
   * first path segment is the provider's base routing slug.
   */
  tag?: string;
  provider_name?: string;
  quantization?: string;
  pricing?: Record<string, string>;
  max_completion_tokens?: number | null;
  context_length?: number | null;
  supported_parameters?: string[];
}

/** The real `openRouterRouting` schema, as accepted by pi's model config. */
export interface OpenRouterRouting {
  allow_fallbacks?: boolean;
  require_parameters?: boolean;
  data_collection?: "deny" | "allow";
  zdr?: boolean;
  enforce_distillable_text?: boolean;
  order?: string[];
  only?: string[];
  ignore?: string[];
  quantizations?: string[];
  sort?: string | { by?: string; partition?: string };
  max_price?: {
    prompt?: number | string;
    completion?: number | string;
    image?: number | string;
    audio?: number | string;
    request?: number | string;
  };
  preferred_min_throughput?: number | { p50?: number; p75?: number; p90?: number; p99?: number };
  preferred_max_latency?: number | { p50?: number; p75?: number; p90?: number; p99?: number };
}

export interface ModelConfigCompat {
  /** Explicit so reasoning pins keep OpenRouter's thinking shape even if pi's baseUrl auto-detection changes. */
  thinkingFormat?: "openrouter";
  openRouterRouting: OpenRouterRouting;
}

export interface ModelConfig {
  id: string;
  name: string;
  reasoning: boolean;
  input: Array<"text" | "image">;
  contextWindow: number;
  maxTokens: number;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  compat: ModelConfigCompat;
}

export interface PinOptions {
  modelId: string;
  slug: string;
  quant?: string;
  name?: string;
  isDefault: boolean;
  allowFallbacks?: boolean;
  order?: string[];
  ignore?: string[];
  /** Privacy flag, orthogonal to routing: works on any pin (strict or relaxed). Strict pins surface it as data_collection=… in /openrouter-pins and get a pin-time "routing stays strict" note, since the provider name only reflects routing. */
  dataCollection?: "allow" | "deny";
  /**
   * The validated endpoint for the anchor provider (and quant, if any). When
   * present, pricing & limits are recorded from it — not the catalog aggregate
   * — so a pin to e.g. novita reflects novita's prices, not whichever provider
   * the model-level `pricing` happens to summarize. Absent when the endpoint
   * could not be validated (no API key); the catalog aggregate is then used as
   * the best available fallback.
   */
  endpoint?: RawEndpointShape;
}

export const COMMON_QUANTIZATIONS = ["none", "fp8", "bf16", "fp16", "int4"];

// ---------------------------------------------------------------------------
// Slug helpers
// ---------------------------------------------------------------------------

/** Normalize an OpenRouter provider slug: "NVIDIA" → "nvidia", "Z AI" → "z-ai". */
export function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

/**
 * The provider's base routing slug for an endpoint: the first path segment of
 * the endpoint's `tag` when present ("google-vertex/global" → "google-vertex"),
 * else the slugified provider_name ("Novita" → "novita"). The tag is
 * OpenRouter's routing slug — a base slug matches all of a provider's
 * variants/regions — while provider_name is a display name that OpenRouter may
 * rename ("Google Vertex AI" → "Google"), so the tag wins when available.
 */
export function endpointSlug(e: RawEndpointShape): string {
  const tag = e.tag?.trim();
  if (tag) {
    const base = slugify(tag.split("/")[0]);
    if (base) return base;
  }
  return slugify(e.provider_name ?? "");
}

/** True when the pin's routing policy is not a strict single-provider pin. */
export function isRelaxedPin(opts: Pick<PinOptions, "allowFallbacks" | "order" | "ignore">): boolean {
  return Boolean(opts.allowFallbacks) || (opts.order?.length ?? 0) > 0 || (opts.ignore?.length ?? 0) > 0;
}

/**
 * Provider name for a pin. Strict single-provider pins (only=[slug]) get
 * `openrouter-<slug>`; any relaxed policy (fallbacks, order, ignore) gets
 * `openrouter-<slug>-plus`, so the name never lies about routing to a
 * single provider.
 */
export function providerNameFor(
  slug: string,
  policy?: Pick<PinOptions, "allowFallbacks" | "order" | "ignore">,
): string {
  const base = `openrouter-${slugify(slug)}`;
  return policy && isRelaxedPin(policy) ? `${base}-plus` : base;
}

// ---------------------------------------------------------------------------
// Cost / pricing & limits conversion
// ---------------------------------------------------------------------------

/** OpenRouter per-token USD → pi per-million-token dollars, rounded to 2dp. */
const perMillion = (s?: string): number => {
  const v = parseFloat(s ?? "");
  return Number.isFinite(v) ? Math.round(v * 1_000_000 * 100) / 100 : 0;
};

/** OpenRouter per-token pricing → pi per-million-token cost, rounded to 2dp. */
export function toCost(pricing: RawModelShape["pricing"] | undefined): ModelConfig["cost"] {
  const p = pricing ?? {};
  return {
    input: perMillion(p.prompt),
    output: perMillion(p.completion),
    cacheRead: perMillion(p.input_cache_read),
    cacheWrite: perMillion(p.input_cache_write),
  };
}

/**
 * Cost from an endpoint's pricing, falling back to the prior cost for keys
 * the endpoint omits. Endpoint pricing carries prompt/completion/
 * input_cache_read (and a discount already applied), but not
 * input_cache_write (a model-level field), so the model-level value is
 * preserved for it. A present-but-"0" endpoint field is honored (distinguished
 * from an absent key) so a provider that genuinely charges 0 is recorded as 0.
 */
export function mergeEndpointCost(
  endpoint: RawEndpointShape,
  fallback: ModelConfig["cost"],
): ModelConfig["cost"] {
  const p = endpoint.pricing ?? {};
  const or = (s: string | undefined, fb: number): number => (s !== undefined ? perMillion(s) : fb);
  return {
    input: or(p.prompt, fallback.input),
    output: or(p.completion, fallback.output),
    cacheRead: or(p.input_cache_read, fallback.cacheRead),
    cacheWrite: or(p.input_cache_write, fallback.cacheWrite),
  };
}

/**
 * The published pricing & limits snapshot of a catalog entry (cost,
 * contextWindow, maxTokens): what a fresh pin would record today. When an
 * endpoint is supplied (a validated provider for this model), its
 * provider-specific pricing & limits override the catalog aggregate — the
 * catalog's model-level `pricing`/`top_provider` summarize a different
 * provider for some models. Shared by pin-time config building and the
 * startup refresh so both always agree on per-provider truth.
 */
export function toPricingAndLimits(
  raw: RawModelShape,
  endpoint?: RawEndpointShape,
): Pick<ModelConfig, "contextWindow" | "maxTokens" | "cost"> {
  const base = {
    contextWindow: raw.context_length ?? 128_000,
    maxTokens: raw.top_provider?.max_completion_tokens ?? raw.per_request_limits?.max_tokens ?? 16_384,
    cost: toCost(raw.pricing),
  };
  if (!endpoint) return base;
  return {
    contextWindow: endpoint.context_length ?? base.contextWindow,
    maxTokens: endpoint.max_completion_tokens ?? base.maxTokens,
    cost: endpoint.pricing ? mergeEndpointCost(endpoint, base.cost) : base.cost,
  };
}

/**
 * Pricing & limits for a pinned model from its validated endpoint, falling
 * back to the previously-stored snapshot for fields the endpoint omits.
 * Used by the startup refresh, which has no catalog raw in hand — only the
 * stored pin — so the fallback keeps unchanged fields stable rather than
 * zeroing them.
 */
export function pricingAndLimitsFromEndpoint(
  endpoint: RawEndpointShape,
  fallback: Pick<ModelConfig, "contextWindow" | "maxTokens" | "cost">,
): Pick<ModelConfig, "contextWindow" | "maxTokens" | "cost"> {
  return {
    contextWindow: endpoint.context_length ?? fallback.contextWindow,
    maxTokens: endpoint.max_completion_tokens ?? fallback.maxTokens,
    cost: endpoint.pricing ? mergeEndpointCost(endpoint, fallback.cost) : fallback.cost,
  };
}

/**
 * The anchor provider slug for a routing policy: the sole `only` provider
 * for strict pins, or the head of `order` for relaxed ones. Endpoint pricing &
 * limits are recorded for the anchor, since a pin always means "at least
 * prefer this provider".
 */
export function anchorSlug(routing: OpenRouterRouting | undefined): string | undefined {
  if (!routing) return undefined;
  return routing.only?.[0] ?? routing.order?.[0];
}

/**
 * Find the endpoint for a provider slug (and optional quantization) among a
 * model's endpoints. When `quant` is given but no endpoint matches it
 * exactly, returns undefined (the provider may have dropped that quant)
 * rather than silently substituting another endpoint's pricing. Without a
 * quant, the first endpoint for the slug is the representative default.
 */
export function findEndpoint(
  endpoints: RawEndpointShape[],
  slug: string,
  quant?: string,
): RawEndpointShape | undefined {
  const bySlug = endpoints.filter((e) => endpointSlug(e) === slug);
  if (bySlug.length === 0) return undefined;
  if (!quant) return bySlug[0];
  return bySlug.find((e) => (e.quantization ?? "").toLowerCase() === quant.toLowerCase());
}

// ---------------------------------------------------------------------------
// Model config building
// ---------------------------------------------------------------------------

export function toModelConfig(raw: RawModelShape, opts: PinOptions): ModelConfig {
  const slug = slugify(opts.slug);
  const supported = raw.supported_parameters ?? [];
  const modalities = (raw.architecture?.input_modalities ?? ["text"]).filter((m): m is "text" | "image" =>
    m === "text" || m === "image",
  );
  // Routing policy. The anchor provider (the positional slug) always heads
  // the preference list; order/ignore/fallbacks are explicit relaxations of
  // the strict single-provider pin:
  //   strict    → only=[slug], fallbacks off
  //   order     → order=[slug, ...user], no only, fallbacks on (try in order,
  //               then fall back)
  //   ignore    → only=[slug], ignore=[...], fallbacks on (ignore only matters
  //               when fallback is possible)
  //   fallbacks → only=[slug], fallbacks on
  const userOrder = [...new Set((opts.order ?? []).map(slugify).filter(Boolean))].filter((s) => s !== slug);
  const order = userOrder.length > 0 ? [slug, ...userOrder] : undefined;
  const ignore = (opts.ignore ?? []).map(slugify).filter(Boolean);
  const fallbacks = opts.allowFallbacks === true || order !== undefined || ignore.length > 0;
  const routing: OpenRouterRouting = { allow_fallbacks: fallbacks };
  if (order) routing.order = order;
  else routing.only = [slug];
  if (ignore.length > 0) routing.ignore = ignore;
  if (opts.quant) routing.quantizations = [opts.quant];
  if (opts.dataCollection) routing.data_collection = opts.dataCollection;
  const caps = toPricingAndLimits(raw, opts.endpoint);
  return {
    id: raw.id,
    name: opts.name ?? `${raw.name ?? raw.id} (${slug}${opts.quant ? ` ${opts.quant}` : ""})`,
    reasoning: supported.includes("reasoning") || supported.includes("include_reasoning"),
    input: modalities,
    contextWindow: caps.contextWindow,
    maxTokens: caps.maxTokens,
    cost: caps.cost,
    compat: {
      thinkingFormat: "openrouter",
      openRouterRouting: routing,
    },
  };
}

export interface ProviderEntryBase {
  baseUrl: string;
  api: string;
  apiKey: string;
  models: ModelConfig[];
}

export function baseProviderConfig(): Omit<ProviderEntryBase, "models"> {
  return {
    baseUrl: "https://openrouter.ai/api/v1",
    api: "openai-completions",
    apiKey: "$OPENROUTER_API_KEY",
    // No `headers`: pi already attributes OpenRouter requests (HTTP-Referer:
    // https://pi.dev, X-OpenRouter-Title: pi). Setting our own would override
    // those on the wire.
  };
}

export interface PinBuildResult {
  providerName: string;
  providerEntry: ProviderEntryBase & { models: ModelConfig[] };
  modelConfig: ModelConfig;
  settingsPatch: { defaultProvider: string; defaultModel: string; enabledModel: string } | undefined;
}

/** Pure pin construction: given a catalog entry, options, and existing models, the full pin. */
export function buildPin(
  raw: RawModelShape,
  opts: PinOptions,
  existingModels: ModelConfig[],
): PinBuildResult {
  const modelConfig = toModelConfig(raw, opts);
  const providerName = providerNameFor(opts.slug, opts);
  const providerEntry = {
    ...baseProviderConfig(),
    models: [...existingModels.filter((m) => m.id !== opts.modelId), modelConfig],
  };
  return {
    providerName,
    providerEntry,
    modelConfig,
    settingsPatch: opts.isDefault
      ? { defaultProvider: providerName, defaultModel: opts.modelId, enabledModel: `${providerName}/${opts.modelId}` }
      : undefined,
  };
}

// Legacy attribution headers written by earlier versions of this extension.
// Recognized so re-pins stop writing them (and strip them from entries this
// extension created); user-authored headers are left alone.
const LEGACY_REFERER_HEADER = "https://github.com/";
const LEGACY_APP_TITLE = "pi-openrouter-pin";

/**
 * Drop the attribution headers this extension itself wrote in earlier
 * versions (so re-pinning an old provider stops misattributing requests on
 * the wire). User-authored headers are preserved as-is.
 *
 * Returns `undefined` when the result would be an empty object.
 */
export function stripLegacyAttribution(
  headers: Record<string, string> | undefined,
): Record<string, string> | undefined {
  if (!headers) return undefined;
  if (headers["HTTP-Referer"] === LEGACY_REFERER_HEADER && headers["X-Title"] === LEGACY_APP_TITLE) {
    const rest = { ...headers };
    delete rest["HTTP-Referer"];
    delete rest["X-Title"];
    return Object.keys(rest).length > 0 ? rest : undefined;
  }
  return headers;
}
