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
}

export const COMMON_QUANTIZATIONS = ["none", "fp8", "bf16", "fp16", "int4"];

// ---------------------------------------------------------------------------
// Slug helpers
// ---------------------------------------------------------------------------

/** Normalize an OpenRouter provider slug: "NVIDIA" → "nvidia", "Z AI" → "z-ai". */
export function slugify(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
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

/** OpenRouter per-token pricing → pi per-million-token cost, rounded to 2dp. */
export function toCost(pricing: RawModelShape["pricing"] | undefined): ModelConfig["cost"] {
  const p = pricing ?? {};
  const num = (s?: string): number => {
    const v = parseFloat(s ?? "");
    return Number.isFinite(v) ? Math.round(v * 1_000_000 * 100) / 100 : 0;
  };
  return {
    input: num(p.prompt),
    output: num(p.completion),
    cacheRead: num(p.input_cache_read),
    cacheWrite: num(p.input_cache_write),
  };
}

/**
 * The published pricing & limits snapshot of a catalog entry (cost,
 * contextWindow, maxTokens): what a fresh pin would record today. Shared by
 * pin-time config building and the startup refresh so both always agree on
 * the current OpenRouter data.
 */
export function toPricingAndLimits(
  raw: RawModelShape,
): Pick<ModelConfig, "contextWindow" | "maxTokens" | "cost"> {
  return {
    contextWindow: raw.context_length ?? 128_000,
    maxTokens: raw.top_provider?.max_completion_tokens ?? raw.per_request_limits?.max_tokens ?? 16_384,
    cost: toCost(raw.pricing),
  };
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
  const caps = toPricingAndLimits(raw);
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
