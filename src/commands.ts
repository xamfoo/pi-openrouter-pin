/**
 * Core command logic: pin, list, refresh. IO lives here (the edges); the
 * config builders it calls are pure (see config.ts).
 */
import type { ExtensionAPI, ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { fuzzyFilter } from "@earendil-works/pi-tui";
import type { OpenRouterClient, RawModel } from "./api.ts";
import { PROVIDER_PREFIX } from "./api.ts";
import {
  anchorSlug,
  buildPin,
  findEndpoint,
  pricingAndLimitsFromEndpoint,
  providerNameFor,
  slugify,
  stripLegacyAttribution,
  type ModelConfig,
  type OpenRouterRouting,
  type PinOptions,
  type RawEndpointShape,
} from "./config.ts";
import { atomicWriteJson, readJsonFile, type ModelsJson, type ProviderEntry, type SettingsJson } from "./files.ts";

export function formatRouting(r: OpenRouterRouting): string {
  const parts: string[] = [];
  if (r.only && r.only.length > 0) parts.push(`only=${r.only.join(",")}`);
  if (r.order && r.order.length > 0) parts.push(`order=${r.order.join(",")}`);
  if (r.ignore && r.ignore.length > 0) parts.push(`ignore=${r.ignore.join(",")}`);
  if (r.quantizations && r.quantizations.length > 0) parts.push(`quant=${r.quantizations.join(",")}`);
  if (r.data_collection) parts.push(`data_collection=${r.data_collection}`);
  parts.push(`fallbacks=${r.allow_fallbacks ?? false}`);
  return parts.join(" ");
}

export async function performPin(
  modelsPath: string,
  settingsPath: string,
  pi: ExtensionAPI,
  ctx: ExtensionUIContext,
  client: OpenRouterClient,
  resolveApiKey: () => Promise<string | undefined>,
  opts: PinOptions,
): Promise<void> {
  try {
    ctx.notify(`Pinning ${opts.modelId} → ${providerNameFor(opts.slug, opts)}…`, "info");

    const raw = await client.fetchRawModel(opts.modelId);
    if (!raw) {
      ctx.notify(`Model "${opts.modelId}" not found on OpenRouter. Check the id (e.g. z-ai/glm-5.2).`, "error");
      return;
    }

    const apiKey = await resolveApiKey();
    const check = await client.validateEndpoint(opts.modelId, opts.slug, opts.quant, apiKey);
    if (check.status === "error") {
      ctx.notify(`Not pinned: ${check.message}`, "error");
      return;
    }

    // Persist to models.json (pi-native, survives restarts & plugin removal).
    const models = (await readJsonFile<ModelsJson>(modelsPath)) ?? { providers: {} };
    models.providers = models.providers ?? {};
    // The provider name reflects the routing policy (openrouter-<provider> for
    // strict pins, openrouter-<provider>-plus for relaxed ones), so look up and
    // write under the SAME name buildPin computes.
    const providerName = providerNameFor(opts.slug, opts);
    const existingEntry = models.providers[providerName];
    const existingModels = existingEntry && Array.isArray(existingEntry.models) ? existingEntry.models : [];
    const built = buildPin(raw, { ...opts, quant: check.quant, endpoint: check.status === "ok" ? check.endpoint : undefined }, existingModels);
    // Build the entry WITHOUT relying on spread precedence for headers:
    // `...(existingEntry ?? {})` would re-introduce legacy attribution
    // headers that stripLegacyAttribution just removed (the conditional
    // spread of `{}` cannot override an already-spread key).
    const providerEntry: ProviderEntry = {
      ...built.providerEntry,
      ...(existingEntry ?? {}),
      models: built.providerEntry.models,
    };
    const stripped = stripLegacyAttribution(existingEntry?.headers);
    if (stripped) providerEntry.headers = stripped;
    else delete providerEntry.headers;
    models.providers[built.providerName] = providerEntry;
    await atomicWriteJson(modelsPath, models);

    // Optional: make it the default for future sessions.
    if (built.settingsPatch) {
      const settings = (await readJsonFile<SettingsJson>(settingsPath)) ?? {};
      settings.defaultProvider = built.settingsPatch.defaultProvider;
      settings.defaultModel = built.settingsPatch.defaultModel;
      const enabled = Array.isArray(settings.enabledModels) ? settings.enabledModels : [];
      if (!enabled.includes(built.settingsPatch.enabledModel)) enabled.push(built.settingsPatch.enabledModel);
      settings.enabledModels = enabled;
      await atomicWriteJson(settingsPath, settings);
    }

    // Register the FULL model list: registerProvider with `models` replaces
    // all models for the provider, so registering just the new pin would
    // silently drop previously pinned models from the live registry.
    pi.registerProvider(built.providerName, providerEntry);

    const defaultSuffix = built.settingsPatch ? " and set as default" : "";
    ctx.notify(`Pinned ${built.providerName}/${opts.modelId}${defaultSuffix}.`, "info");
    // data_collection is orthogonal to routing; on a strict pin the provider
    // name shows no trace of it, so say so (covers the wizard and CLI alike;
    // it also appears as data_collection=… in /openrouter-pins).
    if (opts.dataCollection && !opts.allowFallbacks && (opts.order?.length ?? 0) === 0 && (opts.ignore?.length ?? 0) === 0) {
      ctx.notify(`Data collection set to "${opts.dataCollection}"; routing stays strict (only=${slugify(opts.slug)}).`, "info");
    }
    // The unvalidated arm always carries a note; the ok arm may. Same handling.
    if (check.note) ctx.notify(`Note: ${check.note}`, "warning");
  } catch (err) {
    ctx.notify(`Pin failed: ${err instanceof Error ? err.message : String(err)}`, "error");
  }
}

export async function listPins(modelsPath: string): Promise<Array<{ provider: string; model: ModelConfig }>> {
  const models = await readJsonFile<ModelsJson>(modelsPath);
  const pins: Array<{ provider: string; model: ModelConfig }> = [];
  for (const [name, provider] of Object.entries(models?.providers ?? {})) {
    if (!name.startsWith(PROVIDER_PREFIX)) continue;
    for (const m of Array.isArray(provider.models) ? provider.models : []) {
      if (m.compat?.openRouterRouting) pins.push({ provider: name, model: m });
    }
  }
  return pins;
}

// ---------------------------------------------------------------------------
// Unpin
// ---------------------------------------------------------------------------

/**
 * Pure: remove a model id from every openrouter-* provider in a models.json
 * snapshot. Providers left with zero models are dropped entirely;
 * non-openrouter providers are never touched. Returns the pruned snapshot
 * and whether anything was removed. Never mutates its input.
 */
export function unpinFromModels(
  models: ModelsJson | null,
  modelId: string,
): { models: ModelsJson; removed: boolean } {
  const providers: Record<string, ProviderEntry> = {};
  let removed = false;
  for (const [name, provider] of Object.entries(models?.providers ?? {})) {
    if (!name.startsWith(PROVIDER_PREFIX)) {
      providers[name] = provider;
      continue;
    }
    const list = Array.isArray(provider.models) ? provider.models : [];
    const next = list.filter((m) => m.id !== modelId);
    if (next.length === list.length) {
      providers[name] = provider;
    } else {
      removed = true;
      // Only clone the entries that actually change; a provider left empty is
      // dropped rather than persisted as `models: []`.
      if (next.length > 0) providers[name] = { ...provider, models: next };
    }
  }
  return { models: { ...models, providers }, removed };
}

/** Result of /openrouter-unpin: three explicit arms, each with its own notice. */
export type UnpinOutcome =
  | { status: "removed" }
  | { status: "no-providers" } // models.json missing, or has no providers key
  | { status: "not-found" }; // providers exist, but the model is not pinned

/**
 * Edge: read models.json, unpin the model from every openrouter-* provider,
 * and write the pruned file only when something was actually removed. A
 * missing file or a model that is not pinned never creates or rewrites
 * models.json.
 */
export async function performUnpin(modelsPath: string, modelId: string): Promise<UnpinOutcome> {
  const models = await readJsonFile<ModelsJson>(modelsPath);
  if (!models?.providers) return { status: "no-providers" };
  const { models: pruned, removed } = unpinFromModels(models, modelId);
  if (!removed) return { status: "not-found" };
  await atomicWriteJson(modelsPath, pruned);
  return { status: "removed" };
}

// ---------------------------------------------------------------------------
// Startup snapshot refresh
// ---------------------------------------------------------------------------

export interface RefreshResult {
  refreshed: number;
  failed: string[];
  /** Per-model before → after diff for every pricing/limits change applied. */
  diff: PricingLimitsDiff[];
}

/** One model's pricing & limits before → after, for reload-time reporting. */
export interface PricingLimitsDiff {
  provider: string;
  modelId: string;
  before: { cost: ModelConfig["cost"]; contextWindow: number; maxTokens: number };
  after: { cost: ModelConfig["cost"]; contextWindow: number; maxTokens: number };
}

export interface PricingLimitsPatch {
  provider: string;
  modelId: string;
  cost: ModelConfig["cost"];
  contextWindow: number;
  maxTokens: number;
}

/**
 * Pure: the pinned models eligible for refresh — openrouter-* providers whose
 * entries carry openRouterRouting compat.
 */
export function collectRefreshTargets(
  snapshot: ModelsJson | null,
): Array<{ provider: string; model: ModelConfig }> {
  const targets: Array<{ provider: string; model: ModelConfig }> = [];
  for (const [name, provider] of Object.entries(snapshot?.providers ?? {})) {
    if (!name.startsWith(PROVIDER_PREFIX)) continue;
    for (const m of Array.isArray(provider.models) ? provider.models : []) {
      const r = m.compat?.openRouterRouting;
      // Only include models that have actual routing configuration (not just
      // an empty openRouterRouting object). The presence of any routing field
      // (only, order, ignore, quantizations) indicates a real pin.
      if (r && (r.only?.length || r.order?.length || r.ignore?.length || r.quantizations?.length)) {
        targets.push({ provider: name, model: m });
      }
    }
  }
  return targets;
}

/**
 * Pure: the pricing/limits patch for one pinned model, computed from its
 * validated endpoint (provider-specific truth) rather than the catalog
 * aggregate. The anchor slug (and stored quant, if any) select the endpoint;
 * fields the endpoint omits fall back to the stored snapshot so they stay
 * stable instead of being zeroed. Returns null when there is no matching
 * endpoint (provider dropped, or quant no longer served) — the stored value
 * is left untouched rather than clobbered.
 */
export function computeEndpointPatch(
  target: { provider: string; model: ModelConfig },
  endpoints: RawEndpointShape[],
): PricingLimitsPatch | null {
  const routing = target.model.compat?.openRouterRouting;
  const slug = anchorSlug(routing);
  if (!slug) return null;
  const quant = routing!.quantizations?.[0];
  const endpoint = findEndpoint(endpoints, slug, quant);
  if (!endpoint || !endpoint.pricing) return null;
  const fresh = pricingAndLimitsFromEndpoint(endpoint, {
    contextWindow: target.model.contextWindow,
    maxTokens: target.model.maxTokens,
    cost: target.model.cost,
  });
  const changed =
    JSON.stringify(fresh.cost) !== JSON.stringify(target.model.cost) ||
    fresh.contextWindow !== target.model.contextWindow ||
    fresh.maxTokens !== target.model.maxTokens;
  return changed
    ? { provider: target.provider, modelId: target.model.id, cost: fresh.cost, contextWindow: fresh.contextWindow, maxTokens: fresh.maxTokens }
    : null;
}

/**
 * Pure: apply patches to a freshly-read models.json. Providers or models that
 * vanished between the snapshot read and this read (a concurrent unpin) are
 * skipped, never resurrected. Returns the number of models actually patched
 * and a before → after diff for each one (captured from the freshly-read
 * model, so it reflects what the user actually had on disk, not the stale
 * snapshot read).
 */
export function applyPricingPatches(
  current: ModelsJson | null,
  patches: PricingLimitsPatch[],
): { applied: number; diff: PricingLimitsDiff[] } {
  let applied = 0;
  const diff: PricingLimitsDiff[] = [];
  for (const patch of patches) {
    const provider = current?.providers?.[patch.provider];
    if (!provider || !Array.isArray(provider.models)) continue;
    const model = provider.models.find((m) => m.id === patch.modelId);
    if (!model) continue;
    const before = { cost: model.cost, contextWindow: model.contextWindow, maxTokens: model.maxTokens };
    model.cost = patch.cost;
    model.contextWindow = patch.contextWindow;
    model.maxTokens = patch.maxTokens;
    applied++;
    diff.push({ provider: patch.provider, modelId: patch.modelId, before, after: { cost: patch.cost, contextWindow: patch.contextWindow, maxTokens: patch.maxTokens } });
  }
  return { applied, diff };
}

/**
 * Refresh the stored pricing & limits snapshot (cost, contextWindow,
 * maxTokens) of every pinned model from its validated OpenRouter endpoint.
 * Only those three fields are touched — routing, name, input types, and all
 * other fields are preserved.
 *
 * Per-provider pricing & limits come from /models/<id>/endpoints, not the
 * catalog aggregate: the catalog's model-level `pricing`/`top_provider`
 * summarize whichever provider OpenRouter ranks first, which is frequently a
 * different provider than the pin's anchor (e.g. z-ai/glm-5.2's catalog
 * pricing matches SiliconFlow, while a novita pin should record novita's
 * prices). Using the endpoint keeps a provider-specific pin honest.
 *
 * The refresh computes a patch and re-reads models.json at write time, so a
 * pin/unpin that landed between the first read and the write is preserved
 * rather than clobbered by a stale in-memory object. Without an API key the
 * endpoints API is unreachable, so refresh is skipped (returning nothing)
 * rather than overwriting correct per-provider prices with the catalog
 * aggregate; a model whose endpoints fetch fails is reported as failed, never
 * thrown. The three phases are pure and exported (collectRefreshTargets →
 * computeEndpointPatch → applyPricingPatches) so the concurrency guarantee is
 * testable in isolation as well as end-to-end.
 */
export async function refreshPinnedModels(
  modelsPath: string,
  client: OpenRouterClient,
  resolveApiKey: () => Promise<string | undefined>,
): Promise<RefreshResult> {
  const snapshot = await readJsonFile<ModelsJson>(modelsPath);
  const targets = collectRefreshTargets(snapshot);
  if (targets.length === 0) return { refreshed: 0, failed: [], diff: [] };

  // Per-provider truth needs the endpoints API; without a key it is
  // unreachable, so skip rather than fall back to the catalog aggregate
  // (which would clobber correct per-provider prices with a different
  // provider's).
  const apiKey = await resolveApiKey();
  if (!apiKey) return { refreshed: 0, failed: [], diff: [] };

  const patches: PricingLimitsPatch[] = [];
  const failed: string[] = [];
  for (const target of targets) {
    const result = await client.fetchModelEndpoints(target.model.id, apiKey);
    // A message means the endpoints list is unavailable (404 → model gone, or
    // transport error): report the model as failed. An empty list without a
    // message is treated the same — there is no endpoint to price from.
    if (result.message || result.endpoints.length === 0) {
      failed.push(target.model.id);
      continue;
    }
    const patch = computeEndpointPatch(target, result.endpoints);
    if (patch) patches.push(patch);
  }
  if (patches.length === 0) return { refreshed: 0, failed, diff: [] };

  // Apply to a fresh read so concurrent writes are not lost.
  const current = (await readJsonFile<ModelsJson>(modelsPath)) ?? { providers: {} };
  const { applied, diff } = applyPricingPatches(current, patches);
  if (applied > 0) await atomicWriteJson(modelsPath, current);
  return { refreshed: applied, failed, diff };
}

// ---------------------------------------------------------------------------
// Reload-time diff rendering
// ---------------------------------------------------------------------------

const COST_FIELD_LABELS: Array<{ key: keyof ModelConfig["cost"]; label: string }> = [
  { key: "input", label: "cost.input" },
  { key: "output", label: "cost.output" },
  { key: "cacheRead", label: "cost.cacheRead" },
  { key: "cacheWrite", label: "cost.cacheWrite" },
];

/** Cost is stored as dollars-per-million-tokens (see toCost); render as $/M. */
function formatCost(v: number): string {
  return `$${v.toFixed(2)}/M`;
}

function formatTokens(v: number): string {
  return v.toLocaleString("en-US");
}

/**
 * Render the refresh diff as aligned, model-grouped lines (no header). Only
 * fields that actually changed are emitted, so an unchanged model contributes
 * nothing and an unchanged field within a changed model is hidden. Returns an
 * empty string when there is nothing to show.
 */
export function formatRefreshDiff(diff: PricingLimitsDiff[]): string {
  if (diff.length === 0) return "";
  const lines: string[] = [];
  for (const d of diff) {
    const rows: Array<[string, string, string]> = []; // [label, before, after]
    for (const { key, label } of COST_FIELD_LABELS) {
      if (d.before.cost[key] !== d.after.cost[key]) {
        rows.push([label, formatCost(d.before.cost[key]), formatCost(d.after.cost[key])]);
      }
    }
    if (d.before.contextWindow !== d.after.contextWindow) {
      rows.push(["contextWindow", formatTokens(d.before.contextWindow), formatTokens(d.after.contextWindow)]);
    }
    if (d.before.maxTokens !== d.after.maxTokens) {
      rows.push(["maxTokens", formatTokens(d.before.maxTokens), formatTokens(d.after.maxTokens)]);
    }
    if (rows.length === 0) continue;
    lines.push(`  ${d.modelId} (${d.provider}):`);
    const width = Math.max(...rows.map((r) => r[0].length));
    for (const [label, before, after] of rows) {
      lines.push(`    ${label.padEnd(width)}  ${before}  →  ${after}`);
    }
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Model ranking for the wizard and TAB completion (exact → prefix → fuzzy)
// ---------------------------------------------------------------------------

/**
 * Rank models for a query: exact id match first, then id-prefix matches,
 * then pi-tui's fuzzyFilter over "id name". fuzzyFilter is a subsequence
 * match with word-boundary/consecutive bonuses and splits the query on
 * whitespace and "/", so "glm52" finds "z-ai/glm-5.2" and "glm5 novita"
 * requires both tokens to match. Used by the wizard search and model-id
 * TAB completion alike.
 */
export function rankModelsForQuery(models: RawModel[], query: string): RawModel[] {
  const trimmed = query.trim();
  if (!trimmed) return models;
  const lower = trimmed.toLowerCase();
  const exact: RawModel[] = [];
  const prefix: RawModel[] = [];
  const rest: RawModel[] = [];
  for (const m of models) {
    const id = m.id.toLowerCase();
    if (id === lower) exact.push(m);
    else if (id.startsWith(lower)) prefix.push(m);
    else rest.push(m);
  }
  if (rest.length === 0) return [...exact, ...prefix];
  const fuzzy = fuzzyFilter(rest, trimmed, (m) => `${m.id} ${m.name ?? ""}`);
  return [...exact, ...prefix, ...fuzzy];
}
