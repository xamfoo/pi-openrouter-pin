/**
 * Core command logic: pin, list, refresh. IO lives here (the edges); the
 * config builders it calls are pure (see config.ts).
 */
import type { ExtensionAPI, ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { fuzzyFilter } from "@earendil-works/pi-tui";
import type { OpenRouterClient, RawModel } from "./api.ts";
import { PROVIDER_PREFIX } from "./api.ts";
import {
  buildPin,
  providerNameFor,
  slugify,
  stripLegacyAttribution,
  toPricingAndLimits,
  type ModelConfig,
  type OpenRouterRouting,
  type PinOptions,
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
    const built = buildPin(raw, { ...opts, quant: check.quant }, existingModels);
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
      if (m.compat?.openRouterRouting) targets.push({ provider: name, model: m });
    }
  }
  return targets;
}

/**
 * Pure: diff snapshot targets against the current catalog. Returns the
 * pricing/limits patches for changed models, and the ids that are no longer
 * in the catalog (collected and reported, never thrown).
 */
export function computePricingPatches(
  targets: Array<{ provider: string; model: ModelConfig }>,
  catalog: RawModel[],
): { patches: PricingLimitsPatch[]; failed: string[] } {
  const byId = new Map(catalog.map((m) => [m.id, m]));
  const patches: PricingLimitsPatch[] = [];
  const failed: string[] = [];
  for (const { provider, model } of targets) {
    const raw = byId.get(model.id);
    if (!raw) {
      failed.push(model.id);
      continue;
    }
    const fresh = toPricingAndLimits(raw);
    const changed =
      JSON.stringify(fresh.cost) !== JSON.stringify(model.cost) ||
      fresh.contextWindow !== model.contextWindow ||
      fresh.maxTokens !== model.maxTokens;
    if (changed) {
      patches.push({ provider, modelId: model.id, cost: fresh.cost, contextWindow: fresh.contextWindow, maxTokens: fresh.maxTokens });
    }
  }
  return { patches, failed };
}

/**
 * Pure: apply patches to a freshly-read models.json. Providers or models that
 * vanished between the snapshot read and this read (a concurrent unpin) are
 * skipped, never resurrected. Returns the number of models actually patched.
 */
export function applyPricingPatches(current: ModelsJson | null, patches: PricingLimitsPatch[]): number {
  let applied = 0;
  for (const patch of patches) {
    const provider = current?.providers?.[patch.provider];
    if (!provider || !Array.isArray(provider.models)) continue;
    const model = provider.models.find((m) => m.id === patch.modelId);
    if (!model) continue;
    model.cost = patch.cost;
    model.contextWindow = patch.contextWindow;
    model.maxTokens = patch.maxTokens;
    applied++;
  }
  return applied;
}

/**
 * Refresh the stored pricing & limits snapshot (cost, contextWindow,
 * maxTokens) of every pinned model from OpenRouter's current catalog. Only
 * those three fields are touched — routing, name, input types, and all other
 * fields are preserved.
 *
 * The refresh computes a patch and re-reads models.json at write time, so a
 * pin/unpin that landed between the first read and the write is preserved
 * rather than clobbered by a stale in-memory object. Offline or catalog
 * errors are collected and reported, never thrown.
 *
 * The three phases are pure and exported (collectRefreshTargets,
 * computePricingPatches, applyPricingPatches) so the concurrency guarantee
 * is testable in isolation as well as end-to-end.
 */
export async function refreshPinnedModels(
  modelsPath: string,
  client: OpenRouterClient,
): Promise<RefreshResult> {
  const snapshot = await readJsonFile<ModelsJson>(modelsPath);
  const targets = collectRefreshTargets(snapshot);
  if (targets.length === 0) return { refreshed: 0, failed: [] };

  const catalog = await client.fetchCatalog();
  const { patches, failed } = computePricingPatches(targets, catalog);
  if (patches.length === 0) return { refreshed: 0, failed };

  // Apply to a fresh read so concurrent writes are not lost.
  const current = (await readJsonFile<ModelsJson>(modelsPath)) ?? { providers: {} };
  const applied = applyPricingPatches(current, patches);
  if (applied > 0) await atomicWriteJson(modelsPath, current);
  return { refreshed: applied, failed };
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
