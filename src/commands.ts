/**
 * Core command logic: pin, list, refresh. IO lives here (the edges); the
 * config builders it calls are pure (see config.ts).
 */
import type { ExtensionAPI, ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { fuzzyFilter } from "@earendil-works/pi-tui";
import type { OpenRouterClient, RawModel } from "./api.ts";
import { PROVIDER_PREFIX } from "./config.ts";
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Loader-only stub message from `discoverAndLoadExtensions()` when pi.setModel is an unbound reject.
 *  Exact match intentional — brittle if pi rewords, but correct for known versions: pi >=0.86
 *  ships `setModel`, loader-only sessions use a stub that rejects with this string
 *  (dist/core/extensions/loader.js in pi monorepo). */
const LOADER_UNBOUND_STUB_MSG = "Extension runtime not initialized";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Result of planUnpin: the pure plan values used by the applier. */
export interface UnpinPlan {
  /** Providers left with zero models after removal. */
  emptied: string[];
  /** Providers with surviving siblings, needing re-registration with the pruned list. */
  repruned: string[];
  /** Settings changes to apply (null when nothing changed). */
  settingsPatch: SettingsPatch | null;
  /** Model ids removed from models.json. */
  removed: string[];
  /** Whether this invocation created the settings file (for rollback). */
  createdFile: boolean;
}

/** Settings changes computed by planUnpin or planPin. */
export interface SettingsPatch {
  defaultProvider?: string;
  defaultModel?: string;
  enabledModel?: string;
  /** The prior values for rollback. */
  _before: SettingsBefore;
}

/** Settings values before a change (for rollback). */
export interface SettingsBefore {
  defaultProvider?: string;
  defaultModel?: string;
  enabledModels?: string[];
}

/** Settings before/after comparison result. */
export interface SettingsDiff {
  /** Whether the default was cleared (exact match). */
  defaultCleared: boolean;
  /** Whether enabledModels was pruned. */
  enabledPruned: boolean;
  /** Whether settings.json was actually rewritten. */
  wrote: boolean;
}

/** Capability probe result driving live vs fileOnly behavior. */
export interface CapabilityProbe {
  canUnregister: boolean;
  canSetModel: boolean;
  live: boolean;
}

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

/**
 * Pure: plan a pin operation.
 *
 * Computes the settings patch and whether the settings file
 * would be created or rewritten. Never mutates its inputs.
 */
export function planPin(
  models: ModelsJson | null,
  settings: SettingsJson | null,
  modelId: string,
  settingsPatch: { defaultProvider?: string; defaultModel?: string; enabledModel?: string } | undefined,
): {
  settingsPatch: SettingsPatch | null;
  createdFile: boolean;
  duplicateEnabledModel: boolean;
} {
  const createdFile = settings === null;
  const currentSettings = settings ?? {};
  const existingEnabled = Array.isArray(currentSettings.enabledModels) ? currentSettings.enabledModels : [];

  if (!settingsPatch) {
    return { settingsPatch: null, createdFile, duplicateEnabledModel: false };
  }

  // Check for duplicate enabledModel.
  const duplicateEnabledModel = existingEnabled.includes(settingsPatch.enabledModel!);

  const patch: SettingsPatch = {
    _before: {
      defaultProvider: currentSettings.defaultProvider,
      defaultModel: currentSettings.defaultModel,
      enabledModels: [...existingEnabled],
    },
  };

  if (settingsPatch.defaultProvider) patch.defaultProvider = settingsPatch.defaultProvider;
  if (settingsPatch.defaultModel) patch.defaultModel = settingsPatch.defaultModel;
  if (settingsPatch.enabledModel && !duplicateEnabledModel) {
    patch.enabledModel = settingsPatch.enabledModel;
  }

  return { settingsPatch: patch, createdFile, duplicateEnabledModel };
}

/**
 * Options-object signature for performPin with live default switch.
 */
export interface PerformPinOptions {
  modelsPath: string;
  settingsPath: string;
  pi: ExtensionAPI;
  ctx: ExtensionUIContext;
  client: OpenRouterClient;
  resolveApiKey: () => Promise<string | undefined>;
  opts: PinOptions;
  /** For live `setModel` switching (refresh + find). When absent, falls back to fileOnly. */
  modelRegistry?: { refresh?: () => unknown; find?: (provider: string, modelId: string) => unknown };
  /** Full ExtensionCommandContext for `ctx.setModel` fallback probing and scope check (`scopedModels`). */
  extCtx?: unknown;
}

/**
 * Reworked performPin with options-object signature, live default switch,
 * settings rollback on failure, and idempotency (duplicate enabledModels never duplicated).
 *
 * On failure of the live switch, restores settings.json to its prior content
 * (or removes the file only if this invocation created it). The pin itself is kept.
 */
export async function performPin(options: PerformPinOptions): Promise<void> {
  const { modelsPath, settingsPath, pi, ctx, client, resolveApiKey, opts, modelRegistry, extCtx } = options;
  const cap = probe(pi, extCtx ?? modelRegistry ?? ctx);
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
    const providerName = providerNameFor(opts.slug, opts);
    const existingEntry = models.providers[providerName];
    const existingModels = existingEntry && Array.isArray(existingEntry.models) ? existingEntry.models : [];
    const built = buildPin(raw, { ...opts, quant: check.quant, endpoint: check.status === "ok" ? check.endpoint : undefined }, existingModels);
    // Intentional inversion: existingEntry after built preserves prior baseUrl/api
    // (and placeholder retention) while built's models always win.
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

    // Startup scope: any pin should join `enabledModels` when scoping is
    // active (`scopedModels.length>0`), so `/scoped-models` or `/reload` can
    // pick it up. No live poke; host owns `setScopedModels`.
    const scoped = (extCtx as unknown as { scopedModels?: readonly unknown[] } | undefined)?.scopedModels;
    const isScopeActive = Array.isArray(scoped) && scoped.length > 0;
    const scopeEnabledModel = `${built.providerName}/${opts.modelId}`;

    // Optional: make it the default for future sessions, or join active scope.
    let settingsBefore: SettingsJson | null = null;
    let pinPlan: ReturnType<typeof planPin> | null = null;
    let settingsWrote = false;
    let scopeJoined = false;
    if (built.settingsPatch) {
      settingsBefore = (await readJsonFile<SettingsJson>(settingsPath)) ?? null;
      pinPlan = planPin(null, settingsBefore, opts.modelId, built.settingsPatch);
      if (pinPlan.settingsPatch) {
        // Single read, single write; avoid TOCTOU double-read.
        const base = settingsBefore ? { ...settingsBefore } as SettingsJson : ({} as SettingsJson);
        base.defaultProvider = built.settingsPatch.defaultProvider;
        base.defaultModel = built.settingsPatch.defaultModel;
        if (!pinPlan.duplicateEnabledModel) {
          const enabled = Array.isArray(base.enabledModels) ? [...base.enabledModels] : [];
          if (!enabled.includes(built.settingsPatch.enabledModel)) enabled.push(built.settingsPatch.enabledModel);
          base.enabledModels = enabled;
        }
        await atomicWriteJson(settingsPath, base);
        settingsWrote = true;
        if (isScopeActive && !pinPlan.duplicateEnabledModel) scopeJoined = true;
      }
    } else if (isScopeActive) {
      // Pin without --default should still join active startup scope via file.
      settingsBefore = (await readJsonFile<SettingsJson>(settingsPath)) ?? null;
      const base = settingsBefore ? { ...settingsBefore } as SettingsJson : ({} as SettingsJson);
      const enabled = Array.isArray(base.enabledModels) ? [...base.enabledModels] : [];
      if (!enabled.includes(scopeEnabledModel)) {
        enabled.push(scopeEnabledModel);
        base.enabledModels = enabled;
        const beforeJson = JSON.stringify(settingsBefore ?? {});
        const afterJson = JSON.stringify(base);
        if (beforeJson !== afterJson) {
          await atomicWriteJson(settingsPath, base);
          settingsWrote = true;
        }
        scopeJoined = true;
      }
      pinPlan = null;
    }

    // Register the FULL model list.
    // Inject the resolved key live (never persisted to models.json).
    pi.registerProvider(built.providerName, apiKey ? { ...providerEntry, apiKey } : providerEntry);

    // Attempt live default switch (only if default was requested).
    let liveSwitchFailed = false;
    let liveSwitchSkippedFileOnly = false;
    if (built.settingsPatch) {
      if (!cap.canSetModel) {
        liveSwitchSkippedFileOnly = true;
      } else {
        try {
          // await-agnostic refresh: works for sync void (0.84) and Promise (0.86).
          const registry: any = modelRegistry;
          if (registry?.refresh) {
            const r = registry.refresh();
            if (r instanceof Promise) await r;
          }
          const found = registry?.find ? registry.find(built.providerName, opts.modelId) : undefined;
          // Determine setModel target: prefer pi, fall back to ctx.
          const piAny = pi as unknown as Record<string, unknown>;
          const ctxAny = (extCtx ?? ctx) as unknown as Record<string, unknown>;
          const setModelFn =
            typeof piAny.setModel === "function"
              ? (piAny.setModel as (m: unknown) => unknown)
              : typeof ctxAny.setModel === "function"
                ? (ctxAny.setModel as (m: unknown) => unknown)
                : undefined;
          if (!setModelFn) {
            liveSwitchFailed = true;
          } else {
            // Old registries may lack `find`; fall back to minimal {provider,id}
            // shape rather than treating it as a live failure that triggers rollback.
            const modelArg: unknown = found ?? buildModelObj(built.providerName, opts.modelId);
            const result: unknown = setModelFn.call((piAny.setModel ? pi : extCtx ?? ctx) as unknown, modelArg as never);
            const awaited = result instanceof Promise ? await result : result;
            if (awaited === false) {
              // pi's setModel returns false when auth is unconfigured – the only
              // falsy value that signals failure. `undefined`/`void` (old pi sync
              // success) and truthy values are success.
              liveSwitchFailed = true;
            }
          }
        } catch (err) {
          // loader-only session: unbound stub rejects before runtime init;
          // degrade to fileOnly (persist settings, no rollback).
          const msg = err instanceof Error ? err.message : String(err);
          if (msg === LOADER_UNBOUND_STUB_MSG) {
            liveSwitchSkippedFileOnly = true;
          } else {
            liveSwitchFailed = true;
          }
        }
      }
    }

    if (liveSwitchFailed && built.settingsPatch) {
      // Rollback settings: restore settingsBefore value (keep pin).
      if (settingsBefore) {
        await atomicWriteJson(settingsPath, settingsBefore);
      } else if (pinPlan?.createdFile) {
        try {
          const { rmSync } = await import("node:fs");
          rmSync(settingsPath);
        } catch {
          // Best-effort removal.
        }
      } else if (settingsWrote) {
        // We wrote a new file but had no prior snapshot and pinPlan says not created?
        // Best-effort: remove if we created it and now rolled back to nothing.
        try {
          const cur = await readJsonFile<SettingsJson>(settingsPath);
          if (!cur || Object.keys(cur).length === 0) {
            const { rmSync } = await import("node:fs");
            rmSync(settingsPath);
          } else if (settingsBefore) {
            await atomicWriteJson(settingsPath, settingsBefore);
          }
        } catch {}
      }
      // Re-apply startup scope join so the kept pin remains in enabledModels
      // even though the default was rolled back. Documented as intentional:
      // default rollback restores prior defaults but keeps scope membership.
      if (isScopeActive) {
        try {
          const cur = (await readJsonFile<SettingsJson>(settingsPath)) ?? {};
          const enabled = Array.isArray(cur.enabledModels) ? [...cur.enabledModels] : [];
          if (!enabled.includes(scopeEnabledModel)) {
            enabled.push(scopeEnabledModel);
            (cur as SettingsJson).enabledModels = enabled;
            await atomicWriteJson(settingsPath, cur);
          }
        } catch {}
      }
      ctx.notify(`Pin set ${built.providerName}/${opts.modelId} but live switch failed — default rolled back.`, "error");
      ctx.notify(`Pinned ${built.providerName}/${opts.modelId} (live switch rolled back).`, "info");
      if (isScopeActive) {
        ctx.notify(`Pinned ${built.providerName}/${opts.modelId} — enable it under /scoped-models (or /reload) to join live scope.`, "info");
      }
      if (check.note) ctx.notify(`Note: ${check.note}`, "warning");
      return;
    }

    if (scopeJoined) {
      const msg = built.settingsPatch
        ? `Pinned ${built.providerName}/${opts.modelId} and set as default — enable it under /scoped-models (or /reload) to join live scope.`
        : `Pinned ${built.providerName}/${opts.modelId} — enable it under /scoped-models (or /reload) to join live scope.`;
      ctx.notify(msg, "info");
    } else if (liveSwitchSkippedFileOnly && built.settingsPatch) {
      ctx.notify(`Pinned ${built.providerName}/${opts.modelId} and set as default (applies on /reload or next session).`, "info");
    } else {
      const defaultSuffix = built.settingsPatch ? " and set as default" : "";
      ctx.notify(`Pinned ${built.providerName}/${opts.modelId}${defaultSuffix}.`, "info");
    }
    if (opts.dataCollection && !opts.allowFallbacks && (opts.order?.length ?? 0) === 0 && (opts.ignore?.length ?? 0) === 0) {
      ctx.notify(`Data collection set to "${opts.dataCollection}"; routing stays strict (only=${slugify(opts.slug)}).`, "info");
    }
    if (check.note) ctx.notify(`Note: ${check.note}`, "warning");
  } catch (err) {
    ctx.notify(`Pin failed: ${err instanceof Error ? err.message : String(err)}`, "error");
  }
}

/**
 * Build a minimal model object for setModel fallback when `registry.find`
 * is unavailable (old pi). New pi `setModel` expects a `Model` from `find`,
 * but old pi accepted `{provider, id}`.
 */
function buildModelObj(provider: string | undefined, modelId: string | undefined): unknown {
  return { provider: provider ?? "", id: modelId ?? "" };
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
 * non-openrouter providers are never touched. Accepts an optional pre-built
 * `knownBareIds` set — when supplied (from a caller that already scanned the
 * same snapshot), avoids redundant iteration.
 * Returns the pruned snapshot and whether anything was removed.
 * Never mutates its input.
 */
export function unpinFromModels(
  models: ModelsJson | null,
  modelId: string,
  knownBareIds?: ReadonlySet<string>,
): { models: ModelsJson; removed: boolean } {
  // Use caller-supplied set or build one from scratch via the shared helper.
  const known = knownBareIds ?? collectKnownBareIds(models?.providers);
  const normalized = normalizeUnpinArg(modelId, known);
  const providers: Record<string, ProviderEntry> = {};
  let removed = false;
  for (const [name, provider] of Object.entries(models?.providers ?? {})) {
    if (!name.startsWith(PROVIDER_PREFIX)) {
      providers[name] = provider;
      continue;
    }
    const list = Array.isArray(provider.models) ? provider.models : [];
    const next = list.filter((m) => m.id !== normalized);
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

// ---------------------------------------------------------------------------
// Pure plan values
// ---------------------------------------------------------------------------

/**
 * Pure: plan the unpin operation without side effects.
 *
 * Accepts an optional pre-built `knownBareIds` set — when supplied (from a
 * caller that already scanned the same snapshot), avoids redundant iteration.
 */
export function planUnpin(
  snapshot: ModelsJson | null,
  settings: SettingsJson | null,
  modelId: string,
  knownBareIds?: ReadonlySet<string>,
): UnpinPlan {
  // Use caller-supplied set or build one from scratch via the shared helper.
  const known = knownBareIds ?? collectKnownBareIds(snapshot?.providers);
  const normalized = normalizeUnpinArg(modelId, known);
  const emptied: string[] = [];
  const repruned: string[] = [];
  const removed: string[] = [];

  const providers: Record<string, ProviderEntry> = {};
  for (const [name, provider] of Object.entries(snapshot?.providers ?? {})) {
    if (!name.startsWith(PROVIDER_PREFIX)) {
      providers[name] = provider;
      continue;
    }
    const list = Array.isArray(provider.models) ? provider.models : [];
    const next = list.filter((m) => m.id !== normalized);
    if (next.length === list.length) {
      providers[name] = provider;
    } else {
      removed.push(normalized);
      if (next.length === 0) {
        emptied.push(name);
      } else {
        repruned.push(name);
        providers[name] = { ...provider, models: next };
      }
    }
  }

  // Compute settings changes: prune enabledModels, clear default if needed.
  const settingsPatch = computeSettingsPrune(settings, normalized, emptied);

  return {
    emptied,
    repruned,
    settingsPatch,
    removed,
    createdFile: settings === null,
  };
}

/**
 * Pure: compute the settings prune for an unpin.
 *
 * - Removes the model from enabledModels (matches bare `modelId` or any
 *   `provider/modelId` entry, because settings stores `provider/model`).
 * - If the removed model was the configured default, clears defaultProvider/defaultModel.
 * - Returns null when nothing changed.
 */
export function computeSettingsPrune(
  settings: SettingsJson | null,
  modelId: string,
  _emptiedProviders: string[] = [],
): SettingsPatch | null {
  if (!settings) return null;

  const enabled = Array.isArray(settings.enabledModels) ? settings.enabledModels : [];
  const priorDefaultProvider = settings.defaultProvider;
  const priorDefaultModel = settings.defaultModel;

  // Prune enabledModels entries referencing the removed model.
  // Real disk shape is `provider/modelId` (e.g. `openrouter-novita/z-ai/glm-5.2`),
  // so match bare id or suffix `/${modelId}`.
  const suffix = `/${modelId}`;
  const prunedEnabled = enabled.filter((e) => e !== modelId && !e.endsWith(suffix));
  const enabledPruned = prunedEnabled.length !== enabled.length;

  // Check if the removed model was the configured default.
  const wasDefault = priorDefaultModel === modelId;
  const defaultCleared = wasDefault;

  if (!enabledPruned && !defaultCleared) return null;

  const patch: SettingsPatch = {
    _before: {
      defaultProvider: priorDefaultProvider,
      defaultModel: priorDefaultModel,
      enabledModels: [...enabled],
    },
  };

  if (defaultCleared) {
    // Explicitly present as `undefined` so the applier can distinguish
    // "clear" from "leave untouched" via `'key' in patch`.
    patch.defaultProvider = undefined;
    patch.defaultModel = undefined;
  }
  if (enabledPruned) {
    patch.enabledModel = prunedEnabled.join(",");
  }

  return patch;
}

/**
 * Pure: compute the before/after settings diff for a settings write.
 *
 * Returns a SettingsDiff indicating whether the default was cleared,
 * enabledModels was pruned, and whether the settings.json was rewritten.
 */
export function computeSettingsDiff(
  before: SettingsJson | null,
  after: SettingsJson | null,
): SettingsDiff {
  const beforeDefaultProvider = before?.defaultProvider;
  const afterDefaultProvider = after?.defaultProvider;
  const beforeDefaultModel = before?.defaultModel;
  const afterDefaultModel = after?.defaultModel;
  const beforeEnabled = Array.isArray(before?.enabledModels) ? before.enabledModels : [];
  const afterEnabled = Array.isArray(after?.enabledModels) ? after.enabledModels : [];

  const defaultCleared =
    beforeDefaultProvider !== undefined && afterDefaultProvider === undefined &&
    beforeDefaultModel !== undefined && afterDefaultModel === undefined;
  const enabledPruned =
    beforeEnabled.length > afterEnabled.length &&
    afterEnabled.every((e) => beforeEnabled.includes(e));
  const wrote = JSON.stringify(before) !== JSON.stringify(after);

  return { defaultCleared, enabledPruned, wrote };
}

// ---------------------------------------------------------------------------
// Unpin outcome and applier
// ---------------------------------------------------------------------------

/** Result of /openrouter-unpin: three explicit arms, each with its own notice. */
export type UnpinOutcome =
  | {
      /** Action succeeded — provider dropped or pruned. */
      status: "removed";
      // Internal: handler uses for notice echo; not advertised as a stable contract.
      resolvedModelId: string;
    }
  | { status: "no-providers" } // models.json missing, or has no providers key
  | {
      status: "not-found";
      inputModelId?: string; // Internal: echo original input in not-found notice.
    };

/**
 * Options-object signature for performUnpin with live capability.
 */
export interface PerformUnpinOptions {
  modelsPath: string;
  settingsPath: string;
  pi: ExtensionAPI;
  modelId: string;
  /** For live reprune: inject resolved key into re-registered providers. */
  resolveApiKey?: () => Promise<string | undefined>;
  /** Full command context for probe fallback (e.g. ctx.setModel). */
  ctx?: unknown;
}

/**
 * Edge: read models.json, unpin the model from every openrouter-* provider,
 * and write the pruned file only when something was actually removed. A
 * missing file or a model that is not pinned never creates or rewrites
 * models.json. Supports both the legacy `(modelsPath, modelId)` signature
 * and the options-object live signature.
 */
export async function performUnpin(
  modelsPathOrOptions: string | PerformUnpinOptions,
  modelId?: string,
): Promise<UnpinOutcome> {
  // Legacy positional signature: performUnpin(modelsPath, modelId)
  if (typeof modelsPathOrOptions === "string") {
    const modelsPath = modelsPathOrOptions;
    const mid = modelId!;
    const models = await readJsonFile<ModelsJson>(modelsPath);
    if (!models?.providers) return { status: "no-providers" };
    // Single scan from one snapshot: collect bare ids, normalize, apply.
    const known = collectKnownBareIds(models.providers);
    const normalized = normalizeUnpinArg(mid, known);
    // Thread the pre-built set so unpinFromModels skips its own scan.
    const { models: pruned, removed } = unpinFromModels(models, mid, known);
    if (!removed) return { status: "not-found", inputModelId: mid };
    await atomicWriteJson(modelsPath, pruned);
    return { status: "removed", resolvedModelId: normalized };
  }

  // Options-object live signature
  const { modelsPath, settingsPath, pi, modelId: mid, resolveApiKey, ctx } = modelsPathOrOptions;
  const cap = probe(pi, ctx);
  const snapshot = await readJsonFile<ModelsJson>(modelsPath);
  if (!snapshot?.providers) return { status: "no-providers" };
  // Single scan: collect bare ids once and thread through both pure helpers.
  const known = collectKnownBareIds(snapshot.providers);
  const normalized = normalizeUnpinArg(mid, known);
  const settings = await readJsonFile<SettingsJson>(settingsPath);
  // Pass the pre-built set so planUnpin/unpinFromModels skip their own scans.
  const plan = planUnpin(snapshot, settings, mid, known);
  if (plan.removed.length === 0) return { status: "not-found", inputModelId: mid };

  // Single owner of models.json write: pruned snapshot derived from plan's base
  const { models: pruned } = unpinFromModels(snapshot, mid, known);
  await atomicWriteJson(modelsPath, pruned);

  if (plan.settingsPatch) {
    await applySettingsPatch(plan.settingsPatch, settingsPath, plan.createdFile);
  }

  if (cap.canUnregister) {
    for (const name of plan.emptied) {
      try {
        (pi as unknown as { unregisterProvider: (n: string) => void }).unregisterProvider(name);
      } catch {
        // Best-effort
      }
    }
    if (plan.repruned.length > 0) {
      const apiKey = resolveApiKey ? await resolveApiKey().catch(() => undefined) : undefined;
      for (const name of plan.repruned) {
        const entry = pruned.providers?.[name];
        if (entry) {
          pi.registerProvider(name, apiKey ? { ...entry, apiKey } : entry);
        }
      }
    }
  }

  return { status: "removed", resolvedModelId: normalized };
}

/**
 * Probe pi/ctx for capability to unregister providers and set models live.
 * Returns { canUnregister, canSetModel, live } — used once per invocation.
 * `setModel` may live on `pi` or on the command `ctx` depending on pi version.
 */
export function probe(pi: ExtensionAPI, ctx?: unknown): CapabilityProbe {
  const piRec = pi as unknown as Record<string, unknown>;
  const ctxRec = (ctx ?? {}) as Record<string, unknown>;
  const canUnregister =
    typeof piRec.unregisterProvider === "function" ||
    typeof ctxRec.unregisterProvider === "function";
  const canSetModel =
    typeof piRec.setModel === "function" || typeof ctxRec.setModel === "function";
  return { canUnregister, canSetModel, live: canUnregister || canSetModel };
}

/**
 * Apply a settings patch: write the updated settings.json.
 * If createdFile is true and the patch clears everything, remove the file.
 * Handles clearing via `'key' in patch` so `undefined` means delete.
 */
async function applySettingsPatch(
  patch: SettingsPatch,
  settingsPath: string,
  createdFile: boolean,
): Promise<void> {
  const beforeRaw = await readJsonFile<SettingsJson>(settingsPath);
  const settings: SettingsJson = beforeRaw ? { ...beforeRaw } : {};
  const beforeJson = JSON.stringify(settings);

  if (Object.prototype.hasOwnProperty.call(patch, "defaultProvider")) {
    if (patch.defaultProvider === undefined) delete (settings as Record<string, unknown>).defaultProvider;
    else settings.defaultProvider = patch.defaultProvider;
  }
  if (Object.prototype.hasOwnProperty.call(patch, "defaultModel")) {
    if (patch.defaultModel === undefined) delete (settings as Record<string, unknown>).defaultModel;
    else settings.defaultModel = patch.defaultModel;
  }
  if (Object.prototype.hasOwnProperty.call(patch, "enabledModel")) {
    const enabled = patch.enabledModel ? patch.enabledModel.split(",").filter(Boolean) : [];
    // Preserve empty array semantics (all pruned) rather than deleting the key;
    // write-only-on-change below will skip if no actual change.
    (settings as Record<string, unknown>).enabledModels = enabled;
    // If the caller wants to truly clear the key when empty, uncomment:
    // if (enabled.length === 0) delete (settings as Record<string, unknown>).enabledModels;
  }

  const afterJson = JSON.stringify(settings);
  if (beforeJson === afterJson) return;

  if (Object.keys(settings).length === 0 && createdFile) {
    try {
      const { rmSync } = await import("node:fs");
      rmSync(settingsPath);
    } catch {
      // Best-effort removal.
    }
    return;
  }
  await atomicWriteJson(settingsPath, settings);
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
// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** Build a set of all bare model ids from every provider in a models.json snapshot. */
function collectKnownBareIds(providers: ModelsJson["providers"]): Set<string> {
  const ids = new Set<string>();
  for (const [, provider] of Object.entries(providers ?? {})) {
    for (const m of Array.isArray(provider?.models) ? provider.models : []) {
      ids.add(m.id);
    }
  }
  return ids;
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

export function normalizeUnpinArg(input: string, knownBareIds: ReadonlySet<string>): string {
  const prefixIndex = input.indexOf('/');
  if (prefixIndex <= 0) {
    return input;
  }
  const prefix = input.slice(0, prefixIndex);
  // Match openrouter-<slug>/ where slug may contain lowercase letters, digits,
  // hyphens (e.g. z-ai, google-vertex), underscores, or a trailing -plus
  // for relaxed pins.
  if (!/^openrouter-[a-z0-9-]+(-plus)?$/i.test(prefix)) {
    return input;
  }
  const candidate = input.slice(prefixIndex + 1);
  return knownBareIds.has(candidate) ? candidate : input;
}
