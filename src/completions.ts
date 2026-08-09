/**
 * TAB completion for /openrouter-pin: model ids → provider slugs → flags.
 */
import { fuzzyFilter } from "@earendil-works/pi-tui";
import type { OpenRouterClient } from "./api.ts";
import { pinnedProviderSlugs, providerLabel } from "./api.ts";
import { COMMON_QUANTIZATIONS, slugify } from "./config.ts";
import { rankModelsForQuery } from "./commands.ts";

const FLAGS = ["--default", "--quant", "--name", "--order", "--ignore", "--fallback", "--data-collection"];
const QUANT_FLAGS = COMMON_QUANTIZATIONS.filter((q) => q !== "none");

export type PinCompletion = { value: string; label: string };

/**
 * Build the async completion function for /openrouter-pin.
 *
 * pi's editor calls getArgumentCompletions with everything typed after
 * "/openrouter-pin " and, on accept, replaces ALL of that text with the
 * chosen item's `value`. Values must therefore re-emit the already-typed
 * tokens (first arg = model id, second = "<model> <slug>", flags =
 * "<model> <slug> --flag", ...) — otherwise accepting a later argument
 * silently drops everything before it. Trailing whitespace means the cursor
 * just started a fresh token, so the previous token is complete and the
 * current-token filter is empty.
 */
export function makePinCompletions(
  client: OpenRouterClient,
  modelsPath: string,
): (prefix: string) => Promise<PinCompletion[] | null> {
  return async (prefix: string): Promise<PinCompletion[] | null> => {
    const atTokenStart = /\s$/.test(prefix);
    const tokens = prefix.trimEnd().split(/\s+/).filter(Boolean);
    const last = tokens[tokens.length - 1] ?? "";
    // 0-based positional index of the argument currently being completed.
    const argIndex = atTokenStart ? tokens.length : Math.max(tokens.length - 1, 0);
    // Filter within the current token; empty when a fresh token was just started.
    const filter = atTokenStart ? "" : last;
    // Re-emit everything typed before the current argument so pi's
    // whole-prefix replacement keeps earlier tokens intact.
    const withPrevious = (value: string): string => {
      const before = tokens.slice(0, argIndex).join(" ");
      return before ? `${before} ${value}` : value;
    };

    // Live provider slugs for the model in tokens[0], else the user's own
    // pinned providers, else nothing (completions can't resolve the registry
    // key, so this is best-effort; the pin itself validates authoritatively).
    const slugCandidates = async (): Promise<string[]> => {
      const modelId = tokens[0];
      if (modelId) {
        try {
          const apiKey = process.env.OPENROUTER_API_KEY;
          const { endpoints } = await client.fetchModelEndpoints(modelId, apiKey);
          const slugs = [...new Set(endpoints.map((e) => slugify(e.provider_name ?? "")))].filter(Boolean).sort();
          if (slugs.length > 0) return slugs;
        } catch (err) {
          console.error(`[openrouter-pin] slug completion failed for ${modelId}:`, err);
        }
      }
      return pinnedProviderSlugs(modelsPath);
    };

    if (argIndex === 0) {
      try {
        const catalog = await client.fetchCatalog();
        const matches = rankModelsForQuery(catalog, filter).slice(0, 20);
        return matches.map((m) => ({ value: withPrevious(m.id), label: m.id }));
      } catch (err) {
        console.error("[openrouter-pin] model completion failed:", err);
        return null;
      }
    }

    if (argIndex === 1) {
      // Complete provider slugs for the model typed as the first token.
      const candidates = await slugCandidates();
      if (candidates.length === 0) return null;
      return fuzzyFilter(candidates, filter, (s) => s)
        .map((s) => ({ value: withPrevious(s), label: providerLabel(s) }));
    }

    // argIndex >= 2: flags and their values.
    if (atTokenStart) {
      // A fresh token was just started: the previous token decides whether
      // we complete a flag's value (--quant, --order, --ignore,
      // --data-collection) or a new flag name.
      const prev = tokens[argIndex - 1];
      if (prev === "--quant") {
        const before = tokens.join(" ");
        return QUANT_FLAGS.map((q) => ({ value: `${before} ${q}`, label: q }));
      }
      if (prev === "--order" || prev === "--ignore") {
        const candidates = await slugCandidates();
        if (candidates.length === 0) return null;
        const before = tokens.join(" ");
        return candidates.map((s) => ({ value: `${before} ${s}`, label: s }));
      }
      if (prev === "--data-collection") {
        const before = tokens.join(" ");
        return ["allow", "deny"].map((v) => ({ value: `${before} ${v}`, label: v }));
      }
      if (prev === "--name") return null; // free text
      return FLAGS.map((f) => ({ value: withPrevious(f), label: f }));
    }

    if (last.startsWith("--")) {
      // Completing a flag name itself.
      return fuzzyFilter(FLAGS, last, (f) => f).map((f) => ({ value: withPrevious(f), label: f }));
    }

    // Completing the value of the flag typed before the current token.
    const prev = tokens[argIndex - 1];
    if (prev === "--quant") {
      const before = tokens.slice(0, argIndex).join(" ");
      return fuzzyFilter(QUANT_FLAGS, last, (q) => q)
        .map((q) => ({ value: `${before} ${q}`, label: q }));
    }
    if (prev === "--order" || prev === "--ignore") {
      const candidates = await slugCandidates();
      if (candidates.length === 0) return null;
      const before = tokens.slice(0, argIndex).join(" ");
      return fuzzyFilter(candidates, last, (s) => s)
        .map((s) => ({ value: `${before} ${s}`, label: s }));
    }
    if (prev === "--data-collection") {
      const before = tokens.slice(0, argIndex).join(" ");
      return fuzzyFilter(["allow", "deny"], last, (v) => v)
        .map((v) => ({ value: `${before} ${v}`, label: v }));
    }
    if (prev === "--name") return null; // free text
    return null;
  };
}
