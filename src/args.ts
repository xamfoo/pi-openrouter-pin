/**
 * Command-line argument parsing for /openrouter-pin (pure, testable).
 */
import { slugify } from "./config.ts";

export interface PinArgs {
  modelId?: string;
  slug?: string;
  quant?: string;
  name?: string;
  order?: string[];
  ignore?: string[];
  allowFallbacks?: boolean;
  dataCollection?: "allow" | "deny";
  isDefault: boolean;
  /** Set when the command line contains -h / --help; parsing short-circuits. */
  help?: boolean;
}

/**
 * Split a command line into tokens, honoring single quotes, double quotes
 * (with \" and \\ escapes), and backslash escapes. Unclosed quotes are
 * consumed leniently to the end, like a shell.
 */
export function tokenize(args: string): string[] {
  const tokens: string[] = [];
  let current = "";
  let inToken = false;
  let i = 0;
  while (i < args.length) {
    const c = args[i];
    if (c === "\\") {
      current += args[i + 1] ?? "";
      inToken = true;
      i += 2;
      continue;
    }
    if (c === "'" || c === '"') {
      inToken = true;
      const quote = c;
      i++;
      while (i < args.length && args[i] !== quote) {
        if (quote === '"' && args[i] === "\\" && (args[i + 1] === '"' || args[i + 1] === "\\")) {
          current += args[i + 1];
          i += 2;
          continue;
        }
        current += args[i];
        i++;
      }
      i++; // closing quote (or past end for an unclosed quote)
      continue;
    }
    if (/\s/.test(c)) {
      if (inToken) {
        tokens.push(current);
        current = "";
        inToken = false;
      }
      i++;
      continue;
    }
    current += c;
    inToken = true;
    i++;
  }
  if (inToken) tokens.push(current);
  return tokens;
}

function parseSlugListFlag(value: string | undefined, flag: string): { error: string } | string[] {
  if (!value || !value.trim()) return { error: `${flag} expects a comma-separated list of providers` };
  const slugs = value.split(",").map((s) => slugify(s.trim())).filter(Boolean);
  if (slugs.length === 0) return { error: `${flag} expects at least one provider` };
  return slugs;
}

export const PIN_USAGE =
  "Usage: /openrouter-pin <model-id> <provider> [--quant q] [--name 'Display'] [--default] " +
  "[--order a,b,c] [--ignore a,b] [--fallback] [--data-collection allow|deny]";

// ---------------------------------------------------------------------------
// Help text for the three commands (shown by `-h` / `--help`)
// ---------------------------------------------------------------------------

/** Help for /openrouter-pin: usage, flags, and examples. */
export const PIN_HELP = [
  "Usage: /openrouter-pin <model-id> <provider> [flags]",
  "",
  "Pin an OpenRouter model to a specific provider as a persistent pi provider",
  "(writes models.json; applies on /reload or next session). With no arguments",
  "the interactive wizard opens instead.",
  "",
  "Flags:",
  "  --quant q            quantization to lock (e.g. fp8)",
  "  --name 'Display'     display name for the pinned model",
  "  --default            also make the pin the default pi model",
  "  --order a,b,c        preferred provider order (relaxed pin)",
  "  --ignore a,b         providers to exclude (relaxed pin)",
  "  --fallback           allow fallbacks to other providers (relaxed pin)",
  "  --data-collection    allow|deny — data retention policy",
  "  -h, --help           show this help",
  "",
  "Examples:",
  "  /openrouter-pin z-ai/glm-5.2 novita --quant fp8",
  "  /openrouter-pin openai/gpt-oss-120b groq --order baseten,together",
].join("\n");

/** Help for /openrouter-unpin: usage and behavior. */
export const UNPIN_HELP = [
  "Usage: /openrouter-unpin [model-id]",
  "",
  "Remove a pinned OpenRouter model from models.json (applies on /reload or",
  "next session). Without an argument, picks from the existing pins.",
  "",
  "Flags:",
  "  -h, --help           show this help",
].join("\n");

/** Help for /openrouter-pins: usage and behavior. */
export const PINS_HELP = [
  "Usage: /openrouter-pins",
  "",
  "List every pinned OpenRouter model with its provider and routing policy",
  "(only, order, ignore, quant, fallbacks, data collection).",
  "",
  "Flags:",
  "  -h, --help           show this help",
].join("\n");

/**
 * True when the command line contains a -h / --help token in any position.
 * Token-based, so a flag value that merely contains "--help" (quoted or
 * escaped) does not count unless it is its own token.
 */
export function isHelpRequest(args: string): boolean {
  return tokenize(args).some((t) => t === "-h" || t === "--help");
}

export function parsePinArgs(args: string): { error: string } | PinArgs {
  const tokens = tokenize(args);
  const pin: PinArgs = { isDefault: false };
  const positional: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (t === "--default") pin.isDefault = true;
    else if (t === "--quant") {
      pin.quant = tokens[++i];
      // A quant is constrained vocabulary; never let it silently consume a
      // following flag (or vanish at end of line) as its value.
      if (pin.quant === undefined || pin.quant.startsWith("--")) {
        return { error: "--quant expects a value" };
      }
    } else if (t === "--name") {
      pin.name = tokens[++i];
      // Same guard as --quant: a display name beginning with "--" is almost
      // certainly a forgotten value (a quoted name arrives as one token), and
      // swallowing the flag would silently drop it. Pathological dash-names
      // can still be set via the wizard.
      if (pin.name === undefined || pin.name.startsWith("--")) {
        return { error: "--name expects a value" };
      }
    } else if (t === "--fallback") pin.allowFallbacks = true;
    else if (t === "--data-collection") {
      const v = tokens[++i];
      if (v === undefined) return { error: "--data-collection expects a value (allow or deny)" };
      if (v !== "allow" && v !== "deny") return { error: `--data-collection expects "allow" or "deny", got "${v}"` };
      pin.dataCollection = v;
    } else if (t === "--order" || t === "--ignore") {
      const parsed = parseSlugListFlag(tokens[++i], t);
      if ("error" in parsed) return parsed;
      if (t === "--order") pin.order = parsed;
      else pin.ignore = parsed;
    } else if (t === "-h" || t === "--help") pin.help = true;
    else if (t.startsWith("--")) return { error: `Unknown flag: ${t}` };
    else positional.push(t);
  }
  // Help short-circuits: no positional-count or slug validation is needed.
  if (pin.help) return pin;
  if (positional.length < 2) return { error: PIN_USAGE };
  if (positional.length > 2) return { error: `Too many arguments: ${positional.slice(2).join(" ")}` };
  pin.modelId = positional[0];
  pin.slug = positional[1];
  if (!/^[a-z0-9-]+$/.test(pin.slug)) return { error: `Invalid provider "${pin.slug}" (use lowercase, e.g. novita)` };
  return pin;
}
