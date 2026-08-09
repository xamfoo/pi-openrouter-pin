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
    } else if (t.startsWith("--")) return { error: `Unknown flag: ${t}` };
    else positional.push(t);
  }
  if (positional.length < 2) return { error: PIN_USAGE };
  if (positional.length > 2) return { error: `Too many arguments: ${positional.slice(2).join(" ")}` };
  pin.modelId = positional[0];
  pin.slug = positional[1];
  if (!/^[a-z0-9-]+$/.test(pin.slug)) return { error: `Invalid provider "${pin.slug}" (use lowercase, e.g. novita)` };
  return pin;
}
