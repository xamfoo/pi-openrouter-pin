/**
 * pi-openrouter-pin — entry point.
 *
 * Pin OpenRouter models to a specific provider (e.g. novita) as a dedicated,
 * persistent pi provider. Persistence is pi-native: the pin writes a provider
 * `openrouter-<provider>` (strict) or `openrouter-<provider>-plus` (relaxed:
 * fallbacks/order/ignore) into `~/.pi/agent/models.json`, which pi loads
 * itself at every startup — no in-memory state, no plugin dependency.
 *
 * A pin always means "at least prefer this provider": strict is the default;
 * --fallback, --order, and --ignore are explicit relaxations.
 *
 * Commands:
 *   /openrouter-pin <model> <provider> [--quant fp8] [--name "Display"] [--default]
 *                  [--order a,b,c] [--ignore a,b] [--fallback]
 *                  [--data-collection allow|deny]     one-shot, scriptable
 *   /openrouter-pin                                       interactive wizard
 *   /openrouter-unpin <model>                             one-shot
 *   /openrouter-unpin                                     pick from existing pins
 *   /openrouter-pins                                      list pins (verbose view)
 */
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { join } from "node:path";
import { CATALOG_CACHE_TTL_MS, ENDPOINT_CACHE_TTL_MS, OpenRouterClient } from "./api.ts";
import { parsePinArgs } from "./args.ts";
import { makePinCompletions } from "./completions.ts";
import { formatRouting, formatRefreshDiff, listPins, performPin, performUnpin, refreshPinnedModels } from "./commands.ts";
import { providerNameFor } from "./config.ts";
import { pickFromList } from "./ui.ts";
import { runWizard } from "./wizard.ts";
import { resolveOpenRouterApiKey } from "./api.ts";

export default function openrouterPinExtension(pi: ExtensionAPI) {
  const agentDir = getAgentDir();
  const modelsPath = join(agentDir, "models.json");
  const settingsPath = join(agentDir, "settings.json");
  const client = new OpenRouterClient(CATALOG_CACHE_TTL_MS, ENDPOINT_CACHE_TTL_MS);

  // Refresh pinned model pricing & limits (cost, contextWindow, maxTokens) at
  // session start. Deliberately NOT in the factory: factories run in
  // invocations that never start a session (pi --list-models, --help, RPC
  // health checks) and must not hit the network or rewrite models.json.
  // Fire-and-forget so startup is never blocked; failures are logged, not
  // swallowed. New values apply on the next /reload, which re-fires
  // session_start.
  pi.on("session_start", (_event, ctx) => {
    void refreshPinnedModels(modelsPath, client, () => resolveOpenRouterApiKey(ctx.modelRegistry))
      .then((r) => {
        if (r.refreshed > 0) {
          console.log(`[openrouter-pin] refreshed ${r.refreshed} pinned model pricing & limits — /reload to apply`);
          const diff = formatRefreshDiff(r.diff);
          if (diff) console.log(diff);
        } else if (r.failed.length > 0) {
          console.warn(`[openrouter-pin] pricing & limits refresh unavailable for ${r.failed.length} model(s): ${r.failed.join(", ")}`);
        }
      })
      .catch((err) => {
        console.error("[openrouter-pin] pricing & limits refresh failed:", err);
      });
  });

  const pinCompletions = makePinCompletions(client, modelsPath);

  pi.registerCommand("openrouter-pin", {
    description:
      "Pin an OpenRouter model to a specific provider (persistent, models.json). " +
      "No args opens an interactive wizard. With args: /openrouter-pin <model-id> <provider> " +
      "[--quant q] [--name 'Display'] [--default] [--order a,b,c] [--ignore a,b] [--fallback] [--data-collection allow|deny]",
    getArgumentCompletions: (prefix) => pinCompletions(prefix),
    handler: async (args, ctx: ExtensionCommandContext) => {
      if (!args.trim()) {
        await runWizard(modelsPath, settingsPath, pi, ctx.ui, client, ctx.modelRegistry);
        return;
      }
      const parsed = parsePinArgs(args);
      if ("error" in parsed) {
        ctx.ui.notify(parsed.error, "error");
        return;
      }
      await performPin(modelsPath, settingsPath, pi, ctx.ui, client, () =>
        resolveOpenRouterApiKey(ctx.modelRegistry, providerNameFor(parsed.slug!, parsed)), {
        modelId: parsed.modelId!,
        slug: parsed.slug!,
        quant: parsed.quant,
        // Empty/quoted-whitespace names fall back to the generated one, same
        // as the wizard ("Z.ai: GLM 5.2 (novita)") — never a blank picker entry.
        name: parsed.name?.trim() || undefined,
        isDefault: parsed.isDefault,
        allowFallbacks: parsed.allowFallbacks,
        order: parsed.order,
        ignore: parsed.ignore,
        dataCollection: parsed.dataCollection,
      });
    },
  });

  pi.registerCommand("openrouter-unpin", {
    description:
      "Remove an OpenRouter provider pin from models.json. No args picks from existing pins. " +
      "With args: /openrouter-unpin <model-id> (applies on /reload or next session)",
    handler: async (args, ctx: ExtensionCommandContext) => {
      try {
        let modelId = args.trim().split(/\s+/)[0];
        if (!modelId) {
          const pins = await listPins(modelsPath);
          if (pins.length === 0) {
            ctx.ui.notify("No pins to remove. Use /openrouter-pin to create one.", "info");
            return;
          }
          const chosen = await pickFromList(
            ctx.ui,
            "Pick a pin to remove",
            pins.map((p) => `${p.provider}/${p.model.id}`),
          );
          if (!chosen) {
            return; // Esc: cancel quietly, no notification
          }
          modelId = chosen.slice(chosen.indexOf("/") + 1);
        }
        const outcome = await performUnpin(modelsPath, modelId);
        if (outcome.status === "no-providers") {
          ctx.ui.notify("No pins found (no providers in models.json)", "info");
        } else if (outcome.status === "not-found") {
          ctx.ui.notify(`No pin for "${modelId}" found (checked openrouter-* providers)`, "info");
        } else {
          ctx.ui.notify(`Unpinned ${modelId} from models.json (applies on /reload or next session).`, "info");
        }
      } catch (err) {
        ctx.ui.notify(`Unpin failed: ${err instanceof Error ? err.message : String(err)}`, "error");
      }
    },
  });

  pi.registerCommand("openrouter-pins", {
    description: "List all pinned OpenRouter provider routes from models.json",
    handler: async (_args, ctx: ExtensionCommandContext) => {
      try {
        const pins = await listPins(modelsPath);
        ctx.ui.notify(
          pins.length
            ? `Active pins:\n${pins
                .map((p) => {
                  const r = p.model.compat!.openRouterRouting;
                  return `  ${p.provider}/${p.model.id}  → ${formatRouting(r)}`;
                })
                .join("\n")}`
            : "No pins. Run /openrouter-pin (wizard) or /openrouter-pin <model> <provider> [--quant q] [--default] [--fallback]",
          "info",
        );
      } catch (err) {
        ctx.ui.notify(`List failed: ${err instanceof Error ? err.message : String(err)}`, "error");
      }
    },
  });
}
