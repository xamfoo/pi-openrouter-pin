# pi-openrouter-pin

> This extension is written with the help of AI coding assistants and reviewed
> by humans.

Pin an OpenRouter model to a specific OpenRouter provider (one that serves it) as
a dedicated, persistent pi provider. Persistence is pi-native: the pin writes an
`openrouter-<provider>` entry into `~/.pi/agent/models.json`, which pi loads
at every startup — no in-memory state.

## Install

```bash
pi install npm:pi-openrouter-pin    # once published
pi install /path/to/this/package    # local
```

## Usage

### `/openrouter-pin`

```
/openrouter-pin [<model-id> <provider>] [--quant q] [--name <name>] [--default]
                [--fallback]
                [--order <provider1>[,<provider2>]...]
                [--ignore <provider1>[,<provider2>]...]
                [--data-collection allow|deny]
```

Pins the model to the OpenRouter provider as pi provider `openrouter-<provider>`
so requests go only to the pinned provider — and optionally makes it the default
model.

Interactive wizard — a tabbed flow with a breadcrumb trail of the steps and
values chosen so far (Model › Provider › Quant › Name › Routing › Default):

```bash
/openrouter-pin
```

Every selector is a filter line + list: **type to fuzzy-match** (provider,
quant, routing, …). The model step ranks the **full locally-fetched catalog**
(exact → prefix → fuzzy over id + name), so results appear instantly from the
first character with no server round-trip. `Tab`/`Enter` commit the current
step and advance, `Shift+Tab` steps back, `Esc` cancels. Choosing Routing =
Custom inserts three extra steps (order, ignore, data-collection). Long
breadcrumbs soft-wrap instead of truncating.

Non-wizard tab completion supports model ids, providers, and flags:

```
/openrouter-pin [<input>]<tab>...
```

Pin as `--default`:

```bash
/openrouter-pin qwen/qwen3.8-max alibaba --default
# → openrouter-alibaba
```

Set `--quant` for non-default pin:

```bash
/openrouter-pin deepseek/deepseek-v4-flash-0731 baseten --quant fp8
# → openrouter-baseten
```

Set a display `--name` for the pinned model in pi's model picker:

```bash
/openrouter-pin z-ai/glm-5.2 z-ai --name 'GLM 5.2 on Z AI'
# → openrouter-z-ai
```

When omitted, the name is generated from the catalog entry: `<model name>
(<provider> [<quant>])`, e.g. `Z.ai: GLM 5.2 (z-ai)`. An empty name (e.g.
`--name ''`) falls back to the generated one, never a blank picker entry. Quote
the name — it may contain spaces — and note that a name starting with `--` is
rejected (it's almost certainly a forgotten flag value); the wizard's Name step
accepts anything.

Allow OpenRouter to `--fallback` to other providers:

```bash
/openrouter-pin google/gemma-4-26b-a4b-it:free google-ai-studio --fallback
# → openrouter-google-ai-studio-plus
```

Note the `-plus` suffix when routing policy is less strict than default.

Set a preference `--order` — the pinned provider and the listed ones are tried
in order; if none can serve the request, OpenRouter falls back to other
providers:

```bash
/openrouter-pin openai/gpt-oss-120b groq --order baseten,together
# → openrouter-groq-plus
```

Fallback but `--ignore`:

```bash
/openrouter-pin z-ai/glm-5.2 z-ai --ignore baseten
# → openrouter-z-ai-plus
```

`--data-collection` is a privacy flag, and applies to strict and relaxed pins.
If omitted, OpenRouter's default policy is used. Set it like this:

```bash
/openrouter-pin minimax/minimax-m3 minimax --data-collection deny
# → openrouter-minimax
```

### `/openrouter-pins`

```
/openrouter-pins
```

List pins.

### `/openrouter-unpin`

```
/openrouter-unpin <model-id>
```

Remove a pin.

## Development

```bash
npm install     # peer deps (pi packages) + @types/node, for typechecking & tests
npm run typecheck
npm test        # node --test (Node 22.6+ type stripping, or Node 24+)
```

Start at `src/index.ts` — it registers the three slash commands and wires
together the pieces. From there the call chain is short: `src/commands.ts`
holds the core logic (the IO edges), delegating to the pure builders in
`src/config.ts`; `src/args.ts` parses the one-shot form and `src/wizard.ts`
drives the interactive flow. Tests live in `test/`, one file per module,
runnable with plain `node --test`:

```text
├── src/
│   ├── index.ts        # entry point: registers /openrouter-pin, /openrouter-pins, /openrouter-unpin
│   ├── commands.ts     # core logic (IO edges): performPin, performUnpin, listPins, refreshPinnedModels
│   ├── config.ts       # pure config builders (zero imports): buildPin, providerNameFor, slugify
│   ├── args.ts         # pure parser for the one-shot /openrouter-pin form
│   ├── wizard.ts       # the interactive tabbed wizard, one custom TUI component
│   ├── api.ts          # OpenRouter client with per-instance caches + API-key resolution
│   ├── files.ts        # JSONC-tolerant, atomic read/write of models.json / settings.json
│   ├── ui.ts           # fuzzy type-to-filter picker (falls back to pi's native selector)
│   └── completions.ts  # TAB completion: model id → provider slug → flags
└── test/
    ├── args.test.ts    # arg parser: snapshot/behavior tests against a catalog fixture
    ├── config.test.ts  # pure builders, same fixture
    ├── wizard.test.ts  # end-to-end wizard via a TUI harness against temp files; asserts written models.json/settings.json
    ├── pins.test.ts    # list/unpin three ways: pure surface, extracted core, real registered handlers (fake ExtensionAPI, temp PI_CODING_AGENT_DIR)
    ├── refresh.test.ts # pricing refresh: re-read-on-write concurrency with a mocked client + its pure phases
    ├── schema.test.ts  # live schema check vs OpenRouter /models (runs when OPENROUTER_API_KEY is set)
    └── …               # api.test.ts, fixtures/ (catalog-glm-5.2.json, pin-glm-5.2-novita.snapshot.json)
```
