# pi-openrouter-pin

Pin an OpenRouter model to a specific host provider as a dedicated,
persistent pi provider. No in-memory state, no proxy daemons, no backend
quantization roulette.

> This extension is written with the help of AI coding assistants and reviewed
> by humans.

## Install

```bash
pi install npm:@xamfoo/pi-openrouter-pin
```

## Usage

Every command accepts `--help` (or `-h`) to print its usage. Use `Esc` to stop
tab completion:

```bash
/openrouter-pin --help
/openrouter-unpin --help
/openrouter-pins --help
```

### `/openrouter-pin`

Run without arguments to launch the interactive wizard:

```bash
/openrouter-pin
```

Expect a tabbed flow with instant fuzzy-matching against OpenRouter's full local
catalog (Model › Provider › Quant › Name › Routing › Default). `Tab`/`Enter`
advances, `Shift+Tab` steps back, `Esc` cancels.

Alternatively, pin directly from the command line:

```bash
# Lock to a specific provider and quantization (e.g. avoid 4-bit degradation)
/openrouter-pin deepseek/deepseek-v4-flash-0731 baseten --quant fp8

# Exclude specific unreliable providers
/openrouter-pin z-ai/glm-5.2 z-ai --ignore deepinfra,venice

# Set provider preference order
/openrouter-pin openai/gpt-oss-120b groq --order baseten,together

# Enforce zero data retention / strict privacy policy
/openrouter-pin minimax/minimax-m3 minimax --data-collection deny

# Pin and make it the default model in pi
/openrouter-pin qwen/qwen3.8-max alibaba --default
```

### Routing Policy

By default, pins are strict: requests go only to the specified provider under
`openrouter-<provider>`.

When you relax constraints with `--fallback`, `--order`, or `--ignore`, the
provider is registered with a `-plus` suffix (e.g., `openrouter-groq-plus`) so
the name reflects that fallbacks are allowed. (Naming conventions for relaxed
routing are experimental and subject to change.)

### `/openrouter-pins`

List active pins, their endpoints, and configured routing policies.

```bash
/openrouter-pins
```

### `/openrouter-unpin`

Remove a pinned model and its provider configuration.

```bash
/openrouter-unpin <model-id>
```

## How It Works

* **Per-Endpoint Pricing Truth**: OpenRouter's catalog pricing is an aggregate
  that often reflects the lowest-cost provider rather than the one you call.
  `pi-openrouter-pin` extracts and records the actual endpoint pricing for your
  pinned provider in `~/.pi/agent/models.json`.
* **Startup Drift Sync**: On pi launch, pinned models refresh their pricing and
  context limits against OpenRouter's catalog, updating fields atomically if
  upstream parameters change.
* **Pi-Native Persistence**: Modifies standard `models.json` and
  `settings.json` with atomic writes, preserving formatting and comments.

## Development

```bash
npm install
npm run typecheck
npm test        # runs pure builders, arg parser, and TUI wizard harness
```
