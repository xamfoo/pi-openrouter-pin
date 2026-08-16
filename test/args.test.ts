/**
 * Tests for the command-line parser (src/args.ts).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { isHelpRequest, parsePinArgs, PIN_HELP, PINS_HELP, tokenize, UNPIN_HELP } from "../src/args.ts";

test("tokenize: double quotes, escaped quotes, single quotes, backslashes", () => {
  assert.deepEqual(tokenize('z-ai/glm-5.2 novita --quant fp8 --name "GLM 5.2"'), [
    "z-ai/glm-5.2",
    "novita",
    "--quant",
    "fp8",
    "--name",
    "GLM 5.2",
  ]);
  assert.deepEqual(tokenize('m --name "a \\"b\\" c"'), ["m", "--name", 'a "b" c']);
  assert.deepEqual(tokenize("m --name 'single quoted'"), ["m", "--name", "single quoted"]);
  assert.deepEqual(tokenize("a\\ b"), ["a b"]);
  assert.deepEqual(tokenize("  spaced   tokens  "), ["spaced", "tokens"]);
});

test("parsePinArgs: full flag surface", () => {
  const parsed = parsePinArgs(
    'z-ai/glm-5.2 novita --quant fp8 --name "GLM 5.2" --default --order "Novita, together" --ignore openai --fallback --data-collection deny',
  );
  assert.ok(!("error" in parsed));
  assert.equal(parsed.modelId, "z-ai/glm-5.2");
  assert.equal(parsed.slug, "novita");
  assert.equal(parsed.quant, "fp8");
  assert.equal(parsed.name, "GLM 5.2");
  assert.equal(parsed.isDefault, true);
  assert.equal(parsed.allowFallbacks, true);
  assert.deepEqual(parsed.order, ["novita", "together"]);
  assert.deepEqual(parsed.ignore, ["openai"]);
  assert.equal(parsed.dataCollection, "deny");
});

test("parsePinArgs: errors are reported, not swallowed", () => {
  assert.ok("error" in parsePinArgs("only-one-arg"));
  assert.ok("error" in parsePinArgs("a b --bogus"));
  assert.ok("error" in parsePinArgs("a BAD_SLUG"));
});

test("parsePinArgs: --data-collection is orthogonal — it parses on any pin, value is validated", () => {
  // Alone on a strict pin: allowed. data_collection is a privacy flag that
  // works regardless of routing strictness; strict pins announce it at pin
  // time and in /openrouter-pins.
  const alone = parsePinArgs("a b --data-collection deny");
  assert.ok(!("error" in alone));
  assert.equal(alone.dataCollection, "deny");
  assert.equal(alone.allowFallbacks, undefined);
  // Bad or missing values are rejected.
  assert.ok("error" in parsePinArgs("a b --data-collection maybe"));
  const missingValue = parsePinArgs("a b --data-collection");
  assert.ok("error" in missingValue);
  assert.match(missingValue.error, /--data-collection expects a value/);
  // Works alongside relaxations too, in either order.
  const withFallback = parsePinArgs("a b --data-collection deny --fallback");
  assert.ok(!("error" in withFallback));
  assert.equal(withFallback.dataCollection, "deny");
  assert.equal(withFallback.allowFallbacks, true);
  const withOrder = parsePinArgs("a b --order together --data-collection allow");
  assert.ok(!("error" in withOrder));
  assert.equal(withOrder.dataCollection, "allow");
  // The renamed flag is --fallback; the old spelling is gone.
  assert.ok("error" in parsePinArgs("a b --allow-fallbacks"));
  assert.ok(!("error" in parsePinArgs("a b --fallback")));
});

test("isHelpRequest: -h / --help in any position, exact token only", () => {
  assert.equal(isHelpRequest(""), false);
  assert.equal(isHelpRequest("z-ai/glm-5.2 novita"), false);
  assert.equal(isHelpRequest("--help"), true);
  assert.equal(isHelpRequest("-h"), true);
  assert.equal(isHelpRequest("z-ai/glm-5.2 novita --help"), true);
  assert.equal(isHelpRequest("--quant fp8 -h"), true);
  // Token-based: a quoted "--help" is still its own token after tokenize.
  assert.equal(isHelpRequest('--name "--help"'), true);
  // A token that merely contains -h inside a word is not a help request.
  assert.equal(isHelpRequest("--name sh-orthand"), false);
});

test("parsePinArgs: -h / --help short-circuit positional and slug checks", () => {
  const long = parsePinArgs("--help");
  assert.ok(!("error" in long));
  assert.equal(long.help, true);
  const short = parsePinArgs("-h");
  assert.ok(!("error" in short));
  assert.equal(short.help, true);
  // Help wins even alongside valid or invalid pin arguments.
  const mixed = parsePinArgs("z-ai/glm-5.2 novita --quant fp8 --help");
  assert.ok(!("error" in mixed));
  assert.equal(mixed.help, true);
  const noPositionals = parsePinArgs("-h");
  assert.ok(!("error" in noPositionals), "bare -h never trips the usage error");
  assert.equal(noPositionals.help, true);
});

test("help text: PIN_HELP / UNPIN_HELP / PINS_HELP carry usage and --help", () => {
  for (const text of [PIN_HELP, UNPIN_HELP, PINS_HELP]) {
    assert.ok(text.includes("Usage:"), "each help text has a usage line");
    assert.ok(text.includes("--help"), "each help text mentions --help");
  }
  // PIN_HELP documents the pin-specific flag surface; the others stay short.
  assert.ok(PIN_HELP.includes("--quant"));
  assert.ok(PIN_HELP.includes("--order"));
  assert.ok(PIN_HELP.includes("--data-collection"));
  assert.ok(UNPIN_HELP.includes("model-id"));
  assert.ok(UNPIN_HELP.includes("model-id") && UNPIN_HELP.split("\n").length < 15, "unpin help stays concise");
});

test("parsePinArgs: value-taking flags error when the value is missing, never silently ignore it", () => {
  const noQuant = parsePinArgs("m novita --quant");
  assert.ok("error" in noQuant);
  assert.match(noQuant.error, /--quant expects a value/);
  const noName = parsePinArgs("m novita --name");
  assert.ok("error" in noName);
  assert.match(noName.error, /--name expects a value/);
  // --quant must not consume a following flag as its value.
  const quantEatsFlag = parsePinArgs("m novita --quant --default");
  assert.ok("error" in quantEatsFlag);
  assert.match(quantEatsFlag.error, /--quant expects a value/);
  // --name has the same guard: swallowing the flag would silently drop it.
  const nameEatsFlag = parsePinArgs("m novita --name --fallback");
  assert.ok("error" in nameEatsFlag);
  assert.match(nameEatsFlag.error, /--name expects a value/);
  // A present value still parses.
  assert.ok(!("error" in parsePinArgs("m novita --quant fp8 --name 'GLM 5.2'")));
});
