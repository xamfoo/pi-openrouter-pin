/**
 * Tests for the command-line parser (src/args.ts).
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { parsePinArgs, tokenize } from "../src/args.ts";

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
