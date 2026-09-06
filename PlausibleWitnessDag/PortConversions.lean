import PlausibleWitnessDag

/-! # port-conversion witness

Domain package: certify whether each Python→GGUF conversion HERD produced
under `chibifire/*-gguf` should ship, using the generic iterative-deepening
driver from `PlausibleWitnessDag`.

Sibling to `GgmlForks.lean`. That file's `Evidence` measures the plausibility
of an *upstream mirror*; this file's `ConversionEvidence` measures the
plausibility of *our own conversion* — a roundtrip proof that the produced
GGUF loads on Metal and emits sane tokens against a Python reference.

Candidates from operator directive 2026-09-05 ("port python etc to ggml"):

* `chibifire/gemma-4-12B-it-qat-q4_0-unquantized` → `chibifire/gemma-4-12B-it-qat-q4_0-gguf`
* `chibifire/Qwen2.5-VL-7B-Instruct`              → `chibifire/Qwen2.5-VL-7B-Instruct-gguf`
* `chibifire/qwen3-omni`                          → `chibifire/qwen3-omni-gguf`
* `chibifire/omnigen2-base-df5dca8a`              → `chibifire/omnigen2-gguf`
-/

namespace PlausibleWitnessDag.PortConversions

open PlausibleWitnessDag

/-- Evidence captured from an actual conversion + Metal smoke test.

`kl_scaled` and `perplexity_scaled` are the KL divergence and perplexity ratio
against a Python reference call, each multiplied by 1000 so they're `Nat`s
under `by decide` (Lean 4 kernel doesn't reduce `Float` inequalities). A
smaller `kl_scaled` is better; a `perplexity_scaled` near 1000 is better. -/
structure ConversionEvidence where
  roundtripLoadMetal   : Bool  -- llama-cli init log printed `ggml_metal_init: ... using Metal`
  tokensPerSec         : Nat   -- Metal smoke output tok/s
  klScaled             : Nat   -- KL(python || gguf) × 1000; 0 = identical
  perplexityScaled     : Nat   -- perplexity(gguf) / perplexity(python) × 1000; 1000 = identical
  quantSanctioned      : Bool  -- q4_0 only allowed when upstream QAT was done; else fp16/q8_0
  conversionTool       : String
  deriving Repr, Inhabited

/-- Verdict for a conversion: ship the GGUF, or hold for follow-up. -/
inductive Verdict
  | ships
  | requiresManualReview
  deriving Repr, DecidableEq, Inhabited

/-- Metal must load, tokens/sec ≥ 10, perplexity within ±10% of Python reference
(perplexityScaled ∈ [900, 1100]), KL below 50/1000 = 0.05, and quant must be
sanctioned by CLAUDE.md's PTQ blocklist row (q4/q4_K only allowed when upstream
QAT was done; else fp16 or q8_0). All five must hold. -/
def verdict (e : ConversionEvidence) : Verdict :=
  if e.roundtripLoadMetal
     && e.tokensPerSec ≥ 10
     && e.klScaled ≤ 50
     && 900 ≤ e.perplexityScaled && e.perplexityScaled ≤ 1100
     && e.quantSanctioned then
    Verdict.ships
  else
    Verdict.requiresManualReview

/-- One port candidate = source repo + target gguf repo + conversion evidence. -/
structure PortCandidate where
  source    : String
  target    : String
  quant     : String  -- "q4_0", "q8_0", "f16", "fp16"
  evidence  : ConversionEvidence
  deriving Repr, Inhabited

/-! ## Placeholder evidence bundles.

Real measurements land as HERD's Phase 2 port runs each conversion + Metal
smoke. The bundles below are shaped so a passing conversion drops in cleanly
(swap the numbers, re-run `lake build`). Each `example` witness is written
against a placeholder that would ship IF the numbers hold; the certificates
therefore document the shipping bar, and running `lake build` at any point
re-checks the bar against whatever numbers are inlined at that point.

Placeholders set:
* `roundtripLoadMetal = true`      — assumed until measured
* `tokensPerSec = 15`              — modest above the floor of 10
* `klScaled = 30`                  — small enough (< 50) to ship
* `perplexityScaled = 1010`        — 1% inflation from quant, well within ±10%
* `quantSanctioned` — set per candidate: `true` for Gemma-4 QAT q4_0 and
  everything at fp16/q8_0; would be `false` if a q4/q4_K quant were selected
  on a non-QAT source (which the port plan doesn't do).

If a smoke test produces numbers outside these bounds, edit the bundle to
match reality — a failed conversion should demote the verdict to
`requiresManualReview`, and the `by decide` step will refuse the file until
the bundle is honest about it. -/
def evGemma4Qat_placeholder : ConversionEvidence :=
  { roundtripLoadMetal := true, tokensPerSec := 15,
    klScaled := 30, perplexityScaled := 1010,
    quantSanctioned := true,  -- QAT already done upstream, q4_0 sanctioned
    conversionTool := "llama.cpp convert_hf_to_gguf.py + llama-quantize" }
def evQwen25VL_placeholder : ConversionEvidence :=
  { roundtripLoadMetal := true, tokensPerSec := 15,
    klScaled := 30, perplexityScaled := 1010,
    quantSanctioned := true,  -- fp16 + q8_0, both PTQ-blocklist-safe
    conversionTool := "llama.cpp convert_hf_to_gguf.py (VLM: + mmproj)" }
def evQwen3Omni_placeholder : ConversionEvidence :=
  { roundtripLoadMetal := true, tokensPerSec := 15,
    klScaled := 30, perplexityScaled := 1010,
    quantSanctioned := true,  -- fp16 + q8_0
    conversionTool := "llama.cpp convert_hf_to_gguf.py" }
def evOmniGen2_placeholder : ConversionEvidence :=
  { roundtripLoadMetal := true, tokensPerSec := 15,
    klScaled := 30, perplexityScaled := 1010,
    quantSanctioned := true,  -- fp16 only (diffusion)
    conversionTool := "stable-diffusion.cpp" }

/-- Negative control: a conversion that failed Metal init must NOT ship.
Doctrine: "a check that passes on known-broken input is decoration." -/
def evNoMetal : ConversionEvidence :=
  { roundtripLoadMetal := false, tokensPerSec := 100,
    klScaled := 10, perplexityScaled := 1000,
    quantSanctioned := true, conversionTool := "n/a" }

/-- Negative control: a conversion that used an unsanctioned quant must NOT
ship, even if Metal + tokens + PPL all look fine. -/
def evUnsanctionedQ4 : ConversionEvidence :=
  { roundtripLoadMetal := true, tokensPerSec := 30,
    klScaled := 20, perplexityScaled := 1020,
    quantSanctioned := false,  -- q4_0 on a non-QAT source
    conversionTool := "llama.cpp llama-quantize q4_0" }

example : verdict evGemma4Qat_placeholder = Verdict.ships := by decide
example : verdict evQwen25VL_placeholder  = Verdict.ships := by decide
example : verdict evQwen3Omni_placeholder = Verdict.ships := by decide
example : verdict evOmniGen2_placeholder  = Verdict.ships := by decide

example : verdict evNoMetal          = Verdict.requiresManualReview := by decide
example : verdict evUnsanctionedQ4   = Verdict.requiresManualReview := by decide

end PlausibleWitnessDag.PortConversions
