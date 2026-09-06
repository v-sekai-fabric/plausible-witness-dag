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
smaller `kl_scaled` is better; a `perplexity_scaled` near 1000 is better.

`metricsMeasured` is a hard-required honesty gate: when `false`, the KL and
perplexity fields are placeholders (unmeasured) and the verdict must demote
to `requiresManualReview` regardless of the other axes. This is the workspace
"a check that passes on known-broken input is decoration" rule applied to
missing measurements — a witness that shipped an unmeasured KL as if it were
zero would be decoration, not certification. -/
structure ConversionEvidence where
  roundtripLoadMetal   : Bool  -- llama-cli init log printed `ggml_metal_init: ... using Metal`
  tokensPerSec         : Nat   -- Metal smoke output tok/s
  klScaled             : Nat   -- KL(python || gguf) × 1000; 0 = identical
  perplexityScaled     : Nat   -- perplexity(gguf) / perplexity(python) × 1000; 1000 = identical
  metricsMeasured      : Bool  -- klScaled and perplexityScaled are real measurements, not placeholders
  quantSanctioned      : Bool  -- q4_0 only allowed when upstream QAT was done; else fp16/q8_0
  conversionTool       : String
  deriving Repr, Inhabited

/-- Verdict for a conversion: ship the GGUF, or hold for follow-up. -/
inductive Verdict
  | ships
  | requiresManualReview
  deriving Repr, DecidableEq, Inhabited

/-- Metal must load, tokens/sec ≥ 10, perplexity within ±10% of Python reference
(perplexityScaled ∈ [900, 1100]), KL below 50/1000 = 0.05, quant must be
sanctioned by CLAUDE.md's PTQ blocklist row (q4/q4_K only allowed when upstream
QAT was done; else fp16 or q8_0), AND the KL/perplexity numbers must be real
measurements (metricsMeasured = true). All six must hold. -/
def verdict (e : ConversionEvidence) : Verdict :=
  if e.roundtripLoadMetal
     && e.tokensPerSec ≥ 10
     && e.klScaled ≤ 50
     && 900 ≤ e.perplexityScaled && e.perplexityScaled ≤ 1100
     && e.metricsMeasured
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

/-! ## Real Phase 2 evidence bundles.

Numbers below are HERD's measured Phase 2 output. `tokensPerSec` is the
Metal-backed generation rate reported by `llama-cli`; `roundtripLoadMetal` is
`true` when the `ggml_metal_init` log line appeared during load. KL divergence
and perplexity ratio against a Python reference were NOT measured this round
(no `verify_perplexity.py` invocation), so `metricsMeasured` is `false` on
every real bundle — the verdict then correctly demotes to
`requiresManualReview` even though Metal + tokens + quant-sanction all pass.

To promote a candidate to `ships`, land the KL + perplexity measurement,
edit the two Nat fields with the real numbers, and flip `metricsMeasured` to
`true`. The `by decide` steps at the end of the file will re-check the bar. -/

/-- chibifire/gemma-4-12B-it-qat-q4_0-gguf — Phase 2 measured. -/
def evGemma4Qat : ConversionEvidence :=
  { roundtripLoadMetal := true,
    tokensPerSec       := 22,      -- measured on this Mac, Metal backend
    klScaled           := 0,       -- not measured
    perplexityScaled   := 0,       -- not measured
    metricsMeasured    := false,   -- KL + perplexity not run
    quantSanctioned    := true,    -- QAT already done upstream, q4_0 sanctioned
    conversionTool     := "llama.cpp convert_hf_to_gguf.py + llama-quantize" }

/-- chibifire/Qwen2.5-VL-7B-Instruct-gguf — Phase 2 measured (text-only path;
mmproj follow-up separate). -/
def evQwen25VL : ConversionEvidence :=
  { roundtripLoadMetal := true,
    tokensPerSec       := 22,      -- measured on this Mac, Metal backend
    klScaled           := 0,       -- not measured
    perplexityScaled   := 0,       -- not measured
    metricsMeasured    := false,   -- KL + perplexity not run
    quantSanctioned    := true,    -- fp16 + q8_0, both PTQ-blocklist-safe
    conversionTool     := "llama.cpp convert_hf_to_gguf.py (VLM text path)" }

/-- chibifire/qwen3-omni-gguf — Phase 3 actual. Source repo populated from
`Qwen/Qwen3-Omni-30B-A3B-Instruct` @ `26291f793822fb6be9555850f06dfe95f2d7e695`
(70.5 GB safetensors → chibifire/qwen3-omni); conversion at q8_0 → 32.5 GB
via `llama.cpp convert_hf_to_gguf.py` (`Qwen3OmniMoeTextModel` in
`conversion/qwen3vl.py`). Uploaded to chibifire/qwen3-omni-gguf.

Metal smoke: `ggml_metal_init` ran and produced output tokens before the
30B q8_0 model OOM'd Metal's working set on this 32 GB Mac
(`kIOGPUCommandBufferCallbackErrorOutOfMemory` on command buffer 0). Metal
init itself succeeded — the error is a hardware ceiling, not a Metal-backend
failure. `tokensPerSec` therefore unmeasurable on this desk; a Mac with
larger unified memory (or CUDA with 32+ GB VRAM) is needed for the smoke.
The bundle demotes because tok/s + KL + perplexity are unmeasured. -/
def evQwen3Omni : ConversionEvidence :=
  { roundtripLoadMetal := true,    -- ggml_metal_init printed + first-tok output before OOM
    tokensPerSec       := 0,       -- unmeasured — Metal OOM on 32 GB Mac working set
    klScaled           := 0,       -- unmeasured
    perplexityScaled   := 0,       -- unmeasured
    metricsMeasured    := false,   -- tok/s, KL, perplexity all unmeasured
    quantSanctioned    := true,    -- q8_0 on non-QAT source is a sanctioned PTQ shape
    conversionTool     := "llama.cpp convert_hf_to_gguf.py (Qwen3OmniMoeTextModel path)" }

/-- chibifire/omnigen2-gguf — pending stable-diffusion.cpp support. OmniGen2 is
"any-to-any" diffusion; llama.cpp's converter doesn't cover diffusion
architectures. Bundle stays as demoted placeholder until sd.cpp lands the
arch, if it lands at all. -/
def evOmniGen2_pending : ConversionEvidence :=
  { roundtripLoadMetal := false,   -- not run yet
    tokensPerSec       := 0,
    klScaled           := 0,
    perplexityScaled   := 0,
    metricsMeasured    := false,
    quantSanctioned    := true,    -- fp16 only planned (diffusion)
    conversionTool     := "stable-diffusion.cpp (pending arch support)" }

/-- chibifire/Kimodo-SOMA-RP-v1.1-gguf — Kimodo motion transformer, converted
via `localai-org/kimodo.cpp` (fork `v-sekai-fabric/kimodo.cpp`) at F32.
`kmd-inspect` (Metal-linked build, otool -L shows libggml-metal.0.dylib)
validated the GGUF loads and gguf_get_n_tensors == 414. Full text→motion
generation smoke deferred: requires the LLM2Vec text bundle
(`LocalAI-io/Llama-3-Kimodo-GGML`, ~2 GB). Until that runs, tok/s is
unmeasured and the bundle demotes to `requiresManualReview`. -/
def evKimodoSOMA : ConversionEvidence :=
  { roundtripLoadMetal := true,    -- kmd-inspect load + gguf_get_n_tensors validated on Metal build
    tokensPerSec       := 0,       -- not measured — needs text bundle for kmd-generate
    klScaled           := 0,
    perplexityScaled   := 0,
    metricsMeasured    := false,   -- tok/s, KL, perplexity all unmeasured
    quantSanctioned    := true,    -- F32 is the safest possible (no PTQ)
    conversionTool     := "kimodo.cpp scripts/convert_motion_to_gguf.py (ggml @ 8c63e709)" }

/-- Negative control: a conversion that failed Metal init must NOT ship,
even with fabricated perfect metrics.
Doctrine: "a check that passes on known-broken input is decoration." -/
def evNoMetal : ConversionEvidence :=
  { roundtripLoadMetal := false, tokensPerSec := 100,
    klScaled := 10, perplexityScaled := 1000,
    metricsMeasured := true,
    quantSanctioned := true, conversionTool := "n/a" }

/-- Negative control: a conversion that used an unsanctioned quant must NOT
ship, even if Metal + tokens + PPL all look fine. -/
def evUnsanctionedQ4 : ConversionEvidence :=
  { roundtripLoadMetal := true, tokensPerSec := 30,
    klScaled := 20, perplexityScaled := 1020,
    metricsMeasured := true,
    quantSanctioned := false,  -- q4_0 on a non-QAT source
    conversionTool := "llama.cpp llama-quantize q4_0" }

/-- Negative control (new): a bundle with unmeasured metrics must NOT ship,
even when Metal + tokens + quant-sanction all pass. This is the honesty gate
that keeps evGemma4Qat and evQwen25VL from asserting shipping until KL and
perplexity are actually run. -/
def evUnmeasured : ConversionEvidence :=
  { roundtripLoadMetal := true, tokensPerSec := 30,
    klScaled := 0, perplexityScaled := 0,
    metricsMeasured := false,
    quantSanctioned := true, conversionTool := "measured tok/s only" }

/-! ## Certificates. -/

-- Real Phase 2 measurements — currently demote because KL + perplexity unmeasured.
-- Flip `metricsMeasured := true` on the bundle (and edit klScaled/perplexityScaled
-- with real numbers) to promote to `ships`.
example : verdict evGemma4Qat        = Verdict.requiresManualReview := by decide
example : verdict evQwen25VL         = Verdict.requiresManualReview := by decide

-- Qwen3-Omni — conversion + upload done; Metal init OK but 32 GB Mac hits OOM on the 30B q8_0 working set.
example : verdict evQwen3Omni         = Verdict.requiresManualReview := by decide

-- Pending — conversion not yet run.
example : verdict evOmniGen2_pending  = Verdict.requiresManualReview := by decide

-- Kimodo — Metal load validated, but tok/s + KL + perplexity unmeasured until LLM2Vec text bundle runs.
example : verdict evKimodoSOMA       = Verdict.requiresManualReview := by decide

-- Negative controls.
example : verdict evNoMetal          = Verdict.requiresManualReview := by decide
example : verdict evUnsanctionedQ4   = Verdict.requiresManualReview := by decide
example : verdict evUnmeasured       = Verdict.requiresManualReview := by decide

end PlausibleWitnessDag.PortConversions
