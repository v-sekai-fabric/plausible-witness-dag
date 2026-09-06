import PlausibleWitnessDag

/-! # ggml-fork witness

Domain package: certify whether each of four GGUF fork candidates for
chibifire/* should be forked or held for manual review, using the generic
iterative-deepening driver from `PlausibleWitnessDag`.

Candidates from the fork-#129 audit (2026-09-05):

* `chibifire/qwen3-omni`                     → `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF`
* `chibifire/gemma-4-12B-it-qat-q4_0-*`      → `google/gemma-4-12B-it-qat-q4_0-gguf`
* `chibifire/Qwen2.5-VL-7B-Instruct`         → `unsloth/Qwen2.5-VL-7B-Instruct-GGUF`
* `chibifire/omnigen2-base-df5dca8a`         → `calcuis/omnigen2-gguf`
-/

namespace PlausibleWitnessDag.GgmlForks

open PlausibleWitnessDag

/-- Evidence captured from the HF model card + API for one fork candidate.

Fields are read once (out of band) and stored so the witness search is a pure
function of them. -/
structure Evidence where
  downloads                    : Nat
  likes                        : Nat
  hasPermissiveLicense         : Bool  -- Apache-2.0, MIT, NVIDIA OML, or similar
  hasBaseModelMatchingOurs     : Bool  -- cardData.base_model or README names our repo
  hasGgufFiles                 : Bool  -- file listing contains at least one .gguf
  deriving Repr, Inhabited

/-- Score maps evidence into `[0, 400]` — a Fin-bounded window suitable for
plausible's `Fin 4096` candidate search. Caps on `downloads` and `likes` prevent
outliers from dominating.

Weighting rationale:

* GGUF file presence is the strongest signal (100pt) — without it the fork
  is meaningless for our ggml consumers.
* Permissive license (50pt) and base-model match (50pt) each independently
  matter for shipping.
* Community adoption (`downloads`, `likes`) contributes up to 200pt combined. -/
def score (e : Evidence) : Nat :=
  (if e.hasGgufFiles then 100 else 0) +
  (if e.hasPermissiveLicense then 50 else 0) +
  (if e.hasBaseModelMatchingOurs then 50 else 0) +
  (min e.likes 500 / 5) +          -- 0..100
  (min e.downloads 100000 / 1000)  -- 0..100

/-- Verdict on a candidate: fork straight through, or route to a human. -/
inductive Verdict
  | justified
  | requiresManualReview
  deriving Repr, DecidableEq, Inhabited

/-- Threshold: 150 out of a theoretical 400 max is enough to fork. The three
strongest signals (GGUF present + permissive license + base-model match) alone
sum to 200, so any well-formed upstream with the correct base model clears the
bar without needing high adoption. -/
def threshold : Nat := 150

/-- Verdict has TWO gates: (1) GGUF files must be present — a fork of a
non-GGUF repo is meaningless for our ggml consumers, no amount of community
adoption compensates; (2) evidence score must clear the threshold. Both must
hold. -/
def verdict (e : Evidence) : Verdict :=
  if e.hasGgufFiles && score e ≥ threshold then
    Verdict.justified
  else
    Verdict.requiresManualReview

/-- One fork candidate = our repo + upstream + collected evidence. -/
structure Candidate where
  ourRepo   : String
  upstream  : String
  evidence  : Evidence
  deriving Repr, Inhabited

/-- Evidence table for the four candidates from the #129 audit. Numeric fields
match the values HERD's audit reported; the boolean signals are read from the
HF API + our own repo state at audit time. -/
def candidates : Array Candidate := #[
  { ourRepo := "chibifire/qwen3-omni",
    upstream := "ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF",
    evidence := { downloads := 45000, likes := 18,
                  hasPermissiveLicense := true,      -- Apache-2.0 upstream Qwen
                  hasBaseModelMatchingOurs := true,  -- Qwen3-Omni family
                  hasGgufFiles := true } },
  { ourRepo := "chibifire/gemma-4-12B-it-qat-q4_0-gguf",
    upstream := "google/gemma-4-12B-it-qat-q4_0-gguf",
    evidence := { downloads := 760000, likes := 287,
                  hasPermissiveLicense := true,      -- Gemma Terms of Use, permissive commercial
                  hasBaseModelMatchingOurs := true,  -- gemma-4-12B-it-qat
                  hasGgufFiles := true } },
  { ourRepo := "chibifire/Qwen2.5-VL-7B-Instruct",
    upstream := "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
    evidence := { downloads := 154000, likes := 219,
                  hasPermissiveLicense := true,      -- Apache-2.0
                  hasBaseModelMatchingOurs := true,  -- Qwen2.5-VL-7B-Instruct
                  hasGgufFiles := true } },
  { ourRepo := "chibifire/omnigen2-base-df5dca8a",
    upstream := "calcuis/omnigen2-gguf",
    evidence := { downloads := 1300, likes := 38,
                  hasPermissiveLicense := true,      -- Apache-2.0 upstream OmniGen2
                  hasBaseModelMatchingOurs := true,  -- OmniGen2
                  hasGgufFiles := true } }
]

/-- Deterministic walk: given a step budget, recover the verdict of the
candidate at position `idx` in the table. The walk is trivial (constant-time
table lookup) because the audit already ran; the plausible layer is here to
gate the *decision* on the Fin-bounded score window, not to search over an
open-ended candidate space. -/
def walkVerdict (idx : Nat) (_steps : Nat) : Readback Verdict :=
  match candidates[idx]? with
  | some c =>
      { value := verdict c.evidence,
        found := verdict c.evidence == Verdict.justified,
        witnessIdx := score c.evidence,
        budgetHit := false }
  | none =>
      { value := Verdict.requiresManualReview,
        found := false, budgetHit := true }

/-- Plausible-facing candidate predicate: a candidate `k` is a witness iff `k`
equals the evidence's score AND that score clears the threshold. plausible
searches `Fin lvl.finBound` for such a `k`; on success the fork is certified. -/
def scoreCandidate (idx : Nat) (_lvl : Level) (candidate : Nat) : Bool :=
  match candidates[idx]? with
  | some c => candidate == score c.evidence && score c.evidence ≥ threshold
  | none => false

/-- Resolve one candidate through the standard ladder. -/
def certifyCandidate (idx : Nat) : IO (Verdict × Nat × TraceEntry) := do
  let name := match candidates[idx]? with
              | some c => s!"fork {c.upstream} → {c.ourRepo}"
              | none => s!"idx {idx} out of range"
  resolve name (scoreCandidate idx) (walkVerdict idx)

/-- Certify all four candidates. -/
def certifyAll : IO Unit := do
  IO.println s!"Threshold: {threshold} / 400 max"
  IO.println ""
  for i in [:candidates.size] do
    let (v, lvl, trace) ← certifyCandidate i
    let c := candidates[i]!
    let s := score c.evidence
    IO.println s!"[L{lvl}] {c.upstream}"
    IO.println s!"       → {c.ourRepo}"
    IO.println s!"       score={s}  verdict={repr v}"
    IO.println s!"       trace={repr trace}"
    IO.println ""

/-! ## Compile-time certificates for the four current candidates.

Each `example` invokes `decide` on the concrete `verdict` output, so the Lean
elaborator refuses the file if any listed candidate would fall below
`threshold`. These are the plausible-Fin-bounded witnesses reified as
proof obligations that `lake build` will refuse to leave open.

Evidence bundles are inlined as named constants because `Array.get!` doesn't
reduce under kernel `decide`; we duplicate the literal values here and rely on
the runtime `candidates` table for the executable ladder. The two-write cost
buys compile-time certification. -/
def evQwen3Omni : Evidence :=
  { downloads := 45000,  likes := 18,
    hasPermissiveLicense := true, hasBaseModelMatchingOurs := true,
    hasGgufFiles := true }
def evGemma4Qat : Evidence :=
  { downloads := 760000, likes := 287,
    hasPermissiveLicense := true, hasBaseModelMatchingOurs := true,
    hasGgufFiles := true }
def evQwen25VL : Evidence :=
  { downloads := 154000, likes := 219,
    hasPermissiveLicense := true, hasBaseModelMatchingOurs := true,
    hasGgufFiles := true }
def evOmniGen2 : Evidence :=
  { downloads := 1300,   likes := 38,
    hasPermissiveLicense := true, hasBaseModelMatchingOurs := true,
    hasGgufFiles := true }

example : verdict evQwen3Omni = Verdict.justified := by decide
example : verdict evGemma4Qat = Verdict.justified := by decide
example : verdict evQwen25VL  = Verdict.justified := by decide
example : verdict evOmniGen2  = Verdict.justified := by decide

/-- Threshold isn't a rubber stamp: an evidence bundle with no GGUF files must
be rejected. Kept as a negative control (per workspace doctrine: "a check that
passes on known-broken input is decoration"). -/
example :
    verdict { downloads := 999999, likes := 999, hasPermissiveLicense := true,
              hasBaseModelMatchingOurs := true, hasGgufFiles := false }
      = Verdict.requiresManualReview := by decide

end PlausibleWitnessDag.GgmlForks
