# Yumon v0.2 — Model-by-Model Plan (draft for review)

This expands the architecture in `YUMON_v02.md` into a concrete, per-model plan.

> Note: most model *code* already exists. The bulk of the work is **synthetic
> data** (new `gen_synthetic_*` Rust scripts) and **verification** — a small
> validation script we build only as far as each model needs to be tested objectively.

## High-level tasks

1. **Shared state + memory API** — `SharedState`, episode store, consolidation
   tick.
2. **`gen_synthetic` script scaffold** — one reusable Qwen-endpoint harness
   (seeded prompts, resume, Kilo-Talk output, held-out split) so each model's
   data script is a thin wrapper.
3. **CNS** — a learned action classifier.
4. **Execution** — unified `Executor` trait + validation layer.
5. **Perception** — unify `VisionModel` + scene classifier behind a `Perception`
   trait. We need to expand our vision identification capability, likely through
   new image/prompt pairs beyond the old CIFAR. We will use our DiffusionDb, and potentially new synethetic,
   as the DiffusionDb can be a bit more creative than our simulation world will be.
6. **Memory** — episode store + retrieval, then salience + consolidation.
7. **Temporal** — event log in `SharedState`.
8. **Language** — Mostly already complete, may need some tuning, mimics "conversation"
9. **Reasoning** — Can be a distinct model, mimics "thought"
10. **Behavior** — A learned policy that makes room for emergent choices in new situations
11. **Generation** — TBD (this is deferred for now)
12. **Verification sim** — a minimal world (2 Yumon, a few toys, identifiable
    objects) to score each model's results as they land.

*Deferred:* audio, video (event tracking covers motion for v0.2).

Ground rules from the vision doc that every model must respect:

- **Small models orchestrated, not one big model.** No single model carries broad
  knowledge or problem solving.
- **RoPE / M-RoPE** is the shared position primitive (already implemented for
  dense, MoE and multimodal sequences — `RopeTables::new_mrope` in
  `src/brain/decoder_model.rs`).
- **Kilo Talk** is the compressed English substrate for anything that emits text,
  so embedding dims stay tiny (4,096-token `KiloVocab` for talk, 1,024-token Kilo
  for code).
- **Model-class rule of thumb:** LSTM / number-mapped / classification for cheap
  stateful decisions; GAN for images; MoE for text+.
- **Data:** free Kaggle/HF datasets first, Qwen-via-HF-Inference-Endpoints
  synthetic data as the primary source, seeded with handcrafted/real data.
- **No fall back behaviors or placeholders** this creates false behavior in the final result.

---

## 0. Shared State & Multimodal Memory (build this first)

The "common context, persistent memory, episodic history, active goals, system
variables" block. Everything else reads and writes it, so its shape is the
contract that makes the whole system cohesive.

**Current assets**
- `src/brain/samples.rs`: `WorldContext`, `TrainingStage`, `Action`,
  `CardinalDir`, `GenerationResult` — scattered per-domain state.
- `src/universe.rs` `Place`/`Neighborhood` and `src/kingdom.rs` `Realm` hold
  world state but nothing unifies them.
- `Citizen.memory` (a `String`) and `Realm.events` are ad-hoc memory, not a
  store.

**Gaps**
- No single shared-state struct with a stable vector + structured view.
- No retrieval API, no consolidation loop, no goal stack.

**Proposed design**
1. Define `SharedState` with two parallel views:
   - **Structured:** goals (priority queue), system variables, recent events,
     active perception snapshot, current behavior/plan, working memory slots.
   - **Embedded:** a small `state_embed` (e.g. 128–256 dims) produced by a
     lightweight encoder so downstream models can condition on "where we are"
     without parsing JSON.
2. A **Memory subsystem** (see §3) owns the *episodic* part; `SharedState` holds
   only what is active/current.
3. A **consolidation tick** (driven by the CNS) periodically summarizes stale
   episodes into higher-level memories and prunes. Summarization reuses the
   Language model (§5) in Kilo Talk.

**Inputs/outputs**: every subsystem declares a read set and a write set; the CNS
enforces the order. This is the one place where determinism and testability are
hard requirements (mirror the `infer_reply`/`Realm` test style already in
`universe.rs`/`kingdom.rs`).

**Research questions**
- How much of `state_embed` should be learned vs. hand-structured? (Start
  structured, add a learned projector only when needed.)
- Episodic vs semantic vs working memory split — which is worth paying for at
  this scale?
- Consolidation policy: time-based, importance-based, or both?

---

## 1. Central Nervous Model (CNS) — orchestrator

Responsibilities: global awareness, state management, prioritization,
coordination, decision making.

**Current assets**
- Nothing explicit. Orchestration today lives in bespoke front-ends
  (`yumon_portal.rs`, `yumon_universe.rs`, `yumon_world.rs`, `yumon_paint.rs`,
  `yumon_civ.rs`) and in the rule-based `infer_reply` bridge (`universe.rs`).
- `TrainingStage` in `samples.rs` shows multi-stage handoff exists in training,
  not at runtime.

**Gaps**
- No runtime scheduler that decides *which* subsystem acts, with what priority,
  and with what budget.

**Proposed design**
- A **number-mapped / classification controller**, not a generative model:
  - Input: `state_embed` + a one-hot "last actor/result" + resource budget.
  - Output: `(subsystem, priority, params)` — a small categorical head.
- No fall backs
- Text it emits (rare, e.g. "what should I do?") goes through Kilo Talk.

**Data**
- Synthetic decision traces from Qwen: state snapshot → "Yumon should (play |
  talk | explore | rest) because …", distilled to the chosen action only.
- Handcrafted rule seeds as ground truth.

**Research questions**
- Should the CNS be a single classifier or a tiny LSTM with a hidden recurrent
  state (so it can "remember" it just interrupted a task)?
- Prioritization semantics: hard preemption, aging scores, or a cost/benefit
  scorer?
- Is the CNS even neural, or should it stay a rules engine? (Recommend: rules
  first, learn later.)

---

## 2. Perception

Responsibilities: vision, audio, scene recognition.

**Current assets**
- Vision: `src/vision/model.rs` (`VisionModel` CNN: CIFAR-100 100-class head +
  FER2013 7-class emote head), and `VisionPatchEncoder` (canvas → 256 patch
  tokens for the MoE).
- The `CONTEXT_DIMS` comment in `src/brain/mod.rs` shows the intended layout:
  `class_probs + emote_probs` concatenated — perception was designed to feed the
  brain as probability vectors.
- No audio at all (the `speech` field in `yumon_world.rs` is a speech *bubble*,
  not audio).

**Gaps**
- Audio input (sound/voice/ambience) is absent.
- Scene recognition beyond 100 CIFAR classes and raw pixels.

**Proposed design**
- Keep the small-CNN pattern; add **one audio classifier** (e.g. ESC-50 /
  UrbanSound8K-style sound events, plus a lightweight wake-word/speech detector).
- Add a **scene/room classifier** (multi-label, e.g. a Places365 subset) for
  "global awareness" of *where* Yumon is.
- Every perception head emits a **fixed-size probability vector** into
  `SharedState` (reusing the existing `class_probs`/`emote_probs` idea). No raw
  pixels/samples cross into the CNS.
- Explore M-RoPE for a *sequence* of perception observations (time-stamped scene
  snapshots), giving the memory/reasoning layers positional structure.
- Of course, input images will be resized to a reasonable size for inference

**Data**
- FER2013, CIFAR-100 (already used). Add ESC-50/UrbanSound8K, a Places365 or
  SUN subset, and synthetic scene+emotion captions from Qwen.

**Research questions**
- Is audio worth the dependency at this stage, or is text+vision enough for
  "companion who entertains"? (Recommend: defer audio, keep the slot in the
  design.)
- Do we need object *detection*, or is classification + the MoE's canvas patch
  tokens sufficient for rich world understanding?

---

## 3. Memory

Responsibilities: encoding, retrieval, consolidation.

**Current assets**
- `Citizen.memory: String`, `Realm.events: Vec<String>` (`kingdom.rs`).
- `Memory` / `HandcraftedChats` / `ChatBlock` in `src/brain/mdx.rs` (conversation
  history as training data, not runtime memory).
- `am_distill.rs` streams a distilled reasoning dataset into short
  `<user, assistant>` memories — evidence the author already treats memory as
  short, compressed utterances.

**Gaps**
- No learned encoder/retriever; no consolidation; memory is per-domain strings.

**Proposed design**
- **Encoding:** compress an episode into a short Kilo-Talk sentence (the
  Language model already does short replies; reuse it). Store it keyed by a
  lightweight embedding for retrieval.
- **Retrieval:** given current `state_embed`/query, return top-k memories. Start
  with exact/keyword + cosine over a small static embedder; upgrade to a learned
  retriever only if recall is the bottleneck.
- **Consolidation:** periodic summarization (episode → day → trait), a la the
  "summarization and consolidation capabilities" line in the vision doc.
- **Salience:** an LSTM or number-mapped scorer decides what is worth keeping
  (novelty + emotional weight + goal relevance).

**Data**
- Synthetic episode logs from Qwen; handcrafted "this is worth remembering"
  labels for the salience model.

**Research questions**
- Store embeddings explicitly, or rely on keyword + recentness first?
- Where does consolidation live — Memory or the CNS? (Recommend: Memory owns the
  policy, CNS schedules it.)
- How big should the persistent store grow before summarization must kick in?

**Milestones**
- M5: append-only episode store + keyword/cosine retrieval.
- M6: learned salience + consolidation summarizer.

---

## 4. Reasoning

Responsibilities: interpretation, planning, evaluation.

**Current assets**
- `infer_reply` (`universe.rs`) — conservative rule-based interpretation.
- `fixer.rs` / JSON-schema extraction in `model.rs` — structured-output repair.
- The Code MoE (`YumonMoeBrain`, `moe_model.rs`) is the closest thing to a
  "text+" reasoner, with adaptive depth and shared experts already in place.

**Gaps**
- No explicit planning/evaluation module; reasoning is implicit in generation.

**Proposed design**
- A **small MoE (or the existing MoE text brain with a planning head)** that
  consumes Kilo Talk and emits a short structured plan
  (`{goal, steps[], evaluation}`). Reuse the JSON schema + `fixer` machinery.
- Keep plans tiny (≤ a handful of steps) — consistent with the "no broad problem
  solving" constraint.
- **Evaluation** can be a cheap classification head (plan → good/bad) or a
  scoring pass, trained on synthetic outcome data.

**Data**
- Synthetic plan/evaluate traces from Qwen; puzzle/step datasets where
  lightweight (e.g. small strategy sets).

**Research questions**
- Is a separate Reasoning model worth it, or should planning be a *mode* of the
  Language model (a prompt/schema) to avoid yet another checkpoint?
- Structured plans via constrained decoding (the `outlines_core` import in
  `model.rs`) vs. the current `fix_json_syntax` repair path?

---

## 5. Language

Responsibilities: conversation and communication.

**Current assets (the most mature subsystem)**
- Architectures: `YumonDecBrain` (dense decoder, RoPE + FlashAttention + KV
  cache + GQA), `YumonMoeBrain` (sparse MoE, M-RoPE, adaptive depth, shared
  experts), `YumonXLstmBrain` (mLSTM), `reference_model.rs` (baseline).
- Compression: `KiloVocab` (4,096 talk tokens), Kilo (1,024 code tokens),
  `NGramDecompressor` (`decompressor.rs`), BPE (`bpe.rs`), legacy `tokenizer.rs`.
- Training infra: `train.rs`, `train_code.rs`, `train_bpe.rs`, curriculum and
  adaptive-loss/depth machinery, `sentiment.rs`, `chats.rs`, `keywords.rs`.

**Proposed design**
- **Keep the MoE decoder as the primary talk brain.** Continue the adaptive
  depth/adaptive loss lines; they directly serve "lightweight" goals.
- Condition Language on the shared `state_embed` (a token/prefix), so
  conversation reflects world state.

**Data**
- Existing synthetic corpora (`archive/synthetic/*`, distilled chats) + Qwen
  synthetic dialogue seeded with Yumon's personality and current world.

---

## 6. Behavior

Responsibilities: play, social, discovery, invention.

**Current assets**
- Behavior taxonomy + compatibility rules in `universe.rs` (`Behavior` enum:
  Rest/Visit/Play/Collect/Tend/Inspect/Explore/Build; `verb_and_behavior`,
  `infer_reply`).
- `kingdom.rs` citizen/project behaviors; `paint_action.rs` discrete actions.

**Gaps**
- Behavior selection is rule-based (`infer_reply`) or hardcoded in front-ends;
  no learned policy, no notion of "discovery"/"invention" driven by curiosity.

**Proposed design**
- A **small LSTM policy** over `state_embed` + recent events → next `Behavior`
  (a classification head over the existing enum). Start with the `infer_reply`
  rules as labels, then let it generalize to unseen situations.
- Add **curiosity/discovery** as an intrinsic drive: a number-mapped score of
  "how novel is this place/object" that biases exploration.
- Keep all physical effects grounded through the same `Behavior` enum so the
  world (`universe.rs`) and the policy share one contract.

**Data**
- Synthetic behavior traces from Qwen ("given this scene, Yumon plays with the
  ball"), handcrafted seeds, and logs from existing sims as replay data.

**Research questions**
- LSTM policy vs. classification-per-tick vs. a small MoE — how much temporal
  context does behavior actually need?
- Is "invention" just Behavior::Build + a Generation/Language output, or a
  distinct mechanism?

---

## 7. Execution

Responsibilities: tools, code, physical actions.

**Current assets**
- Yumon Code (the MoE/dense code model) + `yumon_nano.rs` editor completions.
- Physical actions: `universe.rs` (travel/approach/`Behavior`), `kingdom.rs`
  (buy/fund/trade), `paint_action.rs` (paint commands).
- Structured action schema already exists in `model.rs` (`YUMON_SCHEMA`,
  `YUMON_SCHEMA_INPUT`, `Action` enum, JSON repair).

**Gaps**
- No unified "execute an action" API across worlds; tool/code execution is
  separate from physical actions and not driven by shared state.

**Proposed design**
- A single **`Executor` trait** (`execute(action, world_state) -> outcome`) with
  implementations per world (suburb, kingdom, paint, code/tool).
- The Language/Reasoning layers emit structured actions (`Action` / a new
  `ToolCall`); Execution validates and applies them conservatively (never
  invents an invalid action — same guarantee as `infer_reply`).
- Code/tool output stays on the Yumon Code model, surfaced through `yumon_nano`
  or a headless tool executor.

**Data**
- Synthetic tool-use traces (Qwen); existing code corpus for the code model.

**Research questions**
- Do we need a separate Execution *model*, or is Execution a deterministic
  interpreter whose "brain" is Reasoning/Language? (Recommend: interpreter, not
  model.)
- Sandboxing/safety for tool calls on a local desktop companion.

---

## 8. Generation

Responsibilities: images, ideas, creative output.

**Current assets (the second-most mature subsystem)**
- Cascaded AR MoE pixel generator: `moe_model.rs` multimodal path,
  `paint_action.rs` (passes, palette, spatial weights, `FaceRoi`),
  `yumon_paint.rs`, `train_art_moe.rs`.
- `dcgan_model.rs` (vanilla DCGAN), `upscaler.rs` (→256), `colorizer.rs`
  (Zhang 2016), `paint_studio.rs`.
- Design docs: `MULTIMODAL_ART_MOE.md`, `SPATIAL_WEIGHTS.md`, `CONV2D_EDGES.md`.

**Gaps**
- Ideas (text creative output) not unified with image output; no single
  "generate" entry point.
- Spatial/frequency losses (`SPATIAL_WEIGHTS.md`) and causal Conv2D edge mixing
  (`CONV2D_EDGES.md`) are designed but not all implemented.

**Proposed design**
- Keep the AR MoE as the primary image path; adopt the **spatial/frequency
  losses** and **causal Conv2D spatial expert** ideas from the design docs to
  sharpen edges at tiny resolution.
- Route **ideas/creative text** through Language + Reasoning (not a new model).
- A single `generate(kind, seed) -> artifact` facade so the CNS can call it
  uniformly.

**Data**
- DiffusionDB / animals (already used), upscaled to the colorizer/upscaler
  pipeline; Qwen synthetic prompt→image traces.

---

## 9. Temporal

Responsibilities: video, motion, event tracking.

**Current assets**
- M-RoPE already models a (t, y, x) trajectory axis (`RopeTables::new_mrope`,
  `extract_batch_sequence_coords`), and the cascaded paint passes are temporal.
- `Realm.advance(dt)` season loop (`kingdom.rs`) is a discrete-time event system.
- No video/motion model.

**Gaps**
- No event tracking across subsystems; no motion/video generation.

**Proposed design**
- **Event tracking** = a temporal log in `SharedState` (ordered, decaying) that
  Memory consumes — lightweight, deterministic, no model.
- **Motion** for digital worlds = the existing `advance`/position updates; a
  small number-mapped or LSTM model could *predict* motion intent, but likely
  unnecessary.
- **Video** = lowest priority; if ever, start from the AR-MoE temporal axis
  (frame = canvas at time t) or a GAN next-frame model. Defer.

**Data**
- n/a for now (event tracking is deterministic).

**Research questions**
- Does the companion need actual video/motion, or is discrete event tracking +
  the paint-time axis sufficient? (Recommend: event tracking only for v0.2.)
- Should M-RoPE's temporal axis be reused for memory ordering?