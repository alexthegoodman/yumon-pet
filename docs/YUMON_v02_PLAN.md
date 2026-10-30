# Yumon v0.2 — Model-by-Model Plan (draft for review)

This expands the architecture in `YUMON_v02.md` into a concrete, per-model plan.
Each model maps to what already exists in the repo, what is missing, and what
needs research before we build. Treat this as a living document: revise it, then
split it into issues.

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

**Milestone**
- M0: `SharedState` + memory API + consolidation tick, all CPU, fully unit
  tested. No neural models yet.

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
- Fall back to a **deterministic priority scheduler** first (handwritten rules +
  scores), then replace the score function with the learned controller only when
  behavior proves insufficient. This matches the `infer_reply` philosophy:
  conservative, never invents an action.
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

**Milestones**
- M1: deterministic scheduler + priority queue, wired to one front-end.
- M2: learned action classifier, A/B'd against the scheduler.

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

**Data**
- FER2013, CIFAR-100 (already used). Add ESC-50/UrbanSound8K, a Places365 or
  SUN subset, and synthetic scene+emotion captions from Qwen.

**Research questions**
- Is audio worth the dependency at this stage, or is text+vision enough for
  "companion who entertains"? (Recommend: defer audio, keep the slot in the
  design.)
- Do we need object *detection*, or is classification + the MoE's canvas patch
  tokens sufficient for rich world understanding?

**Milestones**
- M3: unify `VisionModel` + a scene classifier behind a `Perception` trait that
  writes a fixed perception vector.
- M4 (stretch): audio classifier.

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

**Milestones**
- M7: planning-mode schema + constrained generation on the Language model.
- M8 (stretch): dedicated small reasoning head.

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

**Gaps**
- Vocabulary "still has some semantic redundancy" (README) — dedup unfinished.
- No runtime wiring of the decompressor into chat (talk → Kilo → decompress) as
  a single conversational loop, beyond what `chat_ui`/`yumon_portal` do.

**Proposed design**
- **Keep the MoE decoder as the primary talk brain.** Continue the adaptive
  depth/adaptive loss lines; they directly serve "lightweight" goals.
- **Finish vocabulary dedup** so more of the 4,096 budget goes to rich
  distinctions; re-run `build_kilo_vocab` → `cache_yumon_talk` →
  `build_decompressor`.
- Make the **decompressor a first-class part of generation** so Kilo Talk →
  fluent English is one pipeline, and measure BLEU/exact-match (per
  `kilo-compressor.md` FR-4).
- Condition Language on the shared `state_embed` (a token/prefix), so
  conversation reflects world state.

**Data**
- Existing synthetic corpora (`archive/synthetic/*`, distilled chats) + Qwen
  synthetic dialogue seeded with Yumon's personality and current world.

**Research questions**
- Dense vs MoE vs xLSTM for the talk model at these sizes — is MoE actually
  winning, or is dense+KV-cache enough? (There is already timing data to lean on.)
- M-RoPE for multi-turn/role conditioning vs. plain RoPE + a role token.
- Where does the Kilo-decompress step best live (inside the model as learnable
  expansion, or as the statistical `NGramDecompressor`)?

**Milestones**
- M9: vocab dedup + rebuild + decompress-in-loop.
- M10: state-conditioned conversation, A/B vs. current chat.

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

**Milestones**
- M11: learned policy head over the existing enum, replacing `infer_reply` in one
  sim.
- M12: curiosity/novelty drive.

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

**Milestones**
- M13: unified `Executor` trait + validation layer.

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

**Research questions**
- For "ideas," is Generation just Language with a creative prompt, or a distinct
  sampling mode?
- GAN vs AR-MoE for textures/backgrounds — where does DCGAN actually win?

**Milestones**
- M14: unify art pipeline behind a generation facade.
- M15: implement spatial/frequency losses + causal Conv2D edge expert.

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

**Milestones**
- M16: temporal event log in `SharedState`. Video deferred.

---

## Cross-cutting research backlog

- **M-RoPE depth:** how many head-dim pairs to give to (t, y, x) at these
  embeddings (the `new_mrope` split is currently heuristic).
- **Kilo Talk fidelity:** quantify decompression BLEU; decide learnable
  expansion vs. `NGramDecompressor`.
- **Adaptive depth / adaptive loss:** are the experimental wins
  (`adaptive_depth.rs`, `adaptive_loss.rs`) worth folding into the default talk
  and reasoning models?
- **Model-class selection table:** write down a decision rule (LSTM vs
  classification vs MoE vs GAN) and stick to it across subsystems.

## Suggested build order

1. M0 Shared state + memory API (foundation).
2. M1/M2 CNS scheduler, then learned controller.
3. M13 Execution trait (so actions are testable early).
4. M5/M6 Memory, M16 temporal log.
5. M3 Perception unification.
6. M9/M10 Language (vocab + state-conditioning) and M7 Reasoning-as-mode.
7. M11/M12 Behavior, M14/M15 Generation.
8. M4 audio and video deferred.
