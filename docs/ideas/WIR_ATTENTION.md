# WIR attention: written transformers

Goal: write attention layers into weights the way WIR 0.2 writes ReLU MLPs. A
program reads a token sequence, defines per-position values with the 0.2
combinators, and reads other positions through attention heads. It compiles to a
small transformer with no training.

## Prior art

- RASP (Weiss, Goldberg, Yahav 2021, "Thinking Like Transformers") is a language for
  what a transformer can compute: `select(keys, queries, predicate)` makes a
  selector, `aggregate(selector, values)` averages values over it, and
  `selector_width` counts it.
- Tracr (Lindner et al., DeepMind 2023, "Tracr: Compiled Transformers as a Laboratory
  for Interpretability") compiles RASP to real transformer weights: categorical values
  are one-hot, numerical ones a single dimension, a BOS token is the default attention
  target, and selector width is computed from attention to BOS plus an MLP. Its
  compressed models learned their projection by gradient descent.

We use both as inspiration. What WIR adds:

1. One language with the existing 0.2 combinators, grid programs and decompiled layers.
2. A reference semantics (exact hard attention) next to the compiled network (softmax),
   checked exhaustively or on samples against reference functions, like Pong.
3. Later: superposition of the residual stream without training (the LP approach of
   `superpose-choice-layer`), instead of a learned compression.

## Language sketch (WIR 0.3)

```
wir 0.3
sequence len 8                      # max length; add `causal` for a decoder
input tok: token in [0, 7]          # integer tokens, one-hot + value in the embedding
let n   = count(true)               # selector width: keys where the predicate holds
let opp = n - pos - 1
output rev = mean(tok, k(pos) == q(opp))
```

- `pos` is the position (0-based). `k(e)` is `e` at the key position, `q(e)` at the query.
- Predicates: `true`, `k(a) == q(b)` and `eq/ne/lt/le/gt/ge(side, side)`. Both sides
  must have integer intervals; the compiler one-hot encodes them (free for `tok` and
  `pos`, K+2 relus otherwise) and writes the predicate table into W_Q W_K.
- `mean(v, pred)` averages v over the keys where pred holds, 0 when none.
- `count(pred)` is the number of such keys (attention to BOS gives 1/(w+1); a
  piecewise-linear MLP turns it back into w, exact at the integers).

## Compilation

- Norm-free transformer. Residual stream: `isbos`, `one`, token one-hot + value, position
  one-hot + value, then one dimension per relu node and head. No carry units: the
  residual connection carries every value.
- Constants ride on `one` (0 at BOS), so at BOS every dimension but `isbos` is 0 and
  a head over an empty set returns 0.
- Hard attention through sharpness beta: matching keys score beta, BOS beta/2 (mean)
  or beta (count), the rest 0. Leak onto other keys is about e^-beta/2 per value.
- Blocks are attention then MLP (one hidden layer). A head goes in the first attention
  sublayer after its inputs exist; a relu in the first MLP after its inputs.

## Milestones

1. Primitives: reverse, histogram, sort (exhaustive over small vocab/length). Done Nov 9,
   see docs/WIR.md "WIR 0.3: written attention".
2. More primitives: previous-token and induction heads, Dyck, most frequent token,
   causal programs, PyTorch float32 export.
3. Yumon intent/emote classifier on 100-char snippets, then AG News as a public number.
4. Superposed residual stream without training.

## Risks

- Norms: RMSNorm rescales the residual stream. We start norm-free; fitting sunrise's
  architecture (norms, RoPE, Burn, KV cache) is its own milestone.
- f32 softmax with large beta is fine with max subtraction; still to be measured.
- Interval soundness: the reference uses exact hard attention; the network differs by
  the leak. Bounding the leak through later relus is not done yet; checks are
  empirical.
