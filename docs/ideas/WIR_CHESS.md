# WIR Chess

A chess player whose judgment is a written WIR 0.3 network (attention over the board),
with the host doing everything that is not a decision: legal moves, search, clocks,
bookkeeping. Then RL on top of the written weights. Phases 1-3 built Nov 9 (docs/WIR.md "WIR 0.3 extensions and WIR Chess"); RL not yet.

## Split

- **Network (decisions):** reads the position, writes attack maps and features, and scores
  it (or each candidate move). Every judgment lives here.
- **Host (responsibilities):** legal move generation, making/unmaking moves, search
  (alpha-beta + quiescence), repetition/50-move rules, the page.
- Difficulty is search depth. Depth 0 = the network picks among legal moves by its own
  score with no lookahead: a pure-WIR decision.

## Network

- Tokens: 64 squares (piece one-hot, 13 values) plus a state token (side to move,
  castling, en passant).
- Sliders: per square and per direction (8), a graded head picks the first occupied square
  on the ray (closest wins) and reads its piece. Rook/bishop/queen attacks with blocking.
- Knights, kings, pawns: square-pair tables; keys restricted to a piece type by writing the
  side as `select(is_piece, sq, -1)` (works in today's language).
- Per square: attackers for each side, lowest attacker value.
- Evaluation: material + piece-square tables tapered by phase (a count head), mobility,
  hanging pieces, king-zone attacks, passed pawns (a count head over file and adjacent files
  ahead), maybe bishop pair, rooks on open files.
- Style knobs are weights of the last linear layer, so sliders change them live and exactly.

## Language extensions needed (shared with tactics coordination)

1. Several integer inputs per token.
2. AND by additive scores (each condition adds beta; BOS threshold just under the sum).
3. Graded scores (argmin / closest-first) with id tie-breaks, and the bound for them.

## Verification

- Attack maps and every feature vs a JS/Rust reference in lockstep on millions of positions
  from random and self-play games (0 differences, like Ant).
- The proven error bound per head.
- If move generation moves into the network later: perft on standard positions.
- Strength: written eval vs the same search with a plain-code port of the eval (should be
  identical moves), then against other baselines. No Elo claims before measuring.

## RL on top

- Freeze the written structure (attack heads, features) so it stays exact and verified;
  learn only the readout: style weights and piece-square values, a few hundred to a few
  thousand numbers.
- Method: TD-Leaf(lambda) (Baxter, Tridgell, Weaver 1998, KnightCap): TD on the eval at
  the leaf of the principal variation, so it learns through search. As we recall, KnightCap
  learned its weights from online games and climbed by hundreds of rating points in a
  few hundred games, starting from material values only; to check before citing.
- Volume: self-play headless in Rust (or Node) at shallow depth. Games per hour depend on
  eval speed; measure before planning.
- Human games: each game played on the page logs positions + result. Few games per day is
  enough for personalization (the engine adapts to its opponent), not for raw strength.
- Learned changes decompile back to WIR text, so a training run reads as a diff
  ("knights on the rim -0.3").
- Caution from Yumon Park: learning there changed ~18% of decisions but did not help
  survival. Chess has a clean reward (win/draw/loss), which should help, but compare against
  the frozen written baseline on every run.

## Phases

1. Language extensions (above), with fuzz coverage and the bound.
2. Attack layer + lockstep check vs reference.
3. Evaluation + host search + playable page (depth slider, style sliders, attention rays).
4. Strength baselines.
5. RL: TD-Leaf self-play, then from human games.
