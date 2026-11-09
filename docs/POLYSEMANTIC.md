# Polysemantic Notes and Ideas

Here is your complete `wir 0.2` ant brain refactored into a pure **ReLU neural network**—zero `select()` statements, zero `and()`/`or()`/`not()` gates, and zero multiplicative routing (`task * grade`).

Instead, every conditional, logical gate, and sensor router is executed through **inhibitory bias shifts** (`relu(signal - penalty_for_inactive_context)`) and **polysemantic hidden units** that serve double duty across actions, drives, emotes, and directional steering.

### Polysemantic `wir 0.2` Ant Brain

```text
wir 0.2
# Polysemantic ReLU Ant Brain: zero select(), zero multiplicative routing (task * grade),
# and zero explicit boolean gates. Every gate, drive, emote, and sensor router is
# computed via shared linear combinations + ReLU bias-clamping.

input o0, o1, o2, o3, o4: R1 in [0, 1]
input cf: R1 in [0, 1]
input cs: R1 in [0, 1]
input gr: R1 in [0, 1]
input hunger: R1 in [0, 15]
input old: R1 in [0, 1]
input store: R1 in [0, 3]
input w0, w1, w2, w3: R1 in [0, 1]
input r0, r1, r2: R1 in [0, 1]
input gn0, gn1, gn2, gn3, gf0, gf1, gf2, gf3, gg0, gg1, gg2, gg3, gm0, gm1, gm2, gm3, gh0, gh1, gh2, gh3: R1 in [-1, 1]
input b0, b1, b2, b3: R1 in [0, 1]
input fg0, fg1, fg2, fg3: R1 in [-1, 1]
input zn, zm, kn, kf, kg, kh, lf, lp, ld: R1 in [0, 1]
input fs: R1 in [0, 3]
input dns: R1 in [0, 120]

# --- Layer 1: Shared Polysemantic State & Metabolic Units ---
# Instead of isolated boolean gates, each ReLU unit combines multiple state checks.
let free = relu(1 - cf - cs - gr)
let hungry = relu(hunger - 5) - relu(hunger - 15)
let hung4 = relu(hunger - 5) - relu(hunger - 9)
let starving = relu(hunger - 11) - relu(hunger - 12)
let scared = relu(fs) - relu(fs - 1)
let hs = relu(store) - relu(store - 1)
let far_free = relu(dns - 59) - relu(dns - 60) - relu(cf + cs + gr)
let idle_lost = 1 - relu(kn - free + kf + kg + kh)

# --- Layer 2: Direct ReLU Action Units (Bias-Shifted Inhibition) ---
# A single linear sum + threshold replaces nested and()/or()/not() trees.
output pick = relu(free + lf - scared - 1)
output dig2 = relu(free + ld - starving - scared - 1)
output grip = relu(gr + relu(free + lp - 1) - starving - scared)
output drop = relu(cf + zn - 1) + relu(cs + zm - 1) - relu(cf + zn + cs + zm - 3)
let eat_nest = relu(zn + hs + relu(hunger - 5) - relu(hunger - 6) - 2)
output eat = relu(eat_nest + relu(cf + starving - 1)) - relu(eat_nest + relu(cf + starving - 1) - 1)
let work = relu(gr + pick + dig2 + grip + drop + eat) - relu(gr + pick + dig2 + grip + drop + eat - 1)

# --- Layer 3: Polysemantic Drive-Emote Bottleneck ---
# These units simultaneously encode task drives AND emotional states, using large
# negative inhibitory weights (-20) to silence pathways when preconditions fail.
let h_home_eat = relu(2*hung4 - 8*(1 - hs))
let h_far = relu(4*far_free)
let home = relu(6*cf + h_home_eat + h_far - 20*(1 - kn))

let h_forage_base = relu(5 - store + 2*old + 3*r1 - 10*(2 - free - kf))
let h_forage_emp = relu(hung4 - 10*(2 - free - kf + hs))
let forage = h_forage_base + h_forage_emp

# h_dig_act serves BOTH the dig task drive AND the 'working' emote (em2)
let h_dig_act = relu(4 - 2*old + 3*r0 - 10*(2 - free - kg + starving))
let dump = relu(6*cs + 6*kn - 6)
let help = relu(3 + 3*r2 - 10*(2 - free - kh + starving))
let explore = 3*idle_lost
let flee = 8*scared

# --- Layer 4: Task Winner-Take-All (ReLU Step Comparators) ---
# Because tkey differences are integers, relu(A - B) - relu(A - B - 1) is an exact > gate.
let tkey0 = 7*home + 6
let tkey1 = 7*forage + 5
let tkey2 = 7*h_dig_act + 4
let tkey3 = 7*dump + 3
let tkey4 = 7*help + 2
let tkey5 = 7*explore + 1
let tkey6 = 7*flee + 0

let tw0_1 = relu(tkey0 - tkey1) - relu(tkey0 - tkey1 - 1)
let tw0_2 = relu(tkey0 - tkey2) - relu(tkey0 - tkey2 - 1)
let tw0_3 = relu(tkey0 - tkey3) - relu(tkey0 - tkey3 - 1)
let tw0_4 = relu(tkey0 - tkey4) - relu(tkey0 - tkey4 - 1)
let tw0_5 = relu(tkey0 - tkey5) - relu(tkey0 - tkey5 - 1)
let tw0_6 = relu(tkey0 - tkey6) - relu(tkey0 - tkey6 - 1)
let tw1_2 = relu(tkey1 - tkey2) - relu(tkey1 - tkey2 - 1)
let tw1_3 = relu(tkey1 - tkey3) - relu(tkey1 - tkey3 - 1)
let tw1_4 = relu(tkey1 - tkey4) - relu(tkey1 - tkey4 - 1)
let tw1_5 = relu(tkey1 - tkey5) - relu(tkey1 - tkey5 - 1)
let tw1_6 = relu(tkey1 - tkey6) - relu(tkey1 - tkey6 - 1)
let tw2_3 = relu(tkey2 - tkey3) - relu(tkey2 - tkey3 - 1)
let tw2_4 = relu(tkey2 - tkey4) - relu(tkey2 - tkey4 - 1)
let tw2_5 = relu(tkey2 - tkey5) - relu(tkey2 - tkey5 - 1)
let tw2_6 = relu(tkey2 - tkey6) - relu(tkey2 - tkey6 - 1)
let tw3_4 = relu(tkey3 - tkey4) - relu(tkey3 - tkey4 - 1)
let tw3_5 = relu(tkey3 - tkey5) - relu(tkey3 - tkey5 - 1)
let tw3_6 = relu(tkey3 - tkey6) - relu(tkey3 - tkey6 - 1)
let tw4_5 = relu(tkey4 - tkey5) - relu(tkey4 - tkey5 - 1)
let tw4_6 = relu(tkey4 - tkey6) - relu(tkey4 - tkey6 - 1)
let tw5_6 = relu(tkey5 - tkey6) - relu(tkey5 - tkey6 - 1)

let task0 = relu(tw0_1 + tw0_2 + tw0_3 + tw0_4 + tw0_5 + tw0_6 - 5)
let task1 = relu(1 - tw0_1 + tw1_2 + tw1_3 + tw1_4 + tw1_5 + tw1_6 - 5)
let task2 = relu(2 - tw0_2 - tw1_2 + tw2_3 + tw2_4 + tw2_5 + tw2_6 - 5)
let task3 = relu(3 - tw0_3 - tw1_3 - tw2_3 + tw3_4 + tw3_5 + tw3_6 - 5)
let task4 = relu(4 - tw0_4 - tw1_4 - tw2_4 - tw3_4 + tw4_5 + tw4_6 - 5)
let task5 = relu(5 - tw0_5 - tw1_5 - tw2_5 - tw3_5 - tw4_5 + tw5_6 - 5)
let task6 = relu(6 - tw0_6 - tw1_6 - tw2_6 - tw3_6 - tw4_6 - tw5_6 - 5)

# --- Layer 5: Additive Bias-Gated Sensor Routing (Zero Multiplication) ---
# Instead of task0*gn0, each sensor channel is shifted by +2*task - 1.
# When task == 0, gn_i - 1 <= 0 (clamped to 0 by ReLU).
# When task == 1, gn_i + 1 >= 0, passing the exact signed gradient via (relu(g + 2*t - 1) - t).
let k0 = relu(2*o0 - 2*o2 - 15*b0 + 17) - 17
let k1 = relu(2*o1 - 2*o3 - 15*b1 + 17) - 17
let k2 = relu(-2*o0 + 2*o2 - 15*b2 + 17) - 17
let k3 = relu(-2*o1 + 2*o3 - 15*b3 + 17) - 17
let k4 = relu(o4)

let g_east = relu(gn0 + 2*task0 - 1) + relu(gf0 + 2*task1 - 1) + relu(gg0 + 2*task2 - 1) + relu(gm0 + 2*task3 - 1) + relu(gh0 + 2*task4 - 1) + relu(w0 + 2*task5 - 1) + relu(fg0 + 2*task6 - 1) - 1
let g_north = relu(gn1 + 2*task0 - 1) + relu(gf1 + 2*task1 - 1) + relu(gg1 + 2*task2 - 1) + relu(gm1 + 2*task3 - 1) + relu(gh1 + 2*task4 - 1) + relu(w1 + 2*task5 - 1) + relu(fg1 + 2*task6 - 1) - 1
let g_west = relu(gn2 + 2*task0 - 1) + relu(gf2 + 2*task1 - 1) + relu(gg2 + 2*task2 - 1) + relu(gm2 + 2*task3 - 1) + relu(gh2 + 2*task4 - 1) + relu(w2 + 2*task5 - 1) + relu(fg2 + 2*task6 - 1) - 1
let g_south = relu(gn3 + 2*task0 - 1) + relu(gf3 + 2*task1 - 1) + relu(gg3 + 2*task2 - 1) + relu(gm3 + 2*task3 - 1) + relu(gh3 + 2*task4 - 1) + relu(w3 + 2*task5 - 1) + relu(fg3 + 2*task6 - 1) - 1

let s0 = 4*g_east + k0
let s1 = 4*g_north + k1
let s2 = 4*g_west + k2
let s3 = 4*g_south + k3
let s4 = 7*work + k4

let key0 = 5*relu(s0 + 21) - 101
let key1 = 5*relu(s1 + 21) - 102
let key2 = 5*relu(s2 + 21) - 103
let key3 = 5*relu(s3 + 21) - 104
let key4 = 5*relu(s4 + 21) - 105

let w0_1 = relu(key0 - key1) - relu(key0 - key1 - 1)
let w0_2 = relu(key0 - key2) - relu(key0 - key2 - 1)
let w0_3 = relu(key0 - key3) - relu(key0 - key3 - 1)
let w0_4 = relu(key0 - key4) - relu(key0 - key4 - 1)
let w1_2 = relu(key1 - key2) - relu(key1 - key2 - 1)
let w1_3 = relu(key1 - key3) - relu(key1 - key3 - 1)
let w1_4 = relu(key1 - key4) - relu(key1 - key4 - 1)
let w2_3 = relu(key2 - key3) - relu(key2 - key3 - 1)
let w2_4 = relu(key2 - key4) - relu(key2 - key4 - 1)
let w3_4 = relu(key3 - key4) - relu(key3 - key4 - 1)

output o2_0 = relu(w0_1 + w0_2 + w0_3 + w0_4 - 3)
output o2_1 = relu(1 - w0_1 + w1_2 + w1_3 + w1_4 - 3)
output o2_2 = relu(2 - w0_2 - w1_2 + w2_3 + w2_4 - 3)
output o2_3 = relu(3 - w0_3 - w1_3 - w2_3 + w3_4 - 3)
output o2_4 = relu(4 - w0_4 - w1_4 - w2_4 - w3_4 - 3)

# --- Layer 6: Emote Readout (Reusing Polysemantic State & Drive Units) ---
let work_em = relu(cs + dig2 + h_dig_act) - relu(cs + dig2 + h_dig_act - 1)
let ekey0 = 13
let ekey1 = 7*hungry + 5
let ekey2 = 21*work_em + 4
let ekey3 = 28*cf + 3
let ekey4 = 35*grip + 2
let ekey5 = 63*scared + 1
let ekey6 = 14*idle_lost

let ew0_1 = relu(ekey0 - ekey1) - relu(ekey0 - ekey1 - 1)
let ew0_2 = relu(ekey0 - ekey2) - relu(ekey0 - ekey2 - 1)
let ew0_3 = relu(ekey0 - ekey3) - relu(ekey0 - ekey3 - 1)
let ew0_4 = relu(ekey0 - ekey4) - relu(ekey0 - ekey4 - 1)
let ew0_5 = relu(ekey0 - ekey5) - relu(ekey0 - ekey5 - 1)
let ew0_6 = relu(ekey0 - ekey6) - relu(ekey0 - ekey6 - 1)
let ew1_2 = relu(ekey1 - ekey2) - relu(ekey1 - ekey2 - 1)
let ew1_3 = relu(ekey1 - ekey3) - relu(ekey1 - ekey3 - 1)
let ew1_4 = relu(ekey1 - ekey4) - relu(ekey1 - ekey4 - 1)
let ew1_5 = relu(ekey1 - ekey5) - relu(ekey1 - ekey5 - 1)
let ew1_6 = relu(ekey1 - ekey6) - relu(ekey1 - ekey6 - 1)
let ew2_3 = relu(ekey2 - ekey3) - relu(ekey2 - ekey3 - 1)
let ew2_4 = relu(ekey2 - ekey4) - relu(ekey2 - ekey4 - 1)
let ew2_5 = relu(ekey2 - ekey5) - relu(ekey2 - ekey5 - 1)
let ew2_6 = relu(ekey2 - ekey6) - relu(ekey2 - ekey6 - 1)
let ew3_4 = relu(ekey3 - ekey4) - relu(ekey3 - ekey4 - 1)
let ew3_5 = relu(ekey3 - ekey5) - relu(ekey3 - ekey5 - 1)
let ew3_6 = relu(ekey3 - ekey6) - relu(ekey3 - ekey6 - 1)
let ew4_5 = relu(ekey4 - ekey5) - relu(ekey4 - ekey5 - 1)
let ew4_6 = relu(ekey4 - ekey6) - relu(ekey4 - ekey6 - 1)
let ew5_6 = relu(ekey5 - ekey6) - relu(ekey5 - ekey6 - 1)

output em0 = relu(ew0_1 + ew0_2 + ew0_3 + ew0_4 + ew0_5 + ew0_6 - 5)
output em1 = relu(1 - ew0_1 + ew1_2 + ew1_3 + ew1_4 + ew1_5 + ew1_6 - 5)
output em2 = relu(2 - ew0_2 - ew1_2 + ew2_3 + ew2_4 + ew2_5 + ew2_6 - 5)
output em3 = relu(3 - ew0_3 - ew1_3 - ew2_3 + ew3_4 + ew3_5 + ew3_6 - 5)
output em4 = relu(4 - ew0_4 - ew1_4 - ew2_4 - ew3_4 + ew4_5 + ew4_6 - 5)
output em5 = relu(5 - ew0_5 - ew1_5 - ew2_5 - ew3_5 - ew4_5 + ew5_6 - 5)
output em6 = relu(6 - ew0_6 - ew1_6 - ew2_6 - ew3_6 - ew4_6 - ew5_6 - 5)

```

### How the Polysemantic Tricks Work Here

* **Additive Sensor Multiplexing (`g_east..g_south`):** In your original code, `task0*gn0 + task1*gf0 + ...` required a hardware multiplier per sensor channel. In the rewritten Layer 5, `relu(gn0 + 2*task0 - 1)` uses `task0` purely as an **additive bias shift**. Because `gn0` is in `[-1, 1]`, when `task0 == 0` the expression `gn0 - 1` is at most `0` (completely silenced by `relu`). When `task0 == 1`, `gn0 + 1` is in `[0, 2]`, passing the linear signal cleanly. Since the seven tasks sum to `1` (`task0 + ... + task6 == 1`), subtracting `1` at the end of `g_east` cancels out the active task's `+1` offset, recovering the exact signed gradient in `[-1, 1]` without a single multiplication.
* **Sub-Threshold Inhibitory Gating (`h_forage_base`, `home`, `h_dig_act`):** Instead of computing a separate boolean `and(free, kf)` and feeding it into `select()`, the neuron `relu(5 - store + 2*old + 3*r1 - 10*(2 - free - kf))` sums the continuous reward features directly with a `-10` penalty for every missing precondition. Whenever `free` or `kf` is `0`, the `-10` shift drags the entire neuron below zero regardless of `old` or `r1`.
* **Cross-Circuit Feature Reuse (`idle_lost` and `h_dig_act`):** Notice how `idle_lost = 1 - relu(kn - free + kf + kg + kh)` folds five state inputs into a single wire that drives both `explore` (`3*idle_lost`) and `ekey6` (`14*idle_lost`). Similarly, `h_dig_act` computes the continuous digging drive (`4 - 2*old + 3*r0`) when the ant is able to dig, and is reused inside `work_em` to trigger the `working` emote whenever `h_dig_act > 0` (barring the rare edge case `starving == 1` while `free` and `kg` are active).

## Notes

Personally, I suspect that with polysemantic weights, we will have less accuracy to the rules. That's okay. We can accept less perfection / memorization, if it means more inter-connected weights. 
We technically want some of that organic quasi-noise that arises from those connections, as it will create surprising behaviors that are usually reasonable, given an unexpected set of inputs.
Although I know in some cases that is controlled just by clipping.
## Results (2026-11-08)

The hand rewrite above is the same circuit the 0.2 compiler already emits (select, and and gt are
lowered to these relu forms anyway). Compiled, it is 1,513 vs 1,559 non-zero weights and 8 vs 10
hidden layers, and 1.07% of 200k random inputs differ from the rule. The cause is a bug, not
interference: `idle_lost` is not `or(idle, not(kn))` and can reach -2. Scripts are in
`wir/scripts/superpose/`. The colony runs are `colony_results.txt`, 10k ticks each.

New 0.2 combinator `unit(x)`: always one hidden unit, relu(x - lo) + lo, so a packed value is
carried as one neuron (otherwise affine lets fold back into their inputs).

**Exact superposition (packed carries).** 69% of the ant brain's hidden units (485 of 706) were
carries, mostly the 24 grade inputs. `gen_pairs.py` packs two ternary grades per neuron
(`unit(3*g0 + g1 + 4)`), the task selects one packed neuron per pair (it means the food gradient
while foraging, the nest gradient going home) and one layer decodes it: 706 -> 631 hidden units,
60.0k -> 44.9k dense params, 11 layers, 0 differences on 200k inputs. Four grades per neuron
(base 81) is also exact (41.4k params) but its sequential decode costs 3 layers.

**Lossy superposition (`fit.py`).** Ten features read by the task and emote choices (home, forage,
dig, dump, help, lost, hungry, workem, cf, grip) go through m shared neurons
`z = unit(relu(C f + c))` and are read back linearly (`D z + d`). Fear, action bits, inertia and
gradient routing stay exact. C and D are fitted in PyTorch on the 998 distinct feature vectors of a
baseline colony run (seed 7, 846k decisions, weighted by frequency), with loss on the task and emote
choices, then rounded to 2 decimals. Neurons came out polysemantic (m = 6: one neuron correlates
with 8 of the 10 features). Colony (seed 7 = fit, 11 and 23 held out; differ = share of decisions
not matching the rule):

| m | differ | starved (7/11/23) | mean pop (7/11/23) |
|---|---|---|---|
| exact | 0 | 17 / 20 / 32 | 86 / 98 / 88 |
| 6 | 0.06-0.10% | 25 / 20 / 37 | 90 / 99 / 91 |
| 5 | 0.20-0.30% | 54 / 12 / 21 | 81 / 81 / 102 |
| 4 | 2.6-4.8% | 63 / 126 / 22 | 76 / 91 / 96 (seed 11 ends at 10 ants) |
| 3 | 5.7-6.2% | 81 / 52 / 67 | 81 / 99 / 101 |
| 2 | 29-30% | 37 / 91 / 41 | 81 / 105 / 113 |

Action bits never changed. Interference falls almost entirely on hungry, free ants at the nest,
where eating competes with foraging and digging, so its cost is starvation, not chaos. m = 4's main
quirk is working <-> hungry (some ants keep working while hungry). m = 3 mostly shows the wrong
emote (hungry shown as straining or carrying). At m = 2, 30% of decisions differ, yet two of three
colonies grew larger than baseline while starving more: a different colony character, not a
collapse. Size is not monotone: m = 4 lost a colony and m = 3 did not, because the fit decides which
features to sacrifice, so loss weights matter as much as m.

Not done: several fits per m (one fit, 6 restarts each), more seeds (single runs diverge
chaotically), an ant.rs generator option, a page to watch the quirks.
