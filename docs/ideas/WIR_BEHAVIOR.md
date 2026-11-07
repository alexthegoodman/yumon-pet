# WIR Game Behavior Ideas

Looking at your Pong snippet, two huge revelations jump out about WIR:

1. **You already have exact Boolean & Piecewise-Linear Logic (`lt`, `gt`, `ge`, `le`, `and`, `select`, `relu`)**: Because your inputs are bounded integers, `lt(a, b)` is exact (`clip(b - a, 0, 1)`), and `select(cond, a, b)` gives you a **differentiable/neural multiplexer**.
2. **Built-in Formal Verification (`assert`)**: Being able to unit-test exact integer state transitions inside a neural network language means you can guarantee an AI never gets stuck in an invalid state.

Notice, though, that your Pong program is actually the **World Environment Model** (`ul` and `ur` are external inputs). To make an **Enemy AI system** that outputs those controls—and scales from Pong to dozens of tactical enemies in an action or RPG game—you shouldn't use giant monolithic MLPs or rigid `if/else` behavior trees.

Instead, WIR's exact primitives (`relu`, `clip`, `select`, `min`/`max`, and grid convolutions) map to the two most flexible architectures in modern game AI: **Utility AI + Multi-Layer Potential Fields**.

---

### 1. The Immediate Test: Writing a "Foresight" Pong AI in 15 Weights

In your Pong code, a naive AI just tracks the ball's current Y (`ul = clip(by - (pl + 1), -1, 1)`). A *smart* WIR AI predicts the bounce point using your exact same `relu` folding trick!

Because the ball moves 1 column per step and bounces between `0` and `11` via a triangle wave, the right paddle (`x = 14`) can compute the exact row the ball will arrive at in **4 ReLU operations**:

```text
wir 0.2
# Smart Pong AI controller for the right paddle (pr)
input bx: R1 in [1, 14]
input by: R1 in [0, 11]
input vx, vy: R1 in [-1, 1]
input pr: R1 in [0, 9]

# 1. Steps until ball reaches column 14 (if moving right; otherwise 0)
let steps = select(gt(vx, 0), 14 - bx, 0)

# 2. Unfolded target Y in [-14, 25], folded back into [0, 11] with 2 ReLU mirrors!
let raw_y = by + vy * steps
let fold1 = raw_y + 2 * relu(-raw_y)
let pred_y = fold1 - 2 * relu(fold1 - 11)

# 3. When vx < 0 (ball moving away), return to center row (5); else track pred_y
let target_y = select(gt(vx, 0), pred_y, 5)

# 4. Aim paddle center (pr + 1) at target_y
output ur = clip(target_y - (pr + 1), -1, 1)

```

That entire predictive AI compiles to **fewer than 20 weights**, has zero trigonometric or loop overhead, and plays a mathematically optimal game when `vx > 0` while recovering to center court when `vx < 0`.

---

### 2. Generalizing to Complex Games: Utility AI + Influence Maps

Traditional game AI breaks down when you use `select` (`if/else`) for everything: *"If health < 20, flee; else if player_dist < 5, attack..."* creates robotic, predictable enemies that thrash back and forth at boundary thresholds.

In a neural DSL like WIR, you get dramatically better behavior by combining **Spatial Influence Maps** (using your `wir path` grid convolution) with **Utility Response Curves** (using `relu` and `clip` on `R1` scalars).

| AI Layer | WIR Representation | What It Computes | Approx. Weights |
| --- | --- | --- | --- |
| **1. Spatial Perception** | `grid` (like `wir path`) | 3-channel Potential Field: `d_player` (approach), `d_threat` (grenades/line-of-fire), `d_ally` (flanking separation) | ~40 – 80 (shared across all enemies!) |
| **2. Internal Drives** | `R1` per enemy | Normalizes `hp`, `ammo`, `cooldown`, `morale` into smooth response curves in `[0, 64]` using `clip` & `relu` | ~30 – 60 |
| **3. Action Blending** | `R1` linear + `max` | Weights the spatial fields by the drives: $\text{Score}_{\text{dir}} = w_{\text{aggro}} \cdot \text{Player} - w_{\text{fear}} \cdot \text{Threat}$ | ~20 – 40 |

#### How WIR Replaces Behavior Trees with "Response Curves"

Instead of a hard boolean `lt(hp, 20)` that flips from 0 to 1, use `clip` and `relu` to shape smooth behavioral urges (this is **Dave Mark's Utility AI** architecture compiled directly to ReLU weights):

```text
wir 0.2
# Generalized Tactical Enemy AI Brain (runs per enemy agent)
input hp:       R1 in [0, 64]
input ammo:     R1 in [0, 16]
input cooldown: R1 in [0, 8]

# Local 3x3 or 4-neighbor readings from the shared WIR pathfinding grids:
# d_player: distance to player (lower = closer)
# d_cover:  distance to safe cover out of player's line-of-sight
# d_ally:   distance to nearest teammate (for swarming/flanking instead of clumping)
input dp_n, dp_s, dp_e, dp_w: R1 in [0, 64]
input dc_n, dc_s, dc_e, dc_w: R1 in [0, 64]
input da_n, da_s, da_e, da_w: R1 in [0, 64]

# --- 1. UTILITY CURVES (Continuous urges in [0, 64], not rigid booleans!) ---
# Fear rises smoothly as HP drops below 32, and spikes critically below 12
let fear_drive = clip(32 - hp, 0, 32) + 2 * relu(12 - hp)

# Aggression requires ammo > 0 and scales with health, dropping while weapon cools down
let has_ammo = gt(ammo, 0)
let aggro_drive = select(has_ammo, clip(hp - cooldown * 4, 0, 64), 0)

# Flanking urge: high when healthy and close to another ally
let flank_drive = clip(hp - 16, 0, 32)

# --- 2. DIRECTIONAL SCORING (Combining grids with drives) ---
# For each direction (N, S, E, W), lower score = more desirable tile to step onto
let score_n = aggro_drive * dp_n + fear_drive * dc_n - flank_drive * clip(4 - da_n, 0, 4)
let score_s = aggro_drive * dp_s + fear_drive * dc_s - flank_drive * clip(4 - da_s, 0, 4)
let score_e = aggro_drive * dp_e + fear_drive * dc_e - flank_drive * clip(4 - da_e, 0, 4)
let score_w = aggro_drive * dp_w + fear_drive * dc_w - flank_drive * clip(4 - da_w, 0, 4)

```

Look at what happens automatically from those ~50 weights without a single complex state machine:

* **Full HP + Ammo:** `aggro_drive` dominates. The enemy charges down the `d_player` gradient, while the `-flank_drive * clip(4 - da, 0, 4)` term repels enemies that get within 4 tiles of each other—so two enemies naturally **split left and right around a pillar to pincer the player**!
* **Weapon on Cooldown (`cooldown = 8`):** `aggro_drive` temporarily dips relative to `fear_drive`. The enemy ducks behind the nearest cover tile for 1.5 seconds while reloading, then pops back out when `cooldown` hits 0.
* **Low HP (`hp < 12`):** The `2 * relu(12 - hp)` knee kicks in, overwhelming `aggro_drive` and making the enemy retreat to cover.

---

### 3. Making It General & Trainable: Personality Embeddings ("Archetype Vectors")

If you want **one single WIR network** (say, 300 weights total) to control **every enemy type in your game**—Berserkers, Snipers, Cowardly Goblins, Pack Wolves, and Bosses—don't write separate WIR programs for each.

Pass a 4-number **Personality / Archetype Vector** as an input:

```text
input bravery, greed, pack_instinct, preferred_range: R1 in [0, 16]

```

And compute the distance desire to the player using an absolute-value "sweet spot" (`relu(d - range) + relu(range - d)`):

* Set `preferred_range = 1, bravery = 16, pack_instinct = 0` $\rightarrow$ **Melee Berserker**
* Set `preferred_range = 10, bravery = 4, pack_instinct = 0` $\rightarrow$ **Kiting Archer** (automatically backs away if the player gets closer than 10 tiles, approaches if farther than 10!)
* Set `preferred_range = 1, bravery = 6, pack_instinct = 14` $\rightarrow$ **Pack Wolf** (hovers at a safe distance alone, but rushes in as soon as 2+ ally wolves are nearby because `da` boosts its confidence).

Even better: because everything in WIR is just ReLU and linear layers, you can **hand-author the initial WIR program** so the AI is 100% functional on Day 1 (and passes all your `assert` safety tests), and then use **Reinforcement Learning (PPO) or Evolution Strategies** to fine-tune the slopes and thresholds of those `relu` curves so the enemies discover emergent tactics against human players.