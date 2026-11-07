# WIR Music Ideas

A MIDI piano roll is literally a 2D integer grid (`X = time step`, `Y = pitch`, `value = velocity in [0, 127]`), and a monophonic melody or drum track is just a 1D integer state machine (`R1`).

Because music theory—scales, chord voicings, voice leading, and Euclidean drum rhythms—is built on **integer intervals and modular arithmetic**, WIR can compile complete generative music engines into **30 to 200 weights** with zero off-key notes.

Two distinct WIR architectures work best here: a **2D Grid Convolution** for polyphonic piano rolls, and a **1D Stateful Stepper (`R1`)** for real-time procedural game soundtracks.

---

### 1. The 2D Piano-Roll Approach: Harmonic Cellular Automata (`grid`)

If you represent a 16-step bar across 12 or 24 pitches as `input note: grid in [0, 1]`, a WIR convolution looks at neighboring time steps (`dx` = past/future beats) and neighboring pitches (`dy` = semitone intervals):

* `note[-1, 0]` = Was this exact pitch played on the previous 16th note? (Sustain / Repeat)
* `note[-1, -2]` or `note[-1, 2]` = Was a whole-step interval played on the previous beat? (Stepwise melodic motion)
* `note[0, -3]`, `note[0, -4]`, `note[0, -7]` = Is there a minor 3rd, major 3rd, or perfect 5th playing *simultaneously* on the current beat? (Harmony / Consonance check)
* `note[0, -1]` or `note[0, 1]` = Is there a harsh semitone clash right next to this note? (Dissonance suppression)

Here is a complete **polyphonic melody & counterpoint generator** in WIR that takes a sparse seed pattern (or repairs a noisy grid) and evolves it according to music theory rules:

```text
wir 0.2
# One step of a Harmonic Piano-Roll Automaton
# X-axis = time (16th notes), Y-axis = pitch (semitones)
input note:  grid in [0, 1] pad 0
input scale: grid in [0, 1] pad 0   # 1 if pitch row is in the active key/scale, 0 otherwise

# 1. Melodic momentum: did a step-wise neighbor (-2, -1, +1, +2 semitones) play on the previous beat?
let melodic_drive = note[-1, -2] + note[-1, -1] + note[-1, 1] + note[-1, 2]

# 2. Harmonic support: is a consonant 3rd (-3, -4) or 5th (-7) active on the current or previous beat?
let harmonic_support = note[0, -7] + note[-1, -7] + note[-1, -4] + note[-1, -3]

# 3. Anti-clutter / Dissonance: penalize simultaneous semitone clashes (dy = -1, +1) 
#    and stop a single pitch from droning for more than 2 steps in a row
let semitone_clash = note[0, -1] + note[0, 1]
let drone_penalty  = and(note[-1, 0], note[-2, 0])

# 4. Trigger note if melodic + harmonic energy == 2, no clashes, and pitch is in scale!
let raw_trigger = eq(melodic_drive + harmonic_support, 2)
let clean       = clip(raw_trigger - semitone_clash - drone_penalty, 0, 1)

output note2 = and(clean, scale)

```

In roughly **25 weights**, this grid rule enforces three pillars of composition simultaneously:

1. **Scale Quantization (`and(clean, scale)`):** By feeding a 12-row repeating mask `[1,0,1,0,1,1,0,1,0,1,0,1]` (Major) or `[1,0,1,1,0,1,0,1,1,0,1,0]` (Minor) into `scale`, the network is mathematically incapable of outputting an out-of-key note.
2. **Voice Leading:** Notes preferentially step up or down by 1–2 semitones (`melodic_drive`) from the previous beat.
3. **Counterpoint & Anti-Muddiness:** `semitone_clash` prevents two adjacent keys from being mashed together, while `drone_penalty` forces rhythmic movement.

---

### 2. The 1D Real-Time Approach: Euclidean Beats + Arpeggiator (`R1` like Pong)

If you want a WIR model running live inside a game audio engine—generating a drum beat and bassline tick-by-tick just like your `wir pong` state machine—you don't even need a 2D grid.

One of the coolest algorithms in rhythm theory is **Bjorklund’s Euclidean Rhythm** (`E(k, n)`), which distributes $k$ drum hits as evenly as possible across $n$ steps (for example, `E(3, 8)` produces the classic Cuban *tresillo* `[x . . x . . x .]`, and `E(5, 16)` produces the *Bossa Nova* clave).

In integer math, a Euclidean rhythm is simply a **Bresenham line accumulator** (`acc = (acc + k) mod n`), which compiles into a single WIR `relu` wrap-around!

```text
wir 0.2
# Real-time 16-step Euclidean Beat + Melodic Bassline Stepper
input step:      R1 in [0, 15]   # Current 16th-note tick
input acc_kick:  R1 in [0, 15]   # Euclidean accumulator for kick drum
input acc_hat:   R1 in [0, 15]   # Euclidean accumulator for hi-hat
input k_kick:    R1 in [1, 16]   # Rhythm density (e.g. 4 = four-on-the-floor, 3 = tresillo)
input k_hat:     R1 in [1, 16]   # Hi-hat density (e.g. 7 or 11 for polyrhythms)
input pitch:     R1 in [0, 24]   # Current melody scale degree (0..24)
input dir:       R1 in [-1, 1]   # Melodic contour direction (-1 = falling, +1 = rising)
input tension:   R1 in [0, 4]    # Game state input! (0 = calm exploration, 4 = boss combat)

# --- 1. EUCLIDEAN DRUM GENERATOR (Exact Bresenham Modulo via ReLU) ---
let next_kick = acc_kick + k_kick
output hit_kick = ge(next_kick, 16)
output acc_kick2 = next_kick - 16 * relu(next_kick - 15)

let next_hat = acc_hat + k_hat
output hit_hat = ge(next_hat, 16)
output acc_hat2 = next_hat - 16 * relu(next_hat - 15)

# Snare hits on backbeats (steps 4 and 12)
output hit_snare = or(eq(step, 4), eq(step, 12))

# --- 2. MELODIC STEPPER (Contour + Tension-Driven Intervals) ---
# Jump by larger intervals when game 'tension' is high; step smoothly when calm
let interval = 1 + tension
let raw_pitch = pitch + dir * select(hit_kick, interval, 1)

# Bounce melody direction at the top (24) and bottom (0) of the 2-octave range (just like Pong!)
let hit_top = gt(raw_pitch, 24)
let hit_bot = lt(raw_pitch, 0)
output pitch2 = raw_pitch + 2 * relu(-raw_pitch) - 2 * relu(raw_pitch - 24)
output dir2   = dir + 2 * hit_bot - 2 * hit_top
output step2  = step + 1 - 16 * eq(step, 15)

```

Look at how your **Pong wall-bounce identity** (`raw_pitch + 2*relu(-raw_pitch) - 2*relu(raw_pitch - 24)`) resurfaces here:

* In Pong, that triangle wave bounced a ball between the top and bottom walls.
* In MIDI space, that exact same ReLU reflection bounces a **melodic phrase or arpeggiator contour** between a low root note and a two-octave high point!

---

### 3. Training vs. Hand-Writing a WIR Music Model

Because WIR compiles to standard neural layers, you can combine both worlds:

* **Hardcode the Music Theory Guardrails (~30 weights):** Lock in the Euclidean accumulator, the octave reflection bounds, and the `scale` mask so the model can *never* drift off-beat or play a sour note.
* **Learn the Transition Matrix (~100 weights):** Replace `melodic_drive` with a small $3 \times 7$ convolutional stencil trained on 50 MIDI files (e.g., Bach chorales, NES chiptunes, or house basslines). Because the receptive field only needs to see the last 4 beats and $\pm 1$ octave, a single learned $4 \times 12$ convolution filter has only **48 weights per channel**—yet it captures the exact interval and syncopation probabilities of the target genre.

---

However, we would ideally not train any part of a WIR model at this time. In the future, perhaps we will have hybrid for the sake of production feasibility. For now, we want to emphasize the utility of WIR by focusing on writing from scratch.