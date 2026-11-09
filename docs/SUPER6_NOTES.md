The superposition bottleneck is far from arbitrary—it exhibits classic **sparse coding and state-mutually-exclusive geometry**. Rather than distributing all 10 features uniformly across the 6 hidden neurons (`z0..z5`), the fitted matrices ($C$ and $D$) exploit the behavioral constraints of the ant colony (e.g., an ant cannot simultaneously carry food, carry soil, grip, and be free to forage) to pack 10 dimensions into 6 using **antipodal pairs**, **ReLU gating**, and **task-competition contrast**.

Here are the specific structural patterns encoded in the weights:

### 1. Near-Monosemantic Feature Anchor (`z4` $\leftrightarrow$ `sp_lost`)

Neuron `z4` is almost entirely dedicated to a single binary feature: `sp_lost` (whether the ant is idle or lacks a home gradient `kn`).

* **Encoding:** `z4` has a negative bias (`-0.31`) and a massive positive weight for `sp_lost` (`+1.39`), while almost all other inputs are near zero or negative. Thanks to the `relu`, `z4` remains strictly `0` during normal working/foraging states and fires strongly (~`1.08`) only when `sp_lost = 1`.
* **Decoding:** `r_lost` reads almost exclusively from `z4` (`+1.25*z4`), and `z4` Simultaneously applies heavy negative suppression to all active task drives (`-7.06` on `r_home`, `-4.28` on `r_help`, `-3.39` on `r_dig`).

### 2. Antipodal State Pairing (`z0` vs. `z5` for `grip` and `hungry`)

Because `grip` requires `not(starving)` and occupies the ant's mandibles (`free = 0`), high hunger and active gripping occupy distinct regions of state space. Neurons `z0` and `z5` form a complementary dipole to multiplex `grip` and `hungry`:

* **`z0` (+grip, +hungry, -work/-cf):** Fires strongly when `grip = 1` (`+1.08`), and scales up with `hungry` (`+0.08` per unit, up to `+0.80`).
* **`z5` (-grip, +hungry):** Has a strong *negative* weight on `grip` (`-0.86`) and a strong *positive* weight on `hungry` (`+0.22` per unit, up to `+2.20`).
* **How the readout untangles them:**
* `r_hungry` sums both neurons positively (`+3.15*z0 + 7.27*z5`), so `grip` cancels out (`3.15 * 1.08 + 7.27 * (-0.86) < 0` clamp) while `hungry` compounds across both.
* `r_grip` takes the difference (`+0.77*z0 - 0.31*z5`), isolating the `grip` signal and cancelling out `hungry`.



### 3. Binary State Multiplexing via ReLU Clamping (`z1` vs. `z3` for `cf` and `sp_work`)

The binary carrying/working states (`cf`, `sp_work`, `grip`) are mutually exclusive with `free` tasks (`forage`, `dig`, `help`). Neurons `z1` and `z3` start with high positive biases (`+0.59` and `+0.52`) that act as "carrier waves" modulated by binary state flags:

* **`cf` (carrying food) splits `z1` and `z3`:** `cf` adds `+0.24` to `z1` but subtracts `-0.99` from `z3` (clamping `z3` to `0` via ReLU).
* Consequently, `z1` high + `z3` zero is the unique signature of `cf = 1`. Look at the decoder: `r_home` (`+13.31*z1 - 8.82*z3`) and `r_cf` (`+0.52*z1 - 0.79*z3`) both reward `z1 - z3`, whereas `r_forage` (`-5.43*z1 + 10.67*z3`) strongly rewards `z3 - z1` (since an ant carrying food cannot forage).


* **`sp_work` (digging/carrying soil) suppresses both `z1` and `z3`:** `sp_work` applies `-0.47` to `z1` and `-0.49` to `z3`, pulling both carrier neurons down toward zero.
* The decoder for `r_workem` (`1.15 - 0.48*z1 - 1.04*z3`) uses an inverted logic gate: it starts with a high positive intercept (`+1.15`) that is actively held down to `~0` when `z1` and `z3` are firing, and gets "released" to `+1.15` when `sp_work = 1` silences `z1` and `z3`.



### 4. Continuous Drive Encoding via Differential Directions (`z1, z2, z3`)

Unlike the binary flags (`cf`, `grip`, `sp_lost`, `sp_work`) which have large encoder weights (`0.3` to `1.39`), the continuous task drives (`home`, `forage`, `dig`, `help`, which range from `0` to `14`) have tiny encoder weights (`0.01` to `0.16`) and huge decoder weights (`3.0` to `13.3`):

* **Why the scale asymmetry?** To prevent the `relu` from clipping at `0` across varying combinations of continuous drives, the encoder scales the continuous features down by ~10x so they gently tilt the activation vector inside the positive orthant around the bias point (`+0.59`, `+0.11`, `+0.52`), and the linear decoder scales them back up by ~10x.
* **Winner-Take-All Task Alignment:** Because `r_home`, `r_forage`, `r_dig`, `r_dump`, and `r_help` only matter inside a `gt()` argmax tournament (`tkey0..tkey6`), the bottleneck doesn't need to reconstruct their exact absolute values—only their *relative differences* when two drives are simultaneously active (i.e., when `free = 1`):
* **`forage` vs. `help` / `dig`:** `forage` pushes `z2` down (`-0.14`) and `z3` up (`+0.02`), whereas `help` (`+0.16*z2`) and `dig` (`+0.08*z2`) push `z2` up. The decoders mirror this exact axis: `r_forage` weights `z2` negatively (`-4.89`) and `z3` positively (`+10.67`), while `r_dig` (`+5.00*z2`) and `r_help` (`+4.44*z2`) weight `z2` strongly positively.
* **`dump` as a state-parasite:** Notice `dump = 6 * and(cs, kn)` barely has weights in the encoder (`-0.08` to `+0.11`). Why? Because whenever `dump` is active, `cs = 1`, which forces `sp_work = 1` (suppressing `z1` and `z3`). The decoder for `r_dump` (`1.38 + 0.98*z0 - 1.34*z2 - 2.54*z3`) piggybacks on `z3` being silenced by `sp_work`, reconstructing `dump` mostly from the absence of `z3` and `z2`!



### Summary Matrix of the 6 Neurons' Roles

| Neuron | Bias | Primary Role in the Bottleneck |
| --- | --- | --- |
| **`z0`** | `+0.19` | **Grip & Dump pole:** Activated strongly by `grip` (`+1.08`), `dump` (`+0.11`), and `hungry` (`+0.08`); inhibited by `cf` (`-0.47`) and `sp_work` (`-0.31`). |
| **`z1`** | `+0.59` | **Home / Carrying-Food pole:** High baseline carrier; boosted by `cf` (`+0.24`), `home` (`+0.07`), and `help` (`+0.10`); silenced by `sp_work` (`-0.47`) and `forage` (`-0.08`). |
| **`z2`** | `+0.11` | **Dig & Help vs. Forage discriminator:** Boosted by `help` (`+0.16`), `dig` (`+0.08`), and `cf` (`+0.41`); suppressed by `forage` (`-0.14`). |
| **`z3`** | `+0.52` | **Free-Forage / Non-Carrying baseline:** High baseline when `free = 1`; clamped to `0` by `cf` (`-0.99`) or `sp_work` (`-0.49`). Drives `r_forage` (`+10.67`) and `r_help` (`+5.60`). |
| **`z4`** | `-0.31` | **Dedicated `sp_lost` detector:** Gated off by negative bias; fires almost exclusively when `sp_lost = 1` (`+1.39`). |
| **`z5`** | `+0.06` | **Hunger pole (Anti-Grip):** Driven by `hungry` (`+0.22`); clamped to `0` when `grip = 1` (`-0.86`). |