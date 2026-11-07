# WIR Physics Ideas

Since WIR is a blank slate and you control the compiler, you are in a really unique position.

If your goal is to simulate realistic fluids, soft bodies, and even rigid collisions in **under 1,000 to 5,000 parameters**, adding **four specific primitives** to WIR will let you run almost any modern physics algorithm (including SPH, MPM, and Position-Based Dynamics) without bloating the network.

**Of course, WIR will be used for much more than just tiny physics and game-related models. But this can be one space in which we innovate and build robustness into WIR.**

The following ideas are merely proposals, nothing has been fully assessed.

---

## How Three Grid Fluid Architectures Map to WIR

Depending on whether you want pooling water (free-surface) or swirling smoke/currents, three classical grid methods compile directly into WIR's `min`, `max`, `clip`, and neighbor indexing:

| Method | Channels per Cell | Approx. WIR Weights | Best For | Key WIR Advantage |
| --- | --- | --- | --- | --- |
| **2.5D Shallow Water (SWE)** | 3 (`h`, `u`, `v`) | ~60 – 120 | Puddles, waves, wakes, sloshing containers | 1 pass per frame; zero pressure-iteration loops |
| **Lattice-Boltzmann (D2Q9)** | 9 (`f0`..`f8`) | ~150 – 300 | Swirling vortices, wakes around obstacles | Purely local neighbor reads + linear combinations |
| **Eulerian Free-Surface** | 4 (`mass`, `u`, `v`, `p`) | ~100 – 200 | Splashing 2D/3D side-view liquids | Explicit mass conservation via `clip` and `min` |

---

### A. The Sweet Spot for Small Fluids: Flux-Limited Cellular Liquid (~80 Weights)

If you want a small volume of water sloshing around walls, pouring through gaps, and pooling at the bottom of a level, a  hydrostatic flux-limited cellular automaton (or 2D Shallow Water solver) is the cleanest match for WIR.

Instead of solving global differential equations, each cell holds a fluid volume $m \in [0, 64]$ (where 64 is a full cell, and $>64$ represents slight compression under pressure). In each step, a cell calculates how much fluid wants to flow Down, Left, Right, and Up using `clip` and `min`:

```text
wir 0.2
# Hydrostatic cellular fluid step (simplified vertical + horizontal leveling)
input m: grid in [0, 64] pad 0
input wall: grid in [0, 1] pad 1

# How much space is left in the cell below (0 if wall below)
let cap_down = clip(64 - m[0, 1] - 64*wall[0, 1], 0, 64)
let flow_down = min(m, cap_down)

# How much fluid arrives from the cell above
let cap_self = clip(64 - m - 64*wall, 0, 64)
let flow_from_up = min(m[0, -1], cap_self)

# Remainder after vertical gravity step
let m_vert = m - flow_down + flow_from_up

```

Because `flow_down` in cell $(x, y)$ is algebraically identical to `flow_from_up` in cell $(x, y+1)$, **mass is 100% conserved to machine precision**—not a single drop of fluid is ever created or destroyed, even over millions of steps. Adding horizontal pressure leveling (`(m[-1, 0] - m) / 4`) and momentum channels (`vx, vy`) turns this into a full shallow-water or compressible liquid solver in under 150 weights.

---

### B. Lattice-Boltzmann (LBM D2Q9) in WIR (~200 Weights)

Lattice-Boltzmann doesn't track velocity and pressure directly; instead, each cell stores **9 particle distribution numbers** ($f_0 \dots f_8$) representing how many fluid particles are moving in the 9 cardinal/diagonal directions:

1. **The Streaming Step (0 weights, just indexing):** Particles simply move from neighbor cells into the current cell: `let f1_stream = f1[-1, 0]`, `let f2_stream = f2[0, -1]`, etc.
2. **Bounce-Back Walls (`clip` masking):** When `wall == 1`, incoming particles reflect into the opposite direction (`f1` swaps with `f3`). This uses the exact same branchless `wall` masking trick from your `wir path` example.
3. **The BGK Collision Step (Linear + Quadratic):** Each direction relaxes toward a local equilibrium:

$$f_i^{\text{new}} = f_i - \frac{1}{\tau}(f_i - f_i^{\text{eq}})$$



In the **Stokes / low-Mach regime** (slow to moderate speeds), the quadratic velocity term $u^2$ in $f_i^{\text{eq}}$ is tiny and can either be dropped (making LBM **100% linear**—just a single $9 \times 9$ matrix multiply per cell!) or approximated in WIR using a 2-segment piecewise-linear ReLU curve: $u^2 \approx \text{ReLU}(\vert{}u\vert{} - c)$.

**The LBM Trade-off:** Standard LBM fills the entire grid with fluid (like air in a wind tunnel or water filling a tank). If you want a *small droplet or splash* moving through empty air, you need **Free-Surface LBM** (tracking an extra `fill` fraction channel per cell) or a two-phase Shan-Chen force, which adds ~100 weights of neighbor density interactions.

---

### C. True Incompressible Eulerian Fluids (Projection Step = Your Pathfinding!)

In a classic Stam "Stable Fluids" Eulerian engine, the hardest part is the **Pressure Projection step** (solving the Poisson equation $\nabla^2 p = \nabla \cdot \mathbf{u}$ so the fluid doesn't compress).

Look closely at what a Jacobi Poisson solver step is on a 2D grid:


$$p_{\text{new}} = \frac{p_{[-1,0]} + p_{[1,0]} + p_{[0,-1]} + p_{[0,1]} - \text{divergence}}{4}$$


That is structurally identical to your `wir path` distance field, except replacing the `min(...)` tree with an **average** (which is even cheaper in neural weights—just four $0.25$ weights in a single convolution filter!). For a small patch of fluid, running 4 to 8 Jacobi passes per frame removes compression and produces realistic swirling vortices.

---

## How SPH Fluid Architectures Could Map to WIR

### 1. Multi-Channel Tuples / Structs per Cell (`grid<4>`)

Right now, `input d: grid in [0, 64]` defines a single scalar field. For physics, a cell (or particle) needs to hold state:

* **Fluids (Eulerian / Reintegration SPH):** `(mass, ox, oy, vx, vy)`
* **Soft/Rigid Bodies:** `(pos_x, pos_y, vel_x, vel_y, phase_id)`

In neural terms, this just turns a 2D convolution with 1 input channel into a 2D convolution with $C$ input channels (`Conv2d(C_in, C_out, kernel_size=3)`).

```text
# Proposed WIR syntax for multi-channel state
state cell: grid<mass: [0, 64], ox: [-32, 32], oy: [-32, 32], vx: [-64, 64], vy: [-64, 64]>

```

---

### 2. Built-in Antisymmetric Flux (`flow` primitive) for Exact Conservation

The #1 reason neural physics models fail is that they violate conservation of mass and momentum—water slowly vanishes or explodes.

If you add a `flow` or `exchange` keyword to WIR, the compiler can automatically wire **antisymmetric weights** ($W_{i \to j} = -W_{j \to i}$). Whatever leaves cell `A` into cell `B` is guaranteed by the network topology to be added to cell `B`:

```text
# WIR compiler guarantees sum(m) across the entire grid never changes
flow m_right = clip((m[0, 0] - m[1, 0]) * 0.25, -m[1, 0], m[0, 0])

```

Under the hood, this is simply a convolution kernel with fixed `[-1, +1]` divergence stencil weights at the final layer. Zero training required, and your fluid or energy can never leak.

---

### 3. Spatial Gather / Splat (`scatter` / `gather`) for True SPH & MPM

If you want particle physics (SPH or Material Point Method) without an $O(N^2)$ all-pairs explosion, WIR needs one bridge between a **1D particle list** and a **2D spatial grid**:

* **Option A: Pure Grid "Reintegration Tracking" (Zero new tensor ops needed!)**
Keep everything as a 2D `grid`, where each cell holds at most 1 particle `(mass, ox, oy, vx, vy)`. To make writing this painless in WIR, add a built-in helper for `advect(field, vx, vy)` or `reintegrate(...)`, which compiles to a fixed $3 \times 3$ bilinear interpolation stencil using `clip` overlaps.
* **Option B: Differentiable Grid Splatting (MPM / PIC)**
Allow WIR to declare both a `particles: list[N]` and a `grid[W, H]`. A single fixed bilinear `splat(particles -> grid)` and `gather(grid -> particles)` layer lets you simulate **snow, mud, water, elastic jelly, and rigid chunks** all interacting in the same 500-weight grid solver (this is how Disney's *Frozen* snow solver and modern MPM engines work).

---

### 4. Learnable Constants (`param`) for Hybrid Symbolic + ML Physics

This is where WIR can genuinely beat classical engines like Jolt or standard fluid solvers.

Classical grid fluids suffer from **numerical viscosity**—on a coarse grid (say $32 \times 32$), water looks like thick syrup because low-res finite differences smooth out sharp vortices.
If you add a `param` keyword to WIR:

```text
param viscosity in [0, 1] = 0.12
param vorticity_boost in [-2, 2] = 0.45
param kernel_3x3: stencil[3, 3]

```

You can write the **90% exact physics backbone** symbolically in WIR (conservation of mass, wall bounce-back, gravity) in ~200 hardcoded weights, and leave **50 to 100 learnable weights (`param`)** inside the pressure/vorticity step. Then you train *only* those 100 weights against a high-resolution offline SPH simulation so your coarse $32 \times 32$ WIR grid behaves with the splashiness of a $256 \times 256$ simulation!

However, we want to avoid training any weights except for verification and comparison purposes, or perhaps analysis. The goal of WIR is to create tiny NNs without training.

---

We are also somewhat curious as to the performance implications of running these simulations via models which are indeed running on GPU. It begs the question, is it ever faster than traditional CPU-bound processes? Is it more generalized? 
Is it at least simpler for an LLM to generate the physics system in WIR instead of a whole program? These are some of the questions we should ask regarding the benefits of WIR for physics at this early stage.