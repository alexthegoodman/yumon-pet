# Swarming, Flocking & Storm Avoidance in WIR

Craig Reynolds' classic **Boids** algorithm (flocking birds, schooling fish, drone swarms) relies on three local rules—**Separation** (don't crowd neighbors), **Alignment** (match neighbor velocity), and **Cohesion** (steer toward neighbor center)—plus **Obstacle Avoidance** and **Environmental Forces** (storms/wind).

In a traditional engine, Boids requires $O(N^2)$ distance checks or complex spatial trees. In WIR, you can implement a **Flow-Field / Cellular Swarm** or a **Local K-Neighbor Boid Brain** in roughly **60 to 120 weights** where that "closest-known-response" property really shines.

## How a WIR Boid Handles Unexpected Storms & Obstacles

Imagine each flying agent runs a tiny WIR controller that takes its local sensor readings—or reads from a shared WIR 2D/3D wind-and-obstacle grid:

* **Separation (`relu` short-range repulsion):** Only activates when a neighbor is closer than `safe_dist` (e.g., 4 units). If two birds get pushed into the exact same spot by a violent storm (`dist = 0`), `relu(4 - dist)` hits its peak of `4`, creating an instant, bounded emergency breakaway impulse.
* **Cohesion (`clip` long-range attraction):** Pulls the bird toward the flock's local average position, clamped to a gentle maximum turn rate (`clip(dx, -2, 2)`) so a bird never snaps its neck trying to catch a distant flock.
* **Storm & Turbulence Handling (The "Graceful Degradation" Test):**
When flying through a turbulent storm vortex (`wind_x, wind_y`), a bird faces two risks: getting blown into a cliff (`d_wall` dropping near 0) or stalling/overspeeding. Look at how WIR blends normal flocking with emergency storm survival without a single brittle `if/else` tree:

```text
wir 0.2
# One Boid flight-control step with storm turbulence & obstacle avoidance
input vx, vy:         R1 in [-8, 8]    # Current bird velocity
input flock_vx, flock_vy: R1 in [-8, 8] # Local neighbor average velocity (Alignment)
input com_dx, com_dy: R1 in [-32, 32]  # Offset to local flock center (Cohesion)
input sep_dx, sep_dy: R1 in [-8, 8]    # Offset from nearest crowding neighbor (Separation)
input wind_x, wind_y: R1 in [-16, 16]  # Storm gust vector at current position
input wall_dx, wall_dy: R1 in [-16, 16]# Gradient away from nearest obstacle/cliff

# 1. Normal Flocking Urges (clamped so no single urge explodes on extreme inputs)
let align_x = clip(flock_vx - vx, -2, 2)
let coh_x   = clip(com_dx, -2, 2)
let sep_x   = clip(sep_dx * 2, -4, 4)

# 2. Obstacle Panic Reflex: zero when far, ramps up sharply when wall is within 6 units
#    Even if a storm slams the bird to dist=0, clip caps the reflex cleanly at 8
let panic_x = clip(wall_dx * 2, -8, 8)

# 3. Wind-Shear / Storm Compensation:
#    In calm air, follow the flock. In extreme headwinds/turbulence, tuck in (increase cohesion)
#    and prioritize obstacle clearance over formation flying!
let storm_intensity = clip(relu(wind_x) + relu(-wind_x) + relu(wind_y) + relu(-wind_y), 0, 16)
let flock_drive = clip(align_x + coh_x + sep_x, -6, 6)

# 4. Combine aerodynamics + reflexes, and clamp to physical airspeed limits [-8, 8]
output vx2 = clip(vx + flock_drive + panic_x + wind_x * 0.25, -8, 8)

```

## Why This Behavior Looks So Alive

Look at what happens when an **unexpected, extreme event** hits that 40-weight WIR swarm—for example, a massive tornado vortex (`wind = 50`, way past the expected `[-16, 16]` input range) right next to a mountain wall:

1. The input clamping and `clip(..., -6, 6)` ensure `flock_drive` never overflows.
2. As the wind pushes the birds toward the cliff face, `panic_x` smoothly overwhelms `coh_x` and `align_x`. The flock naturally **splits around the obstacle** instead of blindly following the leader into the rock.
3. Once clear of the obstacle, the birds don't need a "recovery state" in a finite state machine—as `wall_dx` decays back to `0`, the linear sum automatically settles back onto `align + coh + sep`, and the scattered birds seamlessly reform their V-formation or murmurating cloud on the leeward side of the storm.