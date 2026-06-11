# StochasticWfn Development Guide

This document tracks the current state of `StochasticWfn` and the methods that must be
overridden for it to function as a full-fledged stochastic trial wavefunction in SAFIRE,
rather than a thin delegate wrapper around `NOMSD`.

---

## Current State

`StochasticWfn` (`StochasticWfn.hpp` / `StochasticWfn.icc`) is a `template<bool MP, class devPsiT>`
class that inherits from `AFQMCInfo` and owns:

```cpp
StochasticInnerEnsemble inner_ensemble_;  // inner WalkerSet + RNG (walker init only)
int inner_nwalkers_{1};
int inner_nsteps_{0};                     // parsed; > 0 rejected until Phase 2b+
NOMSD<MP, devPsiT> nomsd_;                // outer delegate (outer-walker-facing)
std::unique_ptr<StochasticInnerStack<MP, devPsiT>> inner_stack_;  // inner NOMSD + Propagator + RNG
```

**Phase 1a is complete.** The inner trial ensemble (`inner_wset()`, sized by `inner_nwalkers`)
is owned and initialized in a two-phase factory/driver workflow.

**Phase 1b is complete.** Outer and inner each have their own `NOMSD` instance with
independently constructed `HamiltonianOperations` and `SlaterDetOperations` (cloned CI and
orbitals for the inner copy).

**Phase 1c is complete.** The inner `NOMSD` (Phase 1b) is wrapped in a heap-allocated
`Wavefunction`, bound to an inner `Propagator` built via `PropagatorFactory` at wavefunction
construction time, with a dedicated device RNG. Default `inner_nsteps = 0`: the propagator
is wired but **not invoked**; the inner ensemble remains static until Phase 2b overrides
`Energy` and enables `inner_nsteps > 0`.

**Phase 2a (Overlap) is implemented (pending build + test).** `Overlap` reduces the inner
ensemble into an effective stochastic-trial overlap per Eq. 24 of arXiv:2505.18519 in the
static-ensemble limit; see [Stochastic overlap (Phase 2a)](#stochastic-overlap-phase-2a).

**Phase 2b (Energy) is next.** `Energy` still delegates to `nomsd_`; the Tier 1 density/bias
methods (`MixedDensityMatrix_for_vbias`, `vbias`, `vHS`) also still delegate. Until 2b is done,
local-energy propagation mode continues to use analytic NOMSD overlaps and energies even though
hybrid mode already consumes the Phase 2a `Overlap` override.

**All other Tier 1–6 visitor methods except `Overlap` still delegate to `nomsd_` (outer) on outer walkers**, so
outer-facing behavior remains identical to plain `NOMSD` in the **single-determinant** delegate
limit (`inner_nwalkers = 1`, `inner_nsteps = 0`) for those methods. Hybrid propagation already
uses the Phase 2a `Overlap` override.

Inner `NOMSD` is evaluated directly on `inner_wset()` via `inner_wfn()` / `inner_nomsd()` for
testing and future overrides. The inner propagator is accessed via `inner_propagator()` once
Phase 2b+ begins driving inner evolution.

---

## Inner stack architecture (Phase 1c)

Phase 1c refactored the inline `inner_nomsd_` member into a **`StochasticInnerStack`**
abstraction because the inner `Propagator` must hold stable references to a `Wavefunction&`
and a device RNG pointer. `StochasticWfn` is moved into the outer `Wavefunction` variant after
construction; heap allocation keeps those references valid.

### Type layout

```
StochasticWfn
├── nomsd_                          (outer delegate; value member)
├── inner_ensemble_                 (WalkerSet + walker-init RNG; Phase 1a)
└── inner_stack_  ──► StochasticInnerStackImpl  (in WavefunctionFactory.cpp)
                      ├── wfn_      unique_ptr<Wavefunction>  → inner NOMSD
                      ├── prop_     unique_ptr<Propagator>     → bound to wfn_, rng_
                      └── rng_      unique_ptr<DeviceRandomGenerator_t>
```

| Component | Role |
|-----------|------|
| `StochasticInnerStack` | Abstract interface in `StochasticWfn.hpp` (`nomsd()`, `wavefunction()`, `propagator()`). Keeps `Propagator.hpp` out of the header (circular include with `Wavefunction.hpp`). |
| `StochasticInnerStackImpl` | Concrete implementation in `WavefunctionFactory.cpp` (anonymous namespace). |
| `inner_ensemble_.rng` | `RandomGenerator_t` used only when constructing/resizing `inner_wset()`. Separate from the propagator RNG. |
| `stack->rng_` | `DeviceRandomGenerator_t` passed to `PropagatorFactory::buildPropagator`. Rank-decorrelated from `inner_seed`. |

### Why the inner NOMSD is wrapped in `Wavefunction`

`PropagatorFactory::buildPropagator` takes a `Wavefunction&`, not a typed `NOMSD&`. The inner
NOMSD is therefore stored inside a heap-allocated `Wavefunction` variant. `inner_wavefunction()`
exposes that wrapper; `inner_nomsd()` / `inner_wfn()` unwrap to the typed `NOMSD` inside it.
`stochastic_inner_propagator_construction` verifies SDetOp pointer identity between the wrapper
and the typed accessor.

### Factory build path

`buildStochasticNomsdWavefunction()` (in `WavefunctionFactory.h`) now:

1. Builds outer/inner `SlaterDetOperations` and `HamiltonianOperations` via the Phase 1b helpers.
2. Calls `buildStochasticInnerStack()` (defined in `WavefunctionFactory.cpp`) to assemble
   `wfn_`, `rng_`, and `prop_`.
3. Passes the finished `unique_ptr<StochasticInnerStack>` into the `StochasticWfn` constructor.

`InnerPropagatorBuilder` is a thin subclass of `PropagatorFactory` that exposes the protected
`buildPropagator` without registering the result in the factory map — the caller owns the
`Propagator` directly.

**Task groups:** inner `HamOps` and the inner propagator both use the same `(TGprop, TGwfn)`
pair as the outer driver build. Propagator-side Cholesky-vector distribution and `vMF`
reductions assume `TGprop` geometry.

**Destruction order:** in `StochasticInnerStackImpl`, `prop_` is declared after `wfn_` and
`rng_` so the propagator (which references both) is destroyed first.

---

## Phase 2 overview: `Overlap` (2a) vs `Energy` (2b)

Phase 2 splits the first stochastic reductions on the propagator hot path into two independent
overrides. They share the same static-ensemble limit (`inner_nsteps = 0`, `B̂_T = 𝟙`) and the
same inner/outer walker geometry, but they answer different questions and integrate with
different propagator modes.

| | **Phase 2a — `Overlap`** | **Phase 2b — `Energy`** |
|---|--------------------------|-------------------------|
| **Status** | **Complete** (CPU-verified) | **Next** (still delegates to `nomsd_`) |
| **Quantity** | Effective trial overlap `Ov[w]` per outer walker | Local energy `E[w]` (E1, EXX, EJ) and overlap `Ov[w]` per outer walker |
| **Reduction** | Cross overlaps `(1/P) Σ_p ⟨ψ_p\|φ_w⟩` between inner walkers `ψ_p` and outer walkers `φ_w` | Average inner `NOMSD::Energy` contributions from `inner_nomsd()` on `inner_wset()`, mapped to each outer walker |
| **Propagator consumer** | **Hybrid** path: `wfn.Overlap(wset, new_overlaps)` in `AFQMCBasePropagator::step` | **Local-energy** path: `wfn.Energy(wset, new_energies, new_overlaps)` |
| **Implementation** | `StochasticWfn.icc` — uses outer `SDetOp`, `FairDivideBoundary` over `nw×P` pairs | Not yet implemented; should mirror the 2a reduction pattern (inner ensemble → per-outer-walker effective quantity) |
| **Phase 3 coupling** | Absolute overlap only; phase factor / importance reweighting (Eq. 25–26) deferred to leapfrog | Same; inner propagation and reweighting arrive with Phase 3 |

At the single-determinant delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`) both
overrides must reproduce plain `NOMSD`. Phase 2a is verified there; Phase 2b will need its
own test once implemented (`stochastic_wfn_matches_nomsd` currently passes because `Energy`
still delegates and overwrites `OVLP` after `Overlap` in some test paths).

---

## Stochastic overlap (Phase 2a)

`StochasticWfn::Overlap` is the first Tier 1 method to diverge from the `nomsd_` delegate. It
computes the **effective trial overlap** of each outer driver walker against the inner ensemble,
following Eq. 24 of [arXiv:2505.18519](https://arxiv.org/abs/2505.18519) ("Implementing advanced
trial wave functions in fermion quantum Monte Carlo via stochastic sampling", Xiao *et al.*,
J. Chem. Phys. **163**, 164109 (2025)).

### Paper formulation

The stochastic trial is an integral over auxiliary fields,
`|Ψ_T⟩ = ∫ dY p_T(Y) B̂_T(Y) |φ_T⟩` (Eq. 21), with `|φ_T⟩` a **single** Slater determinant and
`B̂_T(Y)` an exponential one-body propagator. Each sampled field path `Y^[p]` gives a single
determinant `B̂_T(Y^[p])|φ_T⟩`. The effective bra (Eq. 24) is a phase-weighted, importance-sampled
reduction over `P` paths; the quantity that drives the walk is the **overlap ratio** (Eq. 25-26),
in which the per-walker normalization `𝒩(φ)` cancels.

### What Phase 2a implements (static-ensemble limit)

At `inner_nsteps = 0` there is no field sampling: `B̂_T = 𝟙`, so the phase factor `S(Y^[p])` and
importance reweighting are degenerate and Eq. 24 collapses to a plain average. The implemented
reduction over the `P = inner_nwalkers` inner walkers `ψ_p` (`inner_wset()`) is

```
Ov[w] = (1/P) Σ_p ⟨ψ_p | φ_w⟩
```

where `⟨ψ_p|φ_w⟩` is a single-determinant overlap between an **inner**-walker and an **outer**-walker
Slater matrix. This is *not* `inner_nomsd().Overlap(inner_wset())` (inner-trial-vs-inner-walkers, the
Phase 1b sanity check) — it is a **cross** overlap between the two walker sets.

Implementation notes (`StochasticWfn.icc`):

- Each `⟨ψ_p|φ_w⟩` uses `SlaterDetOperations::Overlap(ψ_p, φ_w, LogOverlapFactor, herm=false)`, which
  forms `det(ψ_p^† φ_w)` — the same convention and `LogOverlapFactor` as `NOMSD::Overlap` (which passes
  the pre-transposed trial with `herm=true`). CLOSED squares the alpha overlap; COLLINEAR multiplies
  alpha and beta; NONCOLLINEAR uses the single packed matrix — mirroring the three
  `NOMSD::Overlap_shared` branches, with the CI weight `conj(ci[nd])` replaced by `1/P`.
- The `nw*P` (outer, inner) pairs are split across the local task group via `FairDivideBoundary`, then
  summed with `all_reduce` over `TG_local`. A new `DeviceBufferManager buffer_manager` member backs the
  transient overlap vector in the `Overlap(wset)` overload (mirrors NOMSD). The outer `nomsd_`'s
  `SlaterDetOperations` and `TG_` (= outer `TGwfn`) are reused.

### Why the visitor `Overlap` is overridden directly, and what stays in Phase 3

The hybrid propagation path (`AFQMCBasePropagator::step`) consumes the overlap **only as a step-to-step
ratio** `ratioOverlaps = new/old` in `hybrid_walker_update`: its argument drives the cosine (phase)
constraint and its log feeds the hybrid energy, and the stored `OVLP` property is overwritten each step.
This is exactly Eq. 25's structure, so overriding the visitor `Overlap` is the correct integration point.
**The exact `𝒩(φ)` cancellation of Eq. 25, however, requires numerator and denominator to share the inner
field samples conditioned on the old walker** — a coupling that only exists in the paper's
propagate-then-resample *leapfrog*, which lives in the propagator hot path. So the phase factor `S(Y)`,
importance sampling, and the leapfrog are deferred to **Phase 3**; `Overlap` itself just produces the
absolute reduction above, which is exact at `inner_nsteps = 0`.

### Delegate limit and the multi-determinant caveat

At the single-determinant delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`, inner walker = the
trial determinant) the reduction is exactly `⟨φ_T|φ_w⟩` = the NOMSD overlap. **This holds only for
single-determinant trials** (RHF/UHF files): the paper's anchor `|φ_T⟩` is a single determinant, and the
inner walkers are initialized from `getInitialGuess` (`Psi0`, the dominant determinant). For a
multi-determinant trial (e.g. `wfn_msd.h5`) the single-determinant inner ensemble cannot reproduce the
CI-weighted NOMSD overlap, so the overridden `Overlap` diverges from NOMSD there by design. The
regression anchor `stochastic_wfn_matches_nomsd` is unaffected because it calls `Energy` after `Overlap`
and `NOMSD::Energy(wset)` overwrites the `OVLP` property; the dedicated `stochastic_overlap_matches_nomsd`
test exercises `Overlap` in isolation and gates the NOMSD-equality assertion on `ndet == 1`.

---

## Stochastic local energy (Phase 2b — planned)

Phase 2b overrides `Energy(wset)` / `Energy(wset, E, Ov)` to return a **stochastic local
energy** (and consistent overlap) for each outer walker, reduced from the inner ensemble in the
same static limit as Phase 2a.

### Target reduction (static-ensemble limit)

At `inner_nsteps = 0`, evaluate `inner_nomsd().Energy` on the inner walker set and reduce to
per-outer-walker `E[w]` (E1, EXX, EJ components) and `Ov[w]`, using the same `(1/P)` averaging
philosophy as Phase 2a. The exact cross-walker indexing (which inner walkers pair with which
outer walkers) must stay consistent with the Phase 2a overlap reduction so hybrid and
local-energy modes do not disagree in the delegate limit.

### Why 2b is separate from 2a

- **Different propagator entry points:** hybrid mode never calls `Energy` for the step update;
  local-energy mode calls `Energy(wset, E, Ov)` and does not use the Phase 2a `Overlap` result
  for weights.
- **Different outputs:** `Overlap` returns a single complex vector; `Energy` fills energy
  components and may also write overlaps used by `local_energy_walker_update`.
- **Independent testability:** Phase 2a is verified via `stochastic_overlap_matches_nomsd`
  (no following `Energy` call). Phase 2b will need a dedicated energy parity test; the existing
  `stochastic_wfn_matches_nomsd` anchor exercises `Energy` but currently passes only because
  `Energy` still delegates to `nomsd_`.

### Deferred to Phase 3 (same as 2a)

Phase factor `S(Y)`, importance reweighting (Eq. 25–26), and driving `inner_propagator()` for
`inner_nsteps > 0` arrive with the propagator leapfrog. The `inner_nsteps > 0` parse guard stays
until Phase 2b (and Phase 3) wire inner propagation into these overrides.

---

## Input keys

Stochastic-specific keys are stripped from plain `NOMSD` input via `strip_stochastic_input_keys()`
(single source of truth in `StochasticWfn.hpp`; `WavefunctionFactory::strip_stochastic_factory_keys`
delegates to it). All stochastic keys require `stochastic: true` in `WavefunctionFactory::interpret_inputs`.

| Key | Default | Phase | Description |
|-----|---------|-------|-------------|
| `stochastic` | `false` | — | Selects `StochasticWfn` instead of plain `NOMSD` on the HDF5 NOMSD path. |
| `inner_nwalkers` | `1` | 1a | Size of the owned inner `WalkerSet`. Must be ≥ 1. |
| `inner_nsteps` | `0` | 1c | Parsed and stored. **`> 0` aborts** at `interpret_inputs` until Phase 2b+ enables inner propagation. |
| `inner_seed` | `777` | 1c | Seed for the inner propagator device RNG. `0` selects a time-based seed (same convention as the driver `seed`). Rank-decorrelated via `split_seed`. |
| `inner_propagator` | *(optional subtree)* | 1c | Propagator input block for `PropagatorFactory`. If omitted, factory defaults apply. `system` and `name` are injected when missing (`name` suffix `_inner_propagator`). |

Example wavefunction block fragment:

```json
{
  "name": "wfn0",
  "type": "nomsd",
  "stochastic": true,
  "inner_nwalkers": 3,
  "inner_nsteps": 0,
  "inner_seed": 777,
  "filename": "wfn.h5",
  "system": "default"
}
```

Optional explicit inner propagator settings:

```json
"inner_propagator": {
  "type": "afqmc",
  "timestep": 0.01
}
```

---

## Factory integration

- Selected via `stochastic: true` inside the existing `type == "nomsd"` path in
  `WavefunctionFactory::fromHDF5`.
- Reads the same NOMSD HDF5 data (`Wavefunction/NOMSD`, `PsiT_*`, CI coefficients, etc.).
- When `stochastic: true`, `fromHDF5` builds a `NomsdSdetPair` upfront and routes through
  `buildStochasticNomsdWavefunctionWithPrecision()` (separate from the plain `NOMSD` path).
- Plain `NOMSD` builds use `buildNomsdWavefunctionWithPrecision()`; the `stochastic` flag is
  not threaded through those templates.
- Registered as four explicit instantiations in the `Wavefunction` `boost::variant`.
- **Two-phase inner-walker init:** (1) construct `StochasticWfn` (inner propagator built here);
  (2) call `maybe_initialize_stochastic_inner_walkers()` from `WavefunctionFactory` (drivers
  and tests) once `getInitialGuess()` and the outer-walker input block are available.

**Dual-bundle and inner-stack factory helpers** (`WavefunctionFactory.h` / `.cpp`):

| Helper | Role |
|--------|------|
| `NomsdSdetPair` / `makeOuterInnerSlaterDetOperations()` | Two independent `SlaterDetOperations` instances (same layout flags). |
| `NomsdHamOpsPair` / `makeOuterInnerHamOps()` | Two independent `getHamOps()` builds (in-place brace init; `HamOps` is move-only). |
| `clone_orbitals()` | Deep copy of CI/orbital vectors for the inner `NOMSD`. |
| `buildStochasticInnerStack()` | Assembles inner `Wavefunction` + device RNG + `Propagator` (`.cpp` only; avoids `Propagator.hpp` in header). |
| `buildStochasticNomsdWavefunction*()` | Assembles `StochasticWfn` with outer infrastructure + pre-built inner stack. |
| `buildNomsdWavefunction*()` | Plain `NOMSD` only (no stochastic branching). |

This is intentionally **not** first-class like `PHMSD` (no separate HDF5 type, no dedicated
factory branch). That remains deferred to Phase 8.

### Accessors

| Accessor | Role |
|----------|------|
| `inner_wset()` | Owned inner `WalkerSet` (stochastic trial ensemble) |
| `inner_nwalkers()` / `inner_nsteps()` | Parsed ensemble size and step count |
| `inner_wfn()` / `inner_nomsd()` | Inner `NOMSD` inside `inner_stack_` |
| `inner_wavefunction()` | Inner `NOMSD` as `Wavefunction&` (object the propagator binds to) |
| `inner_propagator_built()` / `inner_propagator()` | Inner `Propagator` (always built at construction; dormant when `inner_nsteps = 0`) |
| `outer_nomsd()` | Outer delegate `NOMSD` (same object as `nomsd_`) |
| `inner_walkers_initialized()` | Whether phase-2 walker init has run |

`Wavefunction` variant wrappers: `is_stochastic_wavefunction()`,
`stochastic_inner_walkers_initialized()`, `stochastic_inner_wset()`, and
`initialize_stochastic_inner_walkers()`.

**Outer-facing infrastructure today:** `getSlaterDetOperations()` and all Tier 1–6 methods use
`nomsd_` (outer). Inner `HamOps`/`SDetOp` live inside `inner_stack_->nomsd()` and are not
exposed through the public `StochasticWfn` facade yet (Phase 7 may realign layout queries and
SDetOp access).

---

## Target Architecture

> **Current and future expectation:** Each `StochasticWfn` instance owns its own inner AFQMC stack:
>
> - **Walker set** — stochastic ensemble defining the trial wavefunction *(Phase 1a ✓)*
> - **NOMSD** — analytic MSD structure on inner walkers, with its own HamOps/SDetOp *(Phase 1b ✓)*
> - **Propagator** — built and bound at construction; drives inner evolution once Phase 2b+
>   enables `inner_nsteps > 0` and overrides consume it *(Phase 1c ✓ infrastructure; invocation deferred)*
> - **Effective outer quantities** — overlap *(Phase 2a ✓)*, local energy *(Phase 2b)*, mixed DM, bias potentials reduced
>   from the inner ensemble *(Phases 3–5)*
>
> The outer AFQMC driver will call `StochasticWfn` methods as it does for `NOMSD`/`PHMSD` today;
> `StochasticWfn` will orchestrate the inner calculation and return effective quantities
> (overlap, local energy, mixed density matrix, bias potentials) to the outer propagator.

---

## Methods to Override

The `Wavefunction` visitor dispatches all calls below. Any method not listed still delegates to
`nomsd_` today and may need overriding once stochastic outputs diverge from the delegate limit.

Methods are grouped by priority. **Tier 1** methods are on the propagator hot path (every
outer timestep). These are the hardest and most important to replace.

---

### Tier 1 — Propagator hot path (highest priority)

Called every outer propagation step from `AFQMCBasePropagator`, `AFQMCModelPropagator`, and
distributed propagator variants.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `MixedDensityMatrix_for_vbias(wset, G)` | Delegates to `nomsd_`; computes analytic mixed DM of outer walkers w.r.t. the deterministic MSD trial. | Compute the **stochastic** mixed density matrix used by `vbias`. Run (or sample from) the inner walker/propagator ensemble, evaluate inner `NOMSD` mixed DMs, and return an effective DM in the layout expected by `vbias` (`compact_G_for_vbias`, `transposed_G_for_vbias`). |
| `vbias(G, v, dt, a)` | Delegates to `nomsd_.HamOp.vbias` using the analytic `G` from above. | Apply Cholesky-vector bias using the **stochastic** mixed DM. May call inner `HamOps.vbias` on inner-walker quantities and reduce/average to the outer-walker buffer layout. |
| `vHS(X, v, dt, a)` | Delegates to `nomsd_.HamOp.vHS`. | Compute the spin-dependent or spin-independent one-body propagation matrix contribution from auxiliary fields, using inner stochastic information where the trial is not a single deterministic MSD. |
| `Overlap(wset)` / `Overlap(wset, Ov)` | **Phase 2a ✓ (static limit):** `Ov[w] = (1/P) Σ_p ⟨ψ_p\|φ_w⟩`, a cross overlap between inner walkers `ψ_p` and each outer walker `φ_w` (Eq. 24 of arXiv:2505.18519 with `B̂_T = 𝟙`). Matches NOMSD at the single-determinant delegate limit; diverges for multi-determinant trials by design. Consumed by **hybrid** propagation. | Add the phase factor `S(Y)` / importance reweighting (Eq. 25-26) once the Phase 3 inner-propagation leapfrog samples auxiliary fields; relax the `inner_nsteps > 0` guard. |
| `Energy(wset)` / `Energy(wset, E, Ov)` | **Phase 2b (planned):** still delegates to `nomsd_`; exact local energy from analytic MSD. | Return **stochastic local energy** and overlap: evaluate `inner_nomsd().Energy` on `inner_wset()` and reduce to per-outer-walker `E` (E1, EXX, EJ) and `Ov`, consistent with the Phase 2a overlap reduction. Consumed by **local-energy** propagation. Inner propagation / reweighting deferred to Phase 3. |

---

### Tier 2 — Mixed estimators and observable handlers

Called from `MixedObsHandler`, `FullObsHandler`, force estimators, and related code paths.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `MixedDensityMatrix(wset, G, ...)` | Delegates to `nomsd_`; full analytic mixed DM for outer walkers. | Stochastic mixed DM for **observable evaluation** (may differ in layout/options from the `for_vbias` variant). Average inner ensemble contributions; support `compact` and `transpose` flags as today. |
| `MixedDensityMatrix(wset, G, Ov, ...)` | Same as above, also returns overlaps per determinant/walker. | As above, plus return the overlap vector needed for weighted accumulation in multi-determinant or multi-reference estimators. |
| `DensityMatrix(wset, RefA, RefB, G, Ov, ...)` | Delegates to `nomsd_`; DM w.r.t. a specific reference determinant. | Stochastic DM relative to a chosen reference orbital set. Needed by `MixedObsHandler` for reference-resolved force and density estimators. |
| `accumulate_estimators(...)` | Delegates to `nomsd_`; analytic back-prop / correlated accumulation. | Accumulate estimator contributions using **stochastic** DMs and energies from the inner ensemble. Must remain consistent with `TimeEvolvedObsHandler` call signature. |

---

### Tier 3 — Mean-field subtraction

Called during propagator setup and mean-field initialization.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `G_MF(G)` | Delegates to `nomsd_`; builds mean-field Green's function from CI-weighted determinant DMs. | Build the **stochastic** mean-field Green's function used to subtract the trial mean-field potential. Average inner-walker MF contributions from the owned inner stack. |
| `vMF(v, dt)` | Delegates to `nomsd_`; mean-field contribution of Cholesky vectors. | Compute mean-field bias from the stochastic trial. May require inner `HamOps` and inner walker sampling. |

---

### Tier 4 — Layout and metadata queries

Currently forwarded from **outer** `nomsd_` / its `HamOp`. Inner `inner_stack_->nomsd()` has
matching layout metadata (verified in Phase 1b/1c tests) but is not yet the source of
outer-facing queries. Must remain **consistent** with what the overridden Tier 1–3 methods
actually produce; may need custom logic once inner propagation and stochastic outputs diverge
from the delegate limit.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `size_of_G_for_vbias()` | Returns `nomsd_` DM dimension for `vbias` layout. | Return the DM dimension the stochastic `MixedDensityMatrix_for_vbias` / `vbias` pair actually uses. |
| `transposed_G_for_vbias()` | From outer `HamOp` flags. | Match the memory layout of stochastic `G` passed to `vbias`. |
| `transposed_G_for_E()` | From outer `HamOp` flags. | Match the memory layout of stochastic `G` used in energy evaluation. |
| `transposed_vHS()` | From outer `HamOp` flags. | Match the memory layout expected by `vHS`. |
| `getWalkerType()` | Returns outer `NOMSD` walker type. | Return the walker type the outer driver should use (likely unchanged, but must stay consistent with inner ensemble). |
| `local_number_of_cholesky_vectors()` | From outer `HamOp`. | Return local CV count from **inner** `HamOps` once owned by `StochasticWfn`. |
| `global_number_of_cholesky_vectors()` | From outer `HamOp`. | Global CV count for MPI distribution; must match inner Hamiltonian. |
| `global_origin_cholesky_vector()` | From outer `HamOp`. | Global CV offset for this task group. |
| `distribution_over_cholesky_vectors()` | From outer `HamOp`. | Whether CVs are distributed across the task group. |
| `spin_dependent_vHS()` | From outer `HamOp`. | Whether `vHS` expects spin-resolved output buffers. |

---

### Tier 5 — Hamiltonian and Slater infrastructure

Outer-facing calls delegate to `nomsd_`. Inner `HamOps`/`SDetOp` are owned inside
`inner_stack_->nomsd()` (Phase 1b); public accessors still return outer infrastructure until
Phase 7.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `getHamType()` | Returns `nomsd_.HamOp.getHamType()`. | Return Hamiltonian type from **inner** `HamOps`. |
| `getFieldTypes(...)` | Delegates to outer `HamOp`. | Expose auxiliary-field layout for inner Hamiltonian (used by `PropagatorFactory`). |
| `update_potentials(...)` | Delegates to outer `HamOp`. | Update inner Hamiltonian potentials after grid / ionic moves. |
| `generalizedFockMatrix(...)` | Delegates to outer `HamOp`. | Used by `generalizedFockMatrix` observable; must use stochastic DM inputs when overridden. |
| `getOneBodyPropagatorMatrix(...)` | Delegates to outer `HamOp`. | Return one-body propagator matrix from inner Hamiltonian. |
| `vHS_sparse(...)` | Delegates to outer `HamOp`. | Sparse `vHS` representation for memory-efficient propagation. |
| `getSlaterDetOperations()` | Returns `nomsd_` SlaterDetOperations pointer. | Return Slater determinant ops for the **inner** `NOMSD` (or a dedicated instance owned by `StochasticWfn`). |

---

### Tier 6 — Back propagation

Called from `BackPropagatedEstimator` and related reference-tracking code.

| Method | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `number_of_references_for_back_propagation()` | Returns `nomsd_` reference count. | Number of reference Slater matrices the stochastic trial exposes for back-propagation (may differ if inner ensemble uses a reduced reference set). |
| `getReferenceWeight(i)` | Returns `nomsd_.ci[i]`. | CI weight (or effective stochastic weight) for reference `i`. |
| `getReferencesForBackPropagation(A)` | Copies reference orbitals from `nomsd_`. | Provide reference Slater matrices for back-propagation, possibly averaged or selected from the inner ensemble. |

---

### Tier 7 — Construction and factory (infrastructure)

Not wavefunction visitor methods, but required for a full-fledged type.

| Component | Current behavior | Desired functionality |
|--------------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `StochasticWfn` constructor | Builds outer `nomsd_`; accepts pre-built `StochasticInnerStack` (inner NOMSD + propagator + RNG). Parses `inner_nwalkers`, `inner_nsteps`, `inner_seed`, `inner_propagator`. Defers inner `WalkerSet` resize. | Population control, first-class HDF5 type (Phase 8). |
| `interpret_inputs(pt)` | Validates NOMSD keys plus all stochastic keys listed above. Rejects `inner_nsteps > 0`. | Relax `inner_nsteps` guard when Phase 2b invokes `inner_propagator()`. |
| `WavefunctionFactory` | `stochastic: true` on NOMSD HDF5 path; `buildStochasticInnerStack()` + `buildStochasticNomsdWavefunction*`; `maybe_initialize_stochastic_inner_walkers()` after build. | First-class `stochasticwfn` type with its own `fromHDF5` branch (Phase 8). |
| `getWavefunctionType()` | Not aware of `StochasticWfn`. | Detect stochastic trial wavefunction files on disk. |

---

## Suggested Implementation Order

Step 1 in the original plan ("own the full inner stack") bundles four heavyweight concerns —
walker set, `NOMSD`, `HamOps`, and propagator — each with its own factory wiring and test
surface. **Do not implement Step 1 as a single monolith.** Decompose it into the phases below.

### Phase 1a — Inner walker ensemble skeleton (**complete**)

**Goal:** Introduce the two-walker-set model (outer driver walkers vs. inner trial ensemble)
with no change to outer-facing behavior.

| Item | Status |
|------|--------|
| Members | `StochasticInnerEnsemble` with `unique_ptr<WalkerSet>`, RNG, and init flag. |
| Inputs | `inner_nwalkers` (default `1`) in `StochasticWfn` and `WavefunctionFactory` input. |
| Initialization | Two-phase init via `initialize_inner_walkers()` / `maybe_initialize_stochastic_inner_walkers()` in factory and all AFQMC drivers. |
| Public API | Tier 1–6 methods still delegate to `nomsd_` on outer walkers. Accessors `inner_wset()`, `inner_wfn()` / `inner_nomsd()`, `outer_nomsd()`. |
| Task group | Same `TGwfn` as inner `NOMSD`. |
| Tests | `stochastic_wfn_matches_nomsd`, `stochastic_inner_walkers_init`, `stochastic_inner_walkers_uninitialized_smoke` in `test_wfn_factory.cpp`. |

### Phase 1b — Independent inner `HamOps` and `SDetOp` (**complete**)

**Goal:** Duplicate the factory's `getHamOps()` + `SlaterDetOperations` build for a second inner
`NOMSD` so the inner stack is truly independent of the outer delegate. `HamOps` still lives
inside each `NOMSD`; Phase 1b adds a second `NOMSD` with its own moved-in infrastructure rather
than splitting `HamOps` out of `NOMSD` globally.

| Item | Status |
|------|--------|
| Members | `nomsd_` (outer delegate) + inner `NOMSD` (now inside `inner_stack_`). |
| Constructor | Outer SDetOp/HamOps moved into `nomsd_`; inner bundles assembled in `buildStochasticInnerStack()`. |
| Accessors | `inner_nomsd()`, `outer_nomsd()`; `inner_wfn()` aliases inner. |
| Factory | `makeOuterInnerSlaterDetOperations`, `makeOuterInnerHamOps`, `buildStochasticNomsdWavefunction*`; plain path unchanged. |
| Outer behavior | Tier 1–6 still delegate to `nomsd_`; `getSlaterDetOperations()` returns outer SDetOp. |
| Tests | `stochastic_inner_outer_infrastructure_independent` in `test_wfn_factory.cpp`. |

**Verified:** distinct outer/inner `NOMSD` and `SlaterDetOperations` pointers; matching layout
metadata (Cholesky counts, transpose flags, `size_of_G_for_vbias`, walker/Ham types); inner
overlap/energy on `inner_wset()` matches plain `NOMSD` at delegate limit.

### Phase 1c — Inner propagator (static default) (**complete**)

**Goal:** Own an inner `Propagator` wired through `PropagatorFactory` at wavefunction build
time. Default `inner_nsteps = 0` (no propagation; static ensemble). The propagator exists and
is testable but is not called from `StochasticWfn` methods until Phase 2b+.

| Item | Status |
|------|--------|
| Architecture | `StochasticInnerStack` / `StochasticInnerStackImpl`; heap `Wavefunction` + `Propagator` + device RNG. |
| Inputs | `inner_nsteps` (default `0`, `> 0` rejected), `inner_seed` (default `777`), optional `inner_propagator` subtree. |
| Key stripping | `strip_stochastic_input_keys()` centralizes stochastic key removal. |
| Factory | `buildStochasticInnerStack()`, `InnerPropagatorBuilder`; propagator built with `TGprop` + inner `wavefunction()` + `rng_`. |
| Accessors | `inner_wavefunction()`, `inner_propagator_built()`, `inner_propagator()`, `inner_nsteps()`. |
| Outer behavior | Unchanged; delegate limit (`inner_nwalkers = 1`, `inner_nsteps = 0`) preserved. |
| Tests | `stochastic_inner_propagator_construction` in `test_wfn_factory.cpp`. |

**Verified:** factory constructs without error; inner propagator bound to inner `Wavefunction`;
Cholesky layout matches inner `NOMSD`; default hybrid propagation settings; all `[stochastic_wfn]`
tests pass on CPU build with `wfn_msd.h5`.

**Still deferred (post-1c):** invoking `inner_propagator()` from overrides, enabling
`inner_nsteps > 0`, routing layout/SDetOp queries through inner stack where appropriate (Phase 7).

### Phase 2a — Stochastic `Overlap` (**implemented; pending build + test**)

**Goal:** Override `Overlap` to reduce the inner ensemble into an effective stochastic-trial
overlap per outer walker. First Tier 1 method to diverge from the `nomsd_` delegate.

| Item | Status |
|------|--------|
| Method | `Overlap(wset)` / `Overlap(wset, Ov)` in `StochasticWfn.icc` |
| Reduction | `Ov[w] = (1/P) Σ_p ⟨ψ_p\|φ_w⟩` (Eq. 24, static limit); see [Stochastic overlap (Phase 2a)](#stochastic-overlap-phase-2a) |
| Infrastructure | `DeviceBufferManager buffer_manager`; outer `SDetOp` + `TG_`; `FairDivideBoundary` over `nw×P` |
| Propagator | **Hybrid** mode consumes this override directly |
| Tests | `stochastic_overlap_matches_nomsd` |

**Pending:** CPU build and test; see [Phase 2a-specific tests](#phase-2a-specific-tests-implemented).

### Phase 2b — Stochastic `Energy` (**next**)

**Goal:** Override `Energy` to reduce inner-ensemble local energies to per-outer-walker `E` and
`Ov`, consistent with the Phase 2a overlap reduction. Unblocks correct **local-energy**
propagation; hybrid mode already uses Phase 2a for overlaps.

| Item | Status |
|------|--------|
| Method | `Energy(wset)` / `Energy(wset, E, Ov)` — still delegates to `nomsd_` |
| Reduction | Evaluate `inner_nomsd().Energy` on `inner_wset()`, average to each outer walker; see [Stochastic local energy (Phase 2b)](#stochastic-local-energy-phase-2b--planned) |
| Propagator | **Local-energy** mode: `wfn.Energy(wset, new_energies, new_overlaps)` in `AFQMCBasePropagator::step` |
| Tests | `stochastic_wfn_matches_nomsd` passes today only because `Energy` still delegates; needs dedicated 2b parity test once implemented |
| Guard | `inner_nsteps > 0` parse abort stays until Phase 2b/3 wire inner propagation |

**Still deferred (post-2a):** Phase 2b `Energy` override; invoking `inner_propagator()` from
overrides; enabling `inner_nsteps > 0`; phase factor / importance reweighting (Phase 3).

### Phase 3 — `MixedDensityMatrix_for_vbias` and `vbias`

Unblock correct outer propagation (hardest coupling).

### Phase 4 — `vHS`

Complete the per-step propagator chain.

### Phase 5 — `MixedDensityMatrix` / `DensityMatrix` / `accumulate_estimators`

Mixed and back-propagated observables.

### Phase 6 — `G_MF` / `vMF`

Mean-field subtraction consistency.

### Phase 7 — Layout queries and back-prop references

Align metadata with actual stochastic outputs.

### Phase 8 — Factory / HDF5 first-class treatment

Remove the `stochastic` boolean flag hack; dedicated `fromHDF5` branch like `PHMSD`.

### What to avoid as an early step

- **Overriding `Overlap`/`Energy` before 1a** — without an owned inner ensemble, this only
  re-wraps `nomsd_` with extra indirection.
- **Building the inner propagator before inner walkers exist** — highest coupling, hardest to
  test, and useless until `Overlap`/`Energy` consume the ensemble. *(Phase 1c builds the
  propagator at wavefunction construction time, but does not invoke it; walker init remains
  the separate two-phase step from 1a.)*
- **Splitting `HamOps` out of `NOMSD` globally** — a `NOMSD` refactor, not a `StochasticWfn`
  feature; Phase 1b instead duplicates full `NOMSD` instances with separate moved-in `HamOps`.

---

## Validation Checkpoints

### Delegate limit (primary regression anchor)

Compare against the current `nomsd_` delegate in the limit where the inner ensemble collapses
to a single deterministic state (`inner_nwalkers = 1`, `inner_nsteps = 0`):

- `stochastic_wfn_matches_nomsd` continues to pass unchanged at default inputs.
- `stochastic_inner_outer_infrastructure_independent` confirms dual infrastructure without breaking delegate-limit observables.
- `stochastic_inner_propagator_construction` confirms inner propagator wiring without changing outer behavior.
- `stochastic_overlap_matches_nomsd` confirms the overridden `Overlap` matches NOMSD at the single-determinant delegate limit and is invariant to `inner_nwalkers` in the static limit *(pending CPU verification)*.
- Outer propagator completes without layout/runtime check failures *(integration follow-up)*.
- Local energy and overlap match analytic `NOMSD` within stochastic error bars.
- Mixed estimator (`MixedObsHandler`) forces and densities are consistent *(integration follow-up)*.
- GPU memory and task-group load remain acceptable with the additional inner ensemble *(integration follow-up)*.

### Phase 1a-specific tests (implemented)

In `test_wfn_factory.cpp` / `WfnTestContext`:

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_wfn_matches_nomsd` | Parity preserved at default inputs (delegate limit). |
| `stochastic_inner_walkers_init` | Inner walkers initialize to `inner_nwalkers`; Slater matrices match `getInitialGuess()`; inner `NOMSD` overlap/energy on `inner_wset()` matches reference; replicated ensemble gives identical observables. |
| `stochastic_inner_walkers_uninitialized_smoke` | Stochastic wavefunction built without `walker_pt` stays uninitialized until `maybe_initialize_stochastic_inner_walkers()`; then `stochastic_inner_wset()` is usable. (`inner_wset()` before init calls `APP_ABORT` — not exercised in-process.) |

### Phase 1b-specific tests (implemented)

Requires a **NOMSD** HDF5 input (`getWavefunctionType() == "NOMSD"`); PHMSD inputs skip checks.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_inner_outer_infrastructure_independent` | `outer_nomsd()` and `inner_nomsd()` are distinct objects with distinct `getSlaterDetOperations()` pointers; `StochasticWfn::getSlaterDetOperations()` returns outer; inner/outer layout metadata match (Cholesky distribution, transpose flags, `size_of_G_for_vbias`, walker/Ham types); inner overlap/energy on `inner_wset()` match plain `NOMSD` reference. |

### Phase 1c-specific tests (implemented)

Requires a **NOMSD** HDF5 input; PHMSD inputs skip checks.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_inner_propagator_construction` | `inner_nsteps() == 0`; `inner_propagator_built()`; `inner_wavefunction()` shares SDetOp with `inner_nomsd()` but not outer; layout parity between inner NOMSD and wrapped `Wavefunction`; propagator Cholesky count matches inner NOMSD; default hybrid (not free) propagation. |

### Phase 2a-specific tests (implemented)

Requires a **NOMSD** HDF5 input. Checkpoint (2) additionally requires a **single-determinant**
trial (RHF/UHF); it is gated on `ndet == 1`, so on multi-determinant `wfn_msd.h5` only
checkpoint (1) runs.

| Test case | Checkpoint |
|-----------|------------|
| `stochastic_overlap_matches_nomsd` | `Overlap` in isolation (no following `Energy`): (1) **inner_nwalkers invariance**; (2) **delegate limit** — stochastic overlap equals NOMSD at `ndet == 1`. |

Run the delegate-limit anchor with a single-determinant file, e.g.:

```bash
HAMIL=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
WFN=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/wfn_rhf.h5

mpirun -np 1 ./tests/bin/test_afqmc_wavefunctions \
  --hamil "$HAMIL" --wfn "$WFN" \
  stochastic_overlap_matches_nomsd
```

### Phase 2b-specific tests (planned)

Phase 2b will need a test that exercises `Energy` in isolation (or a full
`stochastic_wfn_matches_nomsd`-style parity check once `Energy` no longer delegates). Until
then, `stochastic_wfn_matches_nomsd` validates the delegate limit only for quantities that still
route through outer `nomsd_` (`Energy`, `vbias`, etc.).

### Running the stochastic test suite

Build `test_afqmc_wavefunctions`, then run from the build directory. Stochastic helpers require
a NOMSD wavefunction file (e.g. bundled `wfn_msd.h5`; RHF/UHF/PHMSD inputs skip stochastic
checks silently).

```bash
HAMIL=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
WFN=/path/to/SAFIRE/tests/unit_test_files/C_1x1x1_dzvp/wfn_msd.h5

cd /path/to/SAFIRE/build
mpirun -np 1 ./tests/bin/test_afqmc_wavefunctions \
  --hamil "$HAMIL" --wfn "$WFN" \
  "[stochastic_wfn]"
```

Single test example:

```bash
mpirun -np 1 ./tests/bin/test_afqmc_wavefunctions \
  --hamil "$HAMIL" --wfn "$WFN" \
  stochastic_inner_propagator_construction
```

All stochastic tests share the Catch2 tag `[stochastic_wfn]`.

### Integration follow-ups (not yet validated)

- Outer propagator completes without layout/runtime failures with `stochastic: true`.
- Mixed estimator (`MixedObsHandler`) forces and densities remain consistent.
- GPU build: all `[stochastic_wfn]` tests pass with `ENABLE_CUDA=ON`.
- GPU memory and task-group load acceptable with the additional inner ensemble and propagator.
