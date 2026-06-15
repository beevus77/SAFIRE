////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#ifndef SFQMC_AFQMC_STOCHASTICWFN_H
#define SFQMC_AFQMC_STOCHASTICWFN_H

#include <memory>

#include "io/ptree/ptree_utilities.hpp"
#include "Utilities/Random.hpp"
#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Wavefunctions/NOMSD.hpp"

namespace sfqmc
{
namespace afqmc
{
// Forward declarations: StochasticWfn is itself an alternative of the Wavefunction
// boost::variant (Wavefunction.hpp includes this header), and Propagator.hpp includes
// Wavefunction.hpp, so neither type can be complete here. The inner stack holds them
// behind the StochasticInnerStack interface below; the concrete implementation lives in
// WavefunctionFactory.cpp where both types are complete.
class Wavefunction;
class Propagator;

/*
 * Strips all StochasticWfn-specific keys from a wavefunction input block, leaving a
 * plain-NOMSD input. Single source of truth for the stochastic key set; used by
 * StochasticWfn::nomsd_inputs and WavefunctionFactory::strip_stochastic_factory_keys.
 */
inline ptree strip_stochastic_input_keys(ptree pt)
{
  for (auto const& key : {"stochastic", "inner_nwalkers", "inner_nsteps", "inner_seed", "inner_propagator"})
    pt.erase(key);
  return pt;
}

/*
 * Owner of the inner AFQMC stack of a StochasticWfn: the inner NOMSD (held inside a
 * heap-allocated Wavefunction variant so propagators can bind a stable Wavefunction&),
 * the inner Propagator, and the device RNG the propagator samples from.
 * Heap allocation matters: StochasticWfn is moved into the Wavefunction variant after
 * construction, and the inner Propagator stores references to the inner Wavefunction
 * and RNG — those must not move with it.
 */
template<bool MP, class devPsiT>
struct StochasticInnerStack
{
  virtual ~StochasticInnerStack() = default;

  virtual NOMSD<MP, devPsiT>& nomsd()             = 0;
  virtual NOMSD<MP, devPsiT> const& nomsd() const = 0;

  virtual Wavefunction& wavefunction()             = 0;
  virtual Wavefunction const& wavefunction() const = 0;

  virtual bool has_propagator() const         = 0;
  virtual Propagator& propagator()             = 0;
  virtual Propagator const& propagator() const = 0;
};

/*
 * Stochastic trial wavefunction wrapper.
 * Owns an outer NOMSD delegate (nomsd_) for outer-walker-facing operations and an inner
 * stack (inner_stack_) holding the inner NOMSD — with its own HamOps/SDetOp — wrapped in
 * a Wavefunction, plus the inner Propagator (Phase 1c; static ensemble, inner_nsteps = 0).
 */
template<bool MP, class devPsiT>
class StochasticWfn : public AFQMCInfo
{
  // Buffer managers and work-vector types for the Overlap / Energy reductions (mirror NOMSD).
  // buffer_alloc_type backs per-core transient vectors/matrices (overlaps, energy components);
  // shm_buffer_alloc_type backs the shared mixed density matrix the energy reduction fills.
  using buffer_alloc_type     = DeviceBufferManager::template allocator_t<ComplexType>;
  using shm_buffer_alloc_type = LocalTGBufferManager::template allocator_t<ComplexType>;
  using StaticVector          = boost::multi::static_array<ComplexType, 1, buffer_alloc_type>;
  using StaticMatrix          = boost::multi::static_array<ComplexType, 2, buffer_alloc_type>;
  using StaticSHMVector       = boost::multi::static_array<ComplexType, 1, shm_buffer_alloc_type>;
  using Allocator             = device_allocator<ComplexType>;
  using pointer               = typename std::allocator_traits<Allocator>::pointer;
  using CMatrix_ref           = boost::multi::array_ref<ComplexType, 2, pointer>;
  using CVector_ref           = boost::multi::array_ref<ComplexType, 1, pointer>;

  struct StochasticInnerEnsemble
  {
    std::unique_ptr<WalkerSet> wset;
    utils::RandomGenerator_t rng;
    bool initialized{false};
  };

  afqmc::TaskGroup_& TG_;
  StochasticInnerEnsemble inner_ensemble_;
  int inner_nwalkers_{1};
  int inner_nsteps_{0};
  double inner_timestep_{0.01}; // dt for the inner free-projection B_T step (Phase 3b)
  // Phase 3b latch: armed by begin_inner_step() at the top of each outer propagator step, consumed
  // (cleared) by maybe_advance_inner_ensemble() the first time a reduction runs that step, so the
  // inner ensemble is resampled exactly once per outer step regardless of which reductions are called.
  bool inner_step_pending_{false};
  // Anchor Slater matrices |phi_T> the inner walkers are reset to before each resample (Phase 3b).
  boost::multi::array<ComplexType, 3> inner_anchor_;
  DeviceBufferManager buffer_manager;
  LocalTGBufferManager shm_buffer_manager;
  NOMSD<MP, devPsiT> nomsd_;
  std::unique_ptr<StochasticInnerStack<MP, devPsiT>> inner_stack_;

  static ptree nomsd_inputs(ptree const& pt0)
  {
    return NOMSD<MP, devPsiT>::interpret_inputs(strip_stochastic_input_keys(pt0));
  }

public:
  template<class MType>
  StochasticWfn(AFQMCInfo& info,
                ptree pt_in,
                afqmc::TaskGroup_& tg_,
                SlaterDetOperations&& outer_sdet_,
                HamiltonianOperations<MP>&& outer_hop_,
                std::vector<ComplexType>&& ci_,
                std::vector<MType>&& orbs_,
                std::unique_ptr<StochasticInnerStack<MP, devPsiT>>&& inner_stack_in,
                WALKER_TYPES wlk,
                ComplexType nce,
                [[maybe_unused]] int targetNW = 1)
      : AFQMCInfo(info),
        TG_(tg_),
        buffer_manager(),
        shm_buffer_manager(),
        nomsd_(info, nomsd_inputs(pt_in), tg_, std::move(outer_sdet_), std::move(outer_hop_), std::move(ci_),
               std::move(orbs_), wlk, nce, targetNW),
        inner_stack_(std::move(inner_stack_in))
  {
    if (inner_stack_ == nullptr)
      APP_ABORT("Error in StochasticWfn: inner stack must be provided (see WavefunctionFactory).");
    ptree pt = interpret_inputs(pt_in);
    inner_nwalkers_ = pt.get<int>("inner_nwalkers");
    inner_nsteps_   = pt.get<int>("inner_nsteps");
    // The inner free-projection step's timestep comes from the inner_propagator subtree (the same
    // block PropagatorFactory consumes); default 0.01. Only used when inner_nsteps_ > 0 (Phase 3b).
    // The inner propagator (Propagate -> generateP1) rebuilds its one-body/CV scaling for this dt on
    // first use (old_dt starts unset), so inner_timestep_ is the single effective source -- no drift.
    if (auto prop_pt = pt.get_child_optional("inner_propagator"))
      inner_timestep_ = prop_pt->get<double>("timestep", 0.01);
    // Phase 3b fail-fast: the dynamic ensemble (inner_nsteps_ > 0) scores moving inner walkers via the
    // un-rotated full-G energy/force bias, implemented for CLOSED (RHF) trials on CPU only. Reject early
    // (here) rather than deep in the first reduction's HamOp dispatch.
    if (inner_nsteps_ > 0)
    {
      if (wlk != CLOSED)
        APP_ABORT("Error in StochasticWfn: inner_nsteps > 0 (dynamic stochastic trial) currently supports "
                  "CLOSED (RHF) trials only -- COLLINEAR/NONCOLLINEAR full-G is a follow-up.");
#if defined(ENABLE_DEVICE)
      APP_ABORT("Error in StochasticWfn: inner_nsteps > 0 (dynamic stochastic trial) is CPU-only; the "
                "un-rotated full-G kernels are not yet implemented for device builds.");
#endif
    }
    app_log(2, "\nStochasticWfn input:\n{}\n", io::to_string(pt));
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    ptree pt1 = nomsd_inputs(pt0);
    int inner_nwalkers = pt0.get<int>("inner_nwalkers", 1);
    if (inner_nwalkers < 1)
      APP_ABORT("Error in StochasticWfn::interpret_inputs: inner_nwalkers must be >= 1.");
    int inner_nsteps = pt0.get<int>("inner_nsteps", 0);
    if (inner_nsteps < 0)
      APP_ABORT("Error in StochasticWfn::interpret_inputs: inner_nsteps must be >= 0.");
    // Phase 3b: inner_nsteps > 0 drives the inner free-projection propagator (dynamic ensemble).
    int inner_seed = pt0.get<int>("inner_seed", 777);
    pt1.put("inner_nwalkers", inner_nwalkers);
    pt1.put("inner_nsteps", inner_nsteps);
    pt1.put("inner_seed", inner_seed);
    if (auto prop_pt = pt0.get_child_optional("inner_propagator"))
      pt1.put_child("inner_propagator", *prop_pt);
    std::unordered_set<std::string> pass_through_keys = {
        "system",
        "name",
        "ndets_to_read",
        "restart_file",
        "filename",
        "stochastic",
        "inner_nwalkers",
        "inner_nsteps",
        "inner_seed",
        "inner_propagator",
    };
    io::compare_known_keys("Stochastic trial wavefunction (StochasticWfn)", pt1, pt0, pass_through_keys);
    return pt1;
  }

  ~StochasticWfn() = default;

  StochasticWfn(StochasticWfn const& other)            = delete;
  StochasticWfn& operator=(StochasticWfn const& other) = delete;
  StochasticWfn(StochasticWfn&& other)                   = default;
  StochasticWfn& operator=(StochasticWfn&& other)        = delete;

  void initialize_inner_walkers(ptree const& walker_pt,
                                boost::multi::array<ComplexType, 3> const& initial_guess,
                                int NAEB);

  bool inner_walkers_initialized() const { return inner_ensemble_.initialized; }
  int inner_nwalkers() const { return inner_nwalkers_; }
  int inner_nsteps() const { return inner_nsteps_; }

  WalkerSet& inner_wset();
  WalkerSet const& inner_wset() const;

  NOMSD<MP, devPsiT>& inner_wfn() { return inner_stack_->nomsd(); }
  NOMSD<MP, devPsiT> const& inner_wfn() const { return inner_stack_->nomsd(); }

  NOMSD<MP, devPsiT>& inner_nomsd() { return inner_stack_->nomsd(); }
  NOMSD<MP, devPsiT> const& inner_nomsd() const { return inner_stack_->nomsd(); }

  // Inner NOMSD wrapped as a Wavefunction (the object the inner Propagator is bound to).
  Wavefunction& inner_wavefunction() { return inner_stack_->wavefunction(); }
  Wavefunction const& inner_wavefunction() const { return inner_stack_->wavefunction(); }

  bool inner_propagator_built() const { return inner_stack_->has_propagator(); }
  Propagator& inner_propagator();
  Propagator const& inner_propagator() const;

  // Phase 3b: arm the per-outer-step latch. The outer propagator (AFQMCBasePropagator::step) calls
  // this at the top of each step; the first reduction that runs then resamples the inner ensemble
  // (reset to anchor + inner_nsteps free-projection B_T steps) exactly once. No-op when
  // inner_nsteps_ == 0 (static ensemble) since maybe_advance_inner_ensemble() ignores the latch then.
  void begin_inner_step() { inner_step_pending_ = true; }

  NOMSD<MP, devPsiT>& outer_nomsd() { return nomsd_; }
  NOMSD<MP, devPsiT> const& outer_nomsd() const { return nomsd_; }

  int local_number_of_cholesky_vectors() const { return nomsd_.local_number_of_cholesky_vectors(); }
  int global_number_of_cholesky_vectors() const { return nomsd_.global_number_of_cholesky_vectors(); }
  int global_origin_cholesky_vector() const { return nomsd_.global_origin_cholesky_vector(); }
  bool distribution_over_cholesky_vectors() const { return nomsd_.distribution_over_cholesky_vectors(); }
  bool spin_dependent_vHS() const { return nomsd_.spin_dependent_vHS(); }

  // Phase 3b: once inner walkers leave the anchor (inner_nsteps_ > 0) the True-Ham force bias can no
  // longer use the nd = 0 half-rotated (compact) layout, so MixedDensityMatrix_for_vbias produces the
  // FULL cross G [NMO*NMO][nwalk] (non-transposed) that vbias_fullG contracts with the full Likn. The
  // outer propagator sizes its vbias G buffer from these, so they must advertise the full layout then.
  // At inner_nsteps_ == 0 they stay the outer nomsd_ (compact/half-rotated) layout (Phase 3a, exact).
  int size_of_G_for_vbias() const
  {
    return (inner_nsteps_ > 0) ? nomsd_.dm_size(true) : nomsd_.size_of_G_for_vbias();
  }

  bool transposed_G_for_vbias() const
  {
    return (inner_nsteps_ > 0) ? false : nomsd_.transposed_G_for_vbias();
  }
  bool transposed_G_for_E() const { return nomsd_.transposed_G_for_E(); }
  bool transposed_vHS() const { return nomsd_.transposed_vHS(); }
  WALKER_TYPES getWalkerType() const { return nomsd_.getWalkerType(); }

  template<class Vec>
  void vMF(Vec&& v, double dt);

  template<class Mat>
  void G_MF(Mat&& G);

  SlaterDetOperations* getSlaterDetOperations() { return nomsd_.getSlaterDetOperations(); }

  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    nomsd_.generalizedFockMatrix(std::forward<Args>(args)...);
  }

  HamiltonianTypes getHamType() { return nomsd_.getHamType(); }

  template<class... Args>
  void getFieldTypes(Args&&... args)
  {
    nomsd_.getFieldTypes(std::forward<Args>(args)...);
  }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    nomsd_.update_potentials(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto getOneBodyPropagatorMatrix(Args&&... args)
  {
    return nomsd_.getOneBodyPropagatorMatrix(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto vHS_sparse(Args&&... args)
  {
    return nomsd_.vHS_sparse(std::forward<Args>(args)...);
  }

  // Phase 3a: force bias. The L*G contraction is trial-independent given G, so this stays a
  // permanent delegate to the OUTER nomsd_ (the True Hamiltonian). What makes it stochastic is its
  // input: MixedDensityMatrix_for_vbias now feeds the inner-ensemble-reduced G[w] (estimator 4,
  // x_gamma[w] = L_gamma . G[w]), and nomsd_.vbias contracts it against the True-Ham Cholesky with
  // the nd = 0 static-anchor half-rotation (no per-determinant nd in vbias). The sqrt(dt)/timestep
  // prefactor stays in the propagator. See StochasticDevelopment.md (Phase 3a).
  template<class MatG, class MatA>
  void vbias(const MatG& G, MatA&& v, double dt, double a = 1.0)
  {
    // Phase 3b: when the inner ensemble is dynamic, MixedDensityMatrix_for_vbias hands us the FULL
    // reduced G, so contract it against the full (un-rotated) True-Ham Cholesky via vbias_fullG.
    // Static limit (inner_nsteps_ == 0): the compact nd = 0 half-rotated delegate (Phase 3a).
    if (inner_nsteps_ > 0)
      nomsd_.vbias_fullG(G, std::forward<MatA>(v), dt, a);
    else
      nomsd_.vbias(G, std::forward<MatA>(v), dt, a);
  }

  template<class MatX, class MatA>
  void vHS(MatX&& X, MatA&& v, double dt, double a = 1.0)
  {
    nomsd_.vHS(std::forward<MatX>(X), std::forward<MatA>(v), dt, a);
  }

  // Phase 2b: stochastic local energy. Reduces the inner ensemble {psi_p} into an effective
  // local energy and overlap per outer walker,
  //   E[w] = sum_p <psi_p|H|phi_w> / sum_p <psi_p|phi_w>,   Ov[w] = (1/P) sum_p <psi_p|phi_w>,
  // the inner_nsteps = 0 specialization of Eq. 27 of arXiv:2505.18519 (B_T = 1, so the phase
  // factor S(Y) and importance reweighting are degenerate). Ov[w] matches Phase 2a Overlap, and
  // at the single-determinant delegate limit both E and Ov equal the NOMSD result. Mirrors
  // NOMSD::Energy_shared with the trial-determinant loop replaced by the inner-walker loop and
  // the CI weight conj(ci[nd]) replaced by 1/P. Definitions in StochasticWfn.icc.
  template<class WlkSet>
  void Energy(WlkSet& wset);

  template<class WlkSet, class Mat, class TVec>
  void Energy(const WlkSet& wset, Mat&& E, TVec&& Ov);

  template<class WlkSet, class MatG>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, bool compact = true, bool transpose = false)
  {
    nomsd_.MixedDensityMatrix(wset, std::forward<MatG>(G), compact, transpose);
  }

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true, bool transpose = false)
  {
    nomsd_.MixedDensityMatrix(wset, std::forward<MatG>(G), std::forward<TVec>(Ov), compact, transpose);
  }

  template<class WlkSet, class MatA, class MatB, class MatG, class TVec>
  void DensityMatrix(const WlkSet& wset,
                     MatA&& RefA,
                     MatB&& RefB,
                     MatG&& G,
                     TVec&& Ov,
                     bool herm,
                     bool compact,
                     bool transposed)
  {
    nomsd_.DensityMatrix(wset, std::forward<MatA>(RefA), std::forward<MatB>(RefB), std::forward<MatG>(G),
                        std::forward<TVec>(Ov), herm, compact, transposed);
  }

  // Phase 3a: stochastic mixed density matrix for the force bias. Reduces the inner ensemble {psi_p}
  // into the effective mixed DM each outer walker's force bias contracts against,
  //   G[w] = sum_p <psi_p|c^dag c|phi_w> / sum_p <psi_p|phi_w>   (estimator 3 of arXiv:2505.18519),
  // the inner_nsteps = 0 specialization (B_T = 1, so the phase factor S(Y) and importance reweighting
  // are degenerate). Returned in the OUTER nomsd_ vbias layout so the unchanged vbias delegate can
  // contract it. Mirrors NOMSD::MixedDensityMatrix_shared's accumulate-then-normalize with the
  // trial-determinant loop replaced by the inner-walker loop and conj(ci[nd]) replaced by 1/P; the
  // 1/P cancels, so G[w]'s overlap normalization matches the Phase 2a/2b Ov. At the single-determinant
  // delegate limit G[w] equals the NOMSD result. Definition in StochasticWfn.icc.
  template<class WlkSet, class MatG>
  void MixedDensityMatrix_for_vbias(const WlkSet& wset, MatG&& G);

  // Phase 2a: stochastic trial overlap. Reduces the inner ensemble {psi_p} into an
  // effective overlap per outer walker, Ov[w] = (1/P) sum_p <psi_p | phi_w>, following
  // Eq. 24 of arXiv:2505.18519 in the static-ensemble limit (inner_nsteps = 0, where
  // B_T = 1 so the phase factor / importance reweighting are degenerate). At the
  // single-determinant delegate limit this equals the NOMSD overlap exactly. The
  // phase/importance-sampling (Eq. 25-26) and inner-propagation leapfrog arrive in
  // Phase 3 (propagator hot path), not here. Definitions in StochasticWfn.icc.
  template<class WlkSet, class TVec>
  void Overlap(const WlkSet& wset, TVec&& Ov);

  template<class WlkSet>
  void Overlap(WlkSet& wset);

  template<class... Args>
  void accumulate_estimators(Args&&... args)
  {
    nomsd_.accumulate_estimators(std::forward<Args>(args)...);
  }

  int number_of_references_for_back_propagation() const
  {
    return nomsd_.number_of_references_for_back_propagation();
  }

  ComplexType getReferenceWeight(int i) const { return nomsd_.getReferenceWeight(i); }

  template<class Mat, class Ptr = ComplexType*>
  void getReferencesForBackPropagation(Mat&& A)
  {
    nomsd_.getReferencesForBackPropagation(std::forward<Mat>(A));
  }

private:
  // Phase 3b: if the latch is armed and inner_nsteps_ > 0, reset the inner walkers to the anchor
  // |phi_T> and apply inner_nsteps_ free-projection B_T steps (fresh Gaussian fields), producing a
  // new walker-independent single-step ensemble {psi_p = B_T(Y^[p])|phi_T>}. Clears the latch.
  // No-op when inner_nsteps_ == 0 (the static Phase 1c-3a ensemble). Defined in StochasticWfn.icc
  // (needs the complete Propagator type).
  void maybe_advance_inner_ensemble();

  // Phase 3b: shared per-inner-walker cross-DM reduction used by Energy (2b) and
  // MixedDensityMatrix_for_vbias (3a). For each inner walker psi_p it forms the cross mixed DM Gp and
  // overlap ov_ = <psi_p|phi_w> against every outer walker (via nomsd_.DensityMatrix(..., herm=false)
  // in the requested compact/transposed layout), computes the phase weight Sp[w] = ov_[w]/|ov_[w]|
  // (Eq. 26), accumulates the denominator D[w] = sum_p Sp[w] and the absolute overlap
  // Ov[w] = (1/P) sum_p ov_[w], and invokes accumulate(Gp, ov_, Sp, ip, r0, rN) so each caller folds
  // Gp into its own (S_p-weighted) numerator. (r0, rN) is this core's band of the Gsize dimension.
  // Calls maybe_advance_inner_ensemble() first so the dynamic ensemble is resampled once per step.
  template<class WlkSet, class TVecD, class TVecOv, class Accumulate>
  void reduce_inner_cross_dm(const WlkSet& wset,
                             bool compact,
                             bool transposed,
                             int Gsize,
                             TVecD&& D,
                             TVecOv&& Ov,
                             Accumulate&& accumulate);
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Wavefunctions/StochasticWfn.icc"

#endif
