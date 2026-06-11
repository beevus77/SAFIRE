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
        nomsd_(info, nomsd_inputs(pt_in), tg_, std::move(outer_sdet_), std::move(outer_hop_), std::move(ci_),
               std::move(orbs_), wlk, nce, targetNW),
        inner_stack_(std::move(inner_stack_in))
  {
    if (inner_stack_ == nullptr)
      APP_ABORT("Error in StochasticWfn: inner stack must be provided (see WavefunctionFactory).");
    ptree pt = interpret_inputs(pt_in);
    inner_nwalkers_ = pt.get<int>("inner_nwalkers");
    inner_nsteps_   = pt.get<int>("inner_nsteps");
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
    if (inner_nsteps > 0)
      APP_ABORT("Error in StochasticWfn::interpret_inputs: inner_nsteps > 0 not yet supported "
                "(inner propagation arrives with Phase 2+; the Phase 1c ensemble is static).");
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

  NOMSD<MP, devPsiT>& outer_nomsd() { return nomsd_; }
  NOMSD<MP, devPsiT> const& outer_nomsd() const { return nomsd_; }

  int local_number_of_cholesky_vectors() const { return nomsd_.local_number_of_cholesky_vectors(); }
  int global_number_of_cholesky_vectors() const { return nomsd_.global_number_of_cholesky_vectors(); }
  int global_origin_cholesky_vector() const { return nomsd_.global_origin_cholesky_vector(); }
  bool distribution_over_cholesky_vectors() const { return nomsd_.distribution_over_cholesky_vectors(); }
  bool spin_dependent_vHS() const { return nomsd_.spin_dependent_vHS(); }

  int size_of_G_for_vbias() const { return nomsd_.size_of_G_for_vbias(); }

  bool transposed_G_for_vbias() const { return nomsd_.transposed_G_for_vbias(); }
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

  template<class MatG, class MatA>
  void vbias(const MatG& G, MatA&& v, double dt, double a = 1.0)
  {
    nomsd_.vbias(G, std::forward<MatA>(v), dt, a);
  }

  template<class MatX, class MatA>
  void vHS(MatX&& X, MatA&& v, double dt, double a = 1.0)
  {
    nomsd_.vHS(std::forward<MatX>(X), std::forward<MatA>(v), dt, a);
  }

  template<class WlkSet>
  void Energy(WlkSet& wset)
  {
    nomsd_.Energy(wset);
  }

  template<class WlkSet, class Mat, class TVec>
  void Energy(const WlkSet& wset, Mat&& E, TVec&& Ov)
  {
    nomsd_.Energy(wset, std::forward<Mat>(E), std::forward<TVec>(Ov));
  }

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

  template<class WlkSet, class MatG>
  void MixedDensityMatrix_for_vbias(const WlkSet& wset, MatG&& G)
  {
    nomsd_.MixedDensityMatrix_for_vbias(wset, std::forward<MatG>(G));
  }

  template<class WlkSet, class TVec>
  void Overlap(const WlkSet& wset, TVec&& Ov)
  {
    nomsd_.Overlap(wset, std::forward<TVec>(Ov));
  }

  template<class WlkSet>
  void Overlap(WlkSet& wset)
  {
    nomsd_.Overlap(wset);
  }

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
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Wavefunctions/StochasticWfn.icc"

#endif
