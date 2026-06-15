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

#ifndef SFQMC_AFQMC_HAMILTONIANOPERATIONS_FULL_G_ESTIMATORS_HPP
#define SFQMC_AFQMC_HAMILTONIANOPERATIONS_FULL_G_ESTIMATORS_HPP

#include <vector>

#include "AFQMC/config.h"
#include "Utilities/AppAbort.hpp"
#include "Utilities/FairDivide.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"
#include "Numerics/ma_operations.hpp"
#include "Numerics/ma_blas.hpp"
#include "Memory/buffer_managers.h"
#include "AFQMC/Utilities/taskgroup.h"

namespace sfqmc
{
namespace afqmc
{
// Phase 3b (StochasticWfn): un-rotated full-G local-energy contraction, shared by the dense
// (Real3IndexFactorization) and sparse (SparseTensor) Cholesky HamiltonianOperations so there is a
// SINGLE implementation of the (genuinely new) EXX kernel. The standard energy() contracts the
// per-trial-determinant half-rotated integrals against a COMPACT G in that determinant's occupied
// basis; once a stochastic inner walker leaves the trial anchor that half-rotation no longer matches
// the cross-DM basis, so this contracts the full Cholesky and bare one-body against the FULL NMO x NMO
// cross G. With the IDENTITY-rotated full Cholesky it reproduces the half-rotated energy when the bra
// IS the anchor (validated by the stochastic_full_g_matches_compact test). CLOSED (RHF) only this phase.
namespace full_g
{
// Energy components (E1, EXX, EJ) from a full mixed Green's function. CLOSED only.
//   E      : [nwalk][>=3] (ComplexType). Filled to 0 here; PARTIAL per core (E0/E1 gated by addH1,
//            EXX/EJ partitioned by core) -- the caller (StochasticWfn::Energy) all-reduces.
//   Gfull  : [nwalk][NMO*NMO] (ComplexType), G[w] flattened as G[w][i*NMO+k].
//   Lankf  : [NMO*local_nCV][NMO] (SPComplexType): the identity-rotated full Cholesky, L_ik^nc at
//            row (i*local_nCV + nc), col k.
//   hijf   : [NMO*NMO] (ComplexType): bare one-body h_ik at (i*NMO+k).
// Mirrors Real3IndexFactorization::energy_impl's CLOSED EXX/EJ structure with the occupied index a -> a
// full orbital index i and the half-rotated Lank/haj replaced by Lankf/hijf.
template<class SPComplexType, class MatE, class MatG, class MatLan, class VecHij>
void energy_closed(TaskGroup_& TG,
                   LocalTGBufferManager& shm,
                   MatE&& E,
                   MatG const& Gfull,
                   MatLan const& Lankf,
                   VecHij const& hijf,
                   int local_nCV,
                   ComplexType E0,
                   bool addH1  = true,
                   bool addEJ  = true,
                   bool addEXX = true)
{
  using std::fill_n;
  static_assert(std::decay_t<MatE>::dimensionality == 2, "full_g::energy_closed: E must be 2D");
  static_assert(std::decay_t<MatG>::dimensionality == 2, "full_g::energy_closed: Gfull must be 2D");
  static_assert(std::decay_t<MatLan>::dimensionality == 2, "full_g::energy_closed: Lankf must be 2D");
  static_assert(std::decay_t<VecHij>::dimensionality == 1, "full_g::energy_closed: hijf must be 1D");
  using ShmSPMatrix = multi::static_array<SPComplexType, 2, typename LocalTGBufferManager::template allocator_t<SPComplexType>>;
  using SpC4Tensor_ref = multi::array_ref<SPComplexType, 4>;

  int nwalk      = int(Gfull.size(0));
  int NMO        = int(Lankf.size(1));
  ComplexType scl(2.0, 0.0); // CLOSED: spatial DM counts both spins

  RUNTIME_CHECK(Gfull.size(1) == long(NMO) * NMO, "");
  RUNTIME_CHECK(Lankf.size(0) == long(NMO) * local_nCV, "");
  RUNTIME_CHECK(long(hijf.size(0)) == long(NMO) * NMO, "");

  for (int n = 0; n < nwalk; ++n)
    fill_n(E[n].origin(), 3, ComplexType(0.0));
  if (addH1)
    for (int i = 0; i < nwalk; ++i)
      E[i][0] += E0;

  // one-body: E[w][0] += scl * sum_ik h_ik G[w][i*NMO+k]  (gated by addH1 -> only the caller's root core)
  if (addH1)
    ma::product(scl, Gfull, hijf, ComplexType(1.0), E(E.extension(0), 0));

  if (addEXX)
  {
    ShmSPMatrix GF({long(nwalk) * NMO, NMO}, shm.get_generator().template get_allocator<SPComplexType>());
    for (int n = 0; n < nwalk; ++n)
    {
      if (n % TG.TG_local().size() != TG.TG_local().rank())
        continue;
      copy_n_cast(raw_pointer_cast(Gfull[n].origin()), long(NMO) * NMO,
                  raw_pointer_cast(GF.origin()) + long(n) * NMO * NMO);
    }
    TG.TG_local().barrier();

    // Twban[(w,i)][(j,nc)] = sum_k GF[(w,i)][k] Lankf[(j,nc)][k] = sum_k G[w][i][k] L_jk^nc
    ShmSPMatrix Twban({long(nwalk) * NMO, long(NMO) * local_nCV},
                      shm.get_generator().template get_allocator<SPComplexType>());
    SpC4Tensor_ref T4D(raw_pointer_cast(Twban.origin()), {nwalk, NMO, NMO, local_nCV});

    long i0, iN;
    std::tie(i0, iN) =
        FairDivideBoundary(long(TG.TG_local().rank()), long(NMO) * local_nCV, long(TG.TG_local().size()));
    ma::product(GF, ma::T(Lankf.sliced(i0, iN)), Twban(Twban.extension(0), {i0, iN}));
    TG.TG_local().barrier();

    // EXX[w] = -1/2 scl sum_{i,j,nc} T4D[w][i][j][nc] T4D[w][j][i][nc]  (partitioned over (w,i) pairs).
    for (int n = 0, an = 0; n < nwalk; ++n)
    {
      ComplexType E_(0.0);
      for (int a = 0; a < NMO; ++a, ++an)
      {
        if (an % TG.TG_local().size() != TG.TG_local().rank())
          continue;
        for (int b = 0; b < NMO; ++b)
          E_ += static_cast<ComplexType>(ma::dot(T4D[n][a][b], T4D[n][b][a]));
      }
      E[n][1] -= 0.5 * scl * E_;
    }

    if (addEJ)
    {
      // Kl[w][nc] = sum_i T4D[w][i][i][nc]; EJ[w] = 1/2 scl^2 sum_nc Kl[w][nc]^2 (per owned walker).
      ShmSPMatrix Kl({nwalk, local_nCV}, SPComplexType(0.0), shm.get_generator().template get_allocator<SPComplexType>());
      for (int n = 0; n < nwalk; ++n)
      {
        if (n % TG.TG_local().size() != TG.TG_local().rank())
          continue;
        for (int a = 0; a < NMO; ++a)
          ma::axpy(SPComplexType(1.0), T4D[n][a][a], Kl[n]);
      }
      TG.TG_local().barrier();
      for (int n = 0; n < nwalk; ++n)
        if (n % TG.TG_local().size() == TG.TG_local().rank())
          E[n][2] += 0.5 * scl * scl * static_cast<ComplexType>(ma::dot(Kl[n], Kl[n]));
    }
  }
  if (addEXX and not addEJ)
    APP_ABORT(" Error: addEXX and not addEJ not yet implemented in full_g::energy_closed. \n\n");

  TG.TG_local().barrier();
}

} // namespace full_g
} // namespace afqmc
} // namespace sfqmc

#endif
