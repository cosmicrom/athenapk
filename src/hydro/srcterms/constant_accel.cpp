//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file constant_accel.cpp
//========================================================================================

// AthenaPK headers
#include "constant_accel.hpp"

namespace const_accel {

void ConstantAccelSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                          const Real dt) {
  // Get cons and prim mesh block packs
  const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &prim_pack =
      md->PackVariables(std::vector<std::string>{"prim"});
  const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &cons_pack =
      md->PackVariables(std::vector<std::string>{"cons"});
  // Get bounds
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // Get variables from hydro package
  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg =
      md->GetMeshPointer()->packages.Get("Hydro");
  const Real const_accel_srcterm = hydro_pkg->Param<Real>("const_accel_srcterm");
  const parthenon::CoordinateDirection dir =
      hydro_pkg->Param<parthenon::CoordinateDirection>("const_accel_dir");

  // Determine the enum values of velocity and momentum based off the const_accel_dir
  // parameter in the hydro package
  int velocity_enum_val;
  int momentum_enum_val;
  if (dir == parthenon::CoordinateDirection::X1DIR) {
    velocity_enum_val = IV1;
    momentum_enum_val = IM1;
  } else if (dir == parthenon::CoordinateDirection::X2DIR) {
    velocity_enum_val = IV2;
    momentum_enum_val = IM2;
  } else {
    velocity_enum_val = IV3;
    momentum_enum_val = IM3;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConstantAccelSrcTerm", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Get cons and prim variable packs
        parthenon::VariablePack<parthenon::Real> &cons = cons_pack(b);
        parthenon::VariablePack<parthenon::Real> &prim = prim_pack(b);
        // Calculate the source term
        Real src = dt * prim(IDN, k, j, i) * const_accel_srcterm;
        // Update momentum
        cons(momentum_enum_val, k, j, i) += src;
        // Update energy
        cons(IEN, k, j, i) += src * prim(velocity_enum_val, k, j, i);
      });
}

} // namespace const_accel