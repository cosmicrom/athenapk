//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file constant_accel.cpp
//========================================================================================

// General headers
#include <unordered_map>

// AthenaPK headers
#include "constant_accel.hpp"

namespace const_accel {

// Map for coordinate direction to momentum enum
std::unordered_map<parthenon::CoordinateDirection, int> momentum_enum_map = {
    {parthenon::CoordinateDirection::X1DIR, IM1},
    {parthenon::CoordinateDirection::X2DIR, IM2},
    {parthenon::CoordinateDirection::X3DIR, IM3}};

void ConstantAccel(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  // Get cons mesh block packs
  const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &cons_pack =
      md->PackVariables(std::vector<std::string>{"cons"});
  // Get bounds
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // Get variables from hydro package
  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg =
      md->GetMeshPointer()->packages.Get("Hydro");
  const Real const_accel = hydro_pkg->Param<Real>("const_accel");
  const parthenon::CoordinateDirection dir =
      hydro_pkg->Param<parthenon::CoordinateDirection>("const_accel_dir");

  // Get the enum value for momentum based off the direction of constant acceleration
  int momentum_enum_val = momentum_enum_map.at(dir);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConstantAccel", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Get cons variable packs
        parthenon::VariablePack<parthenon::Real> &cons = cons_pack(b);
        // Calculate the source term
        const Real rho = cons(IDN, k, j, i);
        const Real v = cons(momentum_enum_val, k, j, i) / rho;
        const Real src = dt * rho * const_accel;
        // Update momentum
        cons(momentum_enum_val, k, j, i) += src;
        // Update energy
        cons(IEN, k, j, i) += src * v;
      });
}

} // namespace const_accel