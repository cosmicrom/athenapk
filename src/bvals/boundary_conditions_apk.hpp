//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file boundary_conditions_apk.chpp
//  \brief AthenaPK specific boundary conditions
//

#ifndef BVALS_BOUNDARY_CONDITIONS_APK_HPP_
#define BVALS_BOUNDARY_CONDITIONS_APK_HPP_

#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

#include "basic_types.hpp"
#include "bvals/boundary_conditions_generic.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

#include "../main.hpp"

namespace Hydro {
namespace BoundaryFunction {

using namespace parthenon::package::prelude;
using parthenon::CoordinateDirection;
// using parthenon::MeshBlockData;
// using parthenon::Real;
using parthenon::BoundaryFunction::BCSide;

template <CoordinateDirection DIR, BCSide SIDE>
void ReflectBC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  // make sure DIR is X[123]DIR so we don't have to check again
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  MeshBlock *pmb = mbd->GetBlockPointer();

  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto fluid = hydro_pkg->Param<Fluid>("fluid");
  PARTHENON_REQUIRE_THROWS(
      fluid == Fluid::euler,
      "Reflecting boundary conditions for MHD need special treatment.");

  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                               : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // used for reflections
  const int offset = (2 * ref) + (INNER ? -1 : 1);

  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const bool fine = false; // no usage of fine fields in AthenaPK for now

  const auto nv = IndexRange{0, cons.GetDim(4) - 1};
  pmb->par_for_bndry(
      "ReflectBC", nv, domain, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &v, const int &k, const int &j, const int &i) {
        const bool reflect = v == DIR;
        cons(v, k, j, i) =
            (reflect ? -1.0 : 1.0) *
            cons(v, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
      });
}

template <parthenon::CoordinateDirection DIR, parthenon::BoundaryFunction::BCSide SIDE>
void ProjectPressure(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  // Asset that input coordinate direction is acceptable
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  // Get mesh block pointer
  MeshBlock *pmb = mbd->GetBlockPointer();
  // Get conserved variables
  VariablePack<parthenon::Real> cons =
      mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  // Get coordinates
  parthenon::Coordinates_t &coords = pmb->coords;

  // Get variables from hydro package
  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg = pmb->packages.Get("Hydro");
  const Fluid fluid = hydro_pkg->Param<Fluid>("fluid");
  const Real const_accel_srcterm = hydro_pkg->Param<Real>("const_accel_srcterm");
  const Real gamma = hydro_pkg->Param<Real>("AdiabaticIndex");

  // Convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == parthenon::BoundaryFunction::BCSide::Inner);
  // Determine the cell bounds
  const parthenon::IndexShape &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  // Determine the range of the bounds
  const parthenon::IndexRange &range =
      X1 ? bounds.GetBoundsI(IndexDomain::interior)
         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
               : bounds.GetBoundsK(IndexDomain::interior));
  // Determine reference point in mesh based on input coordinate direction and side
  const int ref = INNER ? range.s : range.e;
  // Determine what the index domain is
  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // Calculate a shift value for enums based on the input coordinate direction
  constexpr int enum_shift = X1 ? 0 : (X2 ? 1 : 2);
  // Determine momentum and magnetic field enums to reflect values to ghost cells
  constexpr int m_reflect_enum = IM1 + enum_shift;
  constexpr int b_reflect_enum = IB1 + enum_shift;
  // Save the other momentum and magnetic field enums to copy values to ghost cells
  constexpr int m_copy_enum_1 = IM1 + (enum_shift + 1) % 3;
  constexpr int m_copy_enum_2 = IM1 + (enum_shift + 2) % 3;
  constexpr int b_copy_enum_1 = IB1 + (enum_shift + 1) % 3;
  constexpr int b_copy_enum_2 = IB1 + (enum_shift + 2) % 3;

  // Calculate offset
  const int offset = (2 * ref) + (INNER ? -1 : 1);

  // Set fine to false since there is currently no usage of fine fields in AthenaPK
  const bool fine = false;

  pmb->par_for_bndry(
      "ProjectPressure", IndexRange{0, 0}, domain, parthenon::TopologicalElement::CC,
      coarse, fine, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        // Determine i, j, and k values for a cell in the mesh based on the offset
        // calculated above and the input coordinate direction
        const int mesh_i = X1 ? offset - i : i;
        const int mesh_j = X2 ? offset - j : j;
        const int mesh_k = X3 ? offset - k : k;

        // Copy density
        cons(IDN, k, j, i) = cons(IDN, mesh_k, mesh_j, mesh_i);

        // Reflect momentum in the input coordinate direction
        cons(m_reflect_enum, k, j, i) = -cons(m_reflect_enum, mesh_k, mesh_j, mesh_i);
        // Copy the other momenta
        cons(m_copy_enum_1, k, j, i) = cons(m_copy_enum_1, mesh_k, mesh_j, mesh_i);
        cons(m_copy_enum_2, k, j, i) = cons(m_copy_enum_2, mesh_k, mesh_j, mesh_i);

        // From the conserved variables, calculate the pressure of the cell in the mesh
        const Real mesh_density = cons(IDN, mesh_k, mesh_j, mesh_i);
        const Real mesh_kinetic_energy = 0.5 / mesh_density *
                                         (SQR(cons(IM1, mesh_k, mesh_j, mesh_i)) +
                                          SQR(cons(IM2, mesh_k, mesh_j, mesh_i)) +
                                          SQR(cons(IM3, mesh_k, mesh_j, mesh_i)));
        const Real mesh_pressure =
            (gamma - 1.0) * cons(IEN, mesh_k, mesh_j, mesh_i) - mesh_kinetic_energy;

        // Determine the number of ghost cells that separate the current cell from the
        // rest of the mesh in the input coordinate direction
        const int num_ghost_cells =
            X1 ? (i > ref ? i - ref : ref - i)
               : (X2 ? (j > ref ? j - ref : ref - j) : (k > ref ? k - ref : ref - k));
        // Determine the distance between cell faces for the input coordinate direction
        const Real dxf =
            X1 ? coords.Dxf<1>(i) : (X2 ? coords.Dxf<2>(j) : coords.Dxf<3>(k));
        // Set pressure using the constant acceleration given
        const Real ghost_pressure = mesh_pressure + (INNER ? -1 : 1) * mesh_density *
                                                        const_accel_srcterm *
                                                        (2 * num_ghost_cells - 1) * dxf;
        // Set energy
        cons(IEN, k, j, i) = ghost_pressure / (gamma - 1.0) + mesh_kinetic_energy;

        // If magnetic fields are enabled, set those as well
        if (fluid == Fluid::glmmhd) {
          // Reflect
          cons(b_reflect_enum, k, j, i) = -cons(b_reflect_enum, mesh_k, mesh_j, mesh_i);
          // Copy
          cons(b_copy_enum_1, k, j, i) = cons(b_copy_enum_1, mesh_k, mesh_j, mesh_i);
          cons(b_copy_enum_2, k, j, i) = cons(b_copy_enum_2, mesh_k, mesh_j, mesh_i);
        }
      });
}

} // namespace BoundaryFunction
} // namespace Hydro

#endif // BVALS_BOUNDARY_CONDITIONS_APK_HPP_
