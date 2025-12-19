//========================================================================================
// AthenaPK - a performance portable block
// structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon
// Collaboration. All rights reserved. Licensed
// under the 3-clause BSD License, see LICENSE
// file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone
// <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD
// License, see LICENSE file for details
//========================================================================================
//! \file jet.cpp
//! \brief Problem generator for laboratory jets

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace jet {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {
  // Get the const_accel_srcterm from input and add it as a parameter to the hydro package
  const Real const_accel_srcterm = pin->GetReal("problem/jet", "const_accel_srcterm");
  hydro_pkg->AddParam<>("const_accel_srcterm", const_accel_srcterm);

  // Determine if the problem is 2D or 3D by checking the number of zones in the X3
  // direction defined in the input file. Then set a parameter in the hydro package to
  // indicate which direction the constant acceleration is in/
  if (pin->GetReal("parthenon/mesh", "nx3") == 1) {
    hydro_pkg->AddParam<>("const_accel_dir", X2DIR);
  } else {
    hydro_pkg->AddParam<>("const_accel_dir", X3DIR);
  }
}

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  // Get variables from the hydro package
  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg = pmb->packages.Get("Hydro");
  const Real const_accel_srcterm = hydro_pkg->Param<Real>("const_accel_srcterm");
  // Determine if the problem is 2D or 3D based on the direction of the constant
  // acceleration set in the hydro package
  bool is_2d =
      hydro_pkg->Param<parthenon::CoordinateDirection>("const_accel_dir") == X2DIR;

  // Get index ranges for cells
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Initialize the conserved variables
  auto &u = pmb->meshblock_data.Get()->Get("cons").data;

  // Get coordinates from cell
  parthenon::Coordinates_t &coords = pmb->coords;

  // Determine bounds along the axis of the constant acceleration
  const Real grid_min = pin->GetReal("parthenon/mesh", "x2min");
  const Real grid_max = pin->GetReal("parthenon/mesh", "x2max");

  // Read in densities
  const Real rho_min = pin->GetReal("problem/jet", "rho_min");
  const Real rho_max = pin->GetReal("problem/jet", "rho_max");
  // Read in adiabatic index
  const Real gamma = pin->GetReal("hydro", "gamma");
  const Real gm1 = gamma - 1.0;

  // Calculate initial pressure
  const Real p0 = 1.0 / gamma;
  // Calculate slope of density
  const Real rho_slope = (rho_min - rho_max) / (grid_max - grid_min);

  pmb->par_for(
      "Problem Generator: Jet", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Determine to locations of the cell faces on the axis of constant acceleration
        const Real bottom_face = coords.Xf<2>(j);
        const Real top_face = bottom_face + coords.Dxf<2>(j);
        // Calculate the distance from the minimum value of the grid to each cell face
        const Real bottom_offset = bottom_face - grid_min;
        const Real top_offset = top_face - grid_min;
        // Calculate cell offset average
        const Real offset_avg = (bottom_offset + top_offset) / 2.0;
        // Calculate cell offset squared average
        const Real offset_sq_avg =
            (SQR(bottom_offset) + bottom_offset * top_offset + SQR(top_offset)) / 3.0;

        // Calculate density and pressure
        const Real density = rho_max + rho_slope * offset_avg;
        const Real pressure =
            p0 + const_accel_srcterm *
                     (rho_max * offset_avg + 0.5 * rho_slope * offset_sq_avg);

        // Set cell conserved variables
        u(IDN, k, j, i) = density;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = pressure / gm1;
      });
}

} // namespace jet
