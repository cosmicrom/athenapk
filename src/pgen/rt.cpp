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
//! \file rt.cpp
//! \brief Problem generator for the Rayleigh–Taylor instability.

// Kokkos headers
#include "Kokkos_Random.hpp"

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>

// AthenaPK headers
#include "../main.hpp"

namespace rt {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {
  // Get the const_accel_srcterm from input and add it as a parameter to the hydro package
  const Real const_accel_srcterm = pin->GetReal("problem/rt", "const_accel_srcterm");
  hydro_pkg->AddParam<>("const_accel_srcterm", const_accel_srcterm);

  // Determine if the problem is 2D or 3D by checking the number of zones in the X3
  // direction defined in the input file. Then set a parameter in the hydro package to
  // indicate which direction the constant acceleration is in.
  if (pin->GetReal("parthenon/mesh", "nx3") == 1) {
    hydro_pkg->AddParam<>("const_accel_dir", X2DIR);
  } else {
    hydro_pkg->AddParam<>("const_accel_dir", X3DIR);
  }
}

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  // Get variables from the hydro package
  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg = pmb->packages.Get("Hydro");
  const Fluid fluid = hydro_pkg->Param<Fluid>("fluid");
  const Real const_accel_srcterm = hydro_pkg->Param<Real>("const_accel_srcterm");
  // Determine if the problem is 2D or 3D based on the direction of the constant
  // acceleration set in the hydro package
  bool is_2d =
      hydro_pkg->Param<parthenon::CoordinateDirection>("const_accel_dir") == X2DIR;

  // Get size of entire mesh from input file and calculate wave numbers
  Real kx =
      2.0 * M_PI /
      (pin->GetReal("parthenon/mesh", "x1max") - pin->GetReal("parthenon/mesh", "x1min"));
  Real ky =
      2.0 * M_PI /
      (pin->GetReal("parthenon/mesh", "x2max") - pin->GetReal("parthenon/mesh", "x2min"));
  Real kz =
      2.0 * M_PI /
      (pin->GetReal("parthenon/mesh", "x3max") - pin->GetReal("parthenon/mesh", "x3min"));

  // Get index ranges for cells
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Initialize the conserved variables
  auto &u = pmb->meshblock_data.Get()->Get("cons").data;

  // Get coordinates from cell
  parthenon::Coordinates_t &coords = pmb->coords;

  // Read in adiabatic index
  const Real gamma = pin->GetReal("hydro", "gamma");
  const Real gm1 = gamma - 1.0;
  // Read in perturbation amplitude
  const Real perturb_amp = pin->GetReal("problem/rt", "perturb_amp");
  // Read in problem setting
  const int iprob = pin->GetInteger("problem/rt", "iprob");
  // Read in density ratio
  const Real density_ratio = pin->GetOrAddReal("problem/rt", "density_ratio", 2.0);
  // Read in random number generator seed
  const int input_rseed = pin->GetOrAddInteger("problem/rt", "rseed", -1);

  // Based on input random seed, create a unint64 seed. Anything less than zero will use
  // std::random_device to generate a random seed
  uint64_t pool_seed;
  if (input_rseed >= 0) {
    pool_seed = static_cast<uint64_t>(input_rseed);
  } else {
    pool_seed = static_cast<uint64_t>(std::random_device{}());
  }
  // Create a Kokkos RNG pool
  Kokkos::Random_XorShift64_Pool<parthenon::DevExecSpace> rng_pool(pool_seed);

  // If magnetic field is enabled, set various parameters based on input file
  Real b0;
  Real mag_field_angle_rads;
  if (fluid == Fluid::glmmhd) {
    // Read in b0
    b0 = pin->GetReal("problem/rt", "b0");
    // If the problem is 3D, read in the magnetic field angle in degrees and convert to
    // radians
    if (!is_2d) {
      mag_field_angle_rads = pin->GetReal("problem/rt", "mag_field_angle") * M_PI / 180.0;
    }
  }
  //  Case for a 2D problem
  if (is_2d) {
    pmb->par_for(
        "Problem Generator: Rayleigh-Taylor 2D", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // Initialize the lower of the two fluid densities
          Real density = 1.0;
          // For coordinates in X2 that are greater than 0.0, multiply density by the
          // input density ratio to create the more dense fluid
          if (coords.Xc<2>(j) > 0.0) {
            density *= density_ratio;
          }
          // Set density
          u(IDN, k, j, i) = density;

          // If the problem is set to 1, set momentum in X2 based on the wave numbers
          if (iprob == 1) {
            u(IM2, k, j, i) = (1.0 + std::cos(kx * coords.Xc<1>(i))) *
                              (1.0 + std::cos(ky * coords.Xc<2>(j))) / 4.0;
            // Otherwise use the random number generator
          } else {
            // Get an RNG state from the Kokkos RNG pool
            Kokkos::Random_XorShift64<parthenon::DevExecSpace> rng_state =
                rng_pool.get_state();
            // Set momentum in X2 based on random number generation
            u(IM2, k, j, i) =
                (rng_state.drand() - 0.5) * (1.0 + std::cos(ky * coords.Xc<2>(j)));
            // Free the RNG state
            rng_pool.free_state(rng_state);
          }
          // Set the other momenta and update X2 momentum based on current density
          u(IM1, k, j, i) = 0.0;
          u(IM2, k, j, i) *= density * perturb_amp;
          u(IM3, k, j, i) = 0.0;

          // Set energy
          u(IEN, k, j, i) =
              (1.0 / gamma + const_accel_srcterm * density * coords.Xc<2>(j)) / gm1 +
              0.5 * SQR(u(IM2, k, j, i)) / density;

          // Set magnetic fields if enabled
          if (fluid == Fluid::glmmhd) {
            u(IB1, k, j, i) = b0;
            u(IB2, k, j, i) = 0.0;
            u(IB3, k, j, i) = 0.0;
            // Update energy
            u(IEN, k, j, i) += 0.5 * SQR(b0);
          }
        });

    // Otherwise this is a 3D problem
  } else {
    // Create parthenon loop
    pmb->par_for(
        "Problem Generator: Rayleigh-Taylor 3D", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // Initialize the lower of the two fluid densities
          Real density = 1.0;
          // For coordinates in X3 that are greater than 0.0, multiply density by the
          // input density ratio to create the more dense fluid
          if (coords.Xc<3>(k) > 0.0) {
            density *= density_ratio;
          }
          // Set density
          u(IDN, k, j, i) = density;

          // If the problem is set to 1, set momentum in X3 based on the wave numbers
          if (iprob == 1) {
            u(IM3, k, j, i) = (1.0 + std::cos(kx * (coords.Xc<1>(i)))) / 8.0 *
                              (1.0 + std::cos(ky * coords.Xc<2>(j))) *
                              (1.0 + std::cos(kz * coords.Xc<3>(k)));
            // Otherwise use the random number generator
          } else {
            // Get an RNG state from the Kokkos RNG pool
            Kokkos::Random_XorShift64<parthenon::DevExecSpace> rng_state =
                rng_pool.get_state();
            // Set momentum in X3 based on random number generator
            u(IM3, k, j, i) = perturb_amp * (rng_state.drand() - 0.5) *
                              (1.0 + std::cos(kz * coords.Xc<3>(k)));
            // Free the RNG state
            rng_pool.free_state(rng_state);
          }
          // Set the other momenta and update X3 momentum based on current density
          u(IM1, k, j, i) = 0.0;
          u(IM2, k, j, i) = 0.0;
          u(IM3, k, j, i) *= density * perturb_amp;

          // Set energy
          u(IEN, k, j, i) =
              (1.0 / gamma + const_accel_srcterm * density * coords.Xc<3>(k)) / gm1 +
              0.5 * SQR(u(IM3, k, j, i)) / density;

          /// Set magnetic fields if enabled
          if (fluid == Fluid::glmmhd) {
            // For coordinates in X3 that are greater than 0.0, set the field along X1
            if (coords.Xc<3>(k) > 0.0) {
              u(IB1, k, j, i) = b0;
              u(IB2, k, j, i) = 0.0;
              // Otherwise set using the magnetic field angle
            } else {
              u(IB1, k, j, i) = b0 * std::cos(mag_field_angle_rads);
              u(IB2, k, j, i) = b0 * std::sin(mag_field_angle_rads);
            }
            u(IB3, k, j, i) = 0.0;
            // Update energy
            u(IEN, k, j, i) += 0.5 * SQR(b0);
          }
        });
  }
}

} // namespace rt
