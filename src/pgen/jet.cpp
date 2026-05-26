//========================================================================================
// AthenaPK - a performance portable block
// structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon
// Collaboration. All rights reserved. Licensed
// under the 3-clause BSD License, see LICENSE
// file for details
//========================================================================================
//! \file jet.cpp
//! \brief Problem generator for jets

// General headers
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

// Parthenon headers
#include "basic_types.hpp"
#include "config.hpp"
#include "interface/metadata.hpp"
#include "interface/variable_pack.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../gauss.hpp"
#include "../hydro/srcterms/constant_accel.hpp"
#include "../main.hpp"

namespace jet {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

// Define density profile modes enum and map
enum class RhoProfileMode { Constant, Linear, Power, Exponential };
std::unordered_map<std::string, RhoProfileMode> RhoProfileMap = {
    {"const", RhoProfileMode::Constant},
    {"lin", RhoProfileMode::Linear},
    {"pow", RhoProfileMode::Power},
    {"expo", RhoProfileMode::Exponential}};
// Define magnetic field injection type enum and map
enum class MagFieldInjectType { Loop, Tower };
std::unordered_map<std::string, MagFieldInjectType> MagFieldInjectTypeMap = {
    {"loop", MagFieldInjectType::Loop}, {"tower", MagFieldInjectType::Tower}};

// Define structs
struct IndexRangeStruct {
  IndexRange ib;
  IndexRange jb;
  IndexRange kb;
};

struct JetInitStruct {
  Fluid fluid;
  Real const_accel;
  Real gamma;
  Real x2_min;
  Real rho_0;
  Real rho_ref;
  Real r_ref;
  Real rho_delta;
  RhoProfileMode rho_prof_mode;
  bool enable_tracer;
  int nhydro;
  Real b0;
};

struct HydroInjectStruct {
  Real radius;
  Real volume;
  Real x2_min;
  Real height;
  Real ke_frac;
  Real q_frac;
  Real rho_rate;
  Real power_density;
  bool enable_tracer;
  int nhydro;
};

struct MagInjectStruct {
  Real x2_min;
  MagFieldInjectType type;
  Real l_scale;
  Real offset;
  Real thickness;
  Real b_frac;
};

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {
  // Add parameters to the hydro package
  // Constant acceleration
  hydro_pkg->AddParam("const_accel", pin->GetReal("problem/jet", "const_accel"));
  //
  hydro_pkg->AddParam("x2_min", pin->GetReal("parthenon/mesh", "x2min"));
  // Direction of constant acceleration
  hydro_pkg->AddParam("const_accel_dir", X2DIR);

  // Jet injection radius centered at zero on x and y axes
  hydro_pkg->AddParam("jet_inject_radius",
                      pin->GetReal("problem/jet", "jet_inject_radius"));
  // Jet injection height located at the bottom of the z axis
  const Real jet_height = pin->GetReal("problem/jet", "jet_inject_height");
  const Real inject_height = pin->GetReal("parthenon/mesh", "x2min") + jet_height;
  hydro_pkg->AddParam("jet_inject_height", inject_height);
  // Initialize jet injection volume which is calculated in ProblemGenerator
  hydro_pkg->AddParam("jet_inject_volume", 0.0, true);
  // Jet mass injection rate
  hydro_pkg->AddParam("jet_m_inject_rate",
                      pin->GetReal("problem/jet", "jet_m_inject_rate"));
  // Jet power
  hydro_pkg->AddParam("jet_power", pin->GetReal("problem/jet", "jet_power"));
  // Jet kinetic energy fraction
  const Real ke_frac = pin->GetReal("problem/jet", "jet_ke_frac");
  hydro_pkg->AddParam("jet_ke_frac", ke_frac);
  PARTHENON_REQUIRE(ke_frac >= 0.0, "Input Invalid: jet_ke_frac < 0");
  // Jet thermal energy fraction
  const Real q_frac = pin->GetReal("problem/jet", "jet_q_frac");
  PARTHENON_REQUIRE(q_frac >= 0.0, "Input Invalid: jet_q_frac < 0");
  hydro_pkg->AddParam("jet_q_frac", q_frac);
  // Enable tracer flag
  const bool enable_tracer = pin->GetOrAddBoolean("problem/jet", "enable_tracer", false);
  PARTHENON_REQUIRE(
      !enable_tracer || hydro_pkg->Param<int>("nscalars") >= 1,
      "Input Invalid: Enabling tracer for jet requires hydro/nscalars >= 1");
  hydro_pkg->AddParam("enable_tracer", enable_tracer);

  //
  if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd) {
    // Jet magnetic energy fraction
    const Real b_frac = pin->GetOrAddReal("problem/jet", "jet_b_frac", 0.0);
    PARTHENON_REQUIRE(b_frac >= 0.0, "Input Invalid: jet_b_frac < 0");
    PARTHENON_REQUIRE(abs(ke_frac + q_frac + b_frac - 1.0) < 1e-12,
                      "Input Invalid: Jet kinetic, thermal, and magnetic energy "
                      "fractions must sum to 1.0");
    hydro_pkg->AddParam("jet_b_frac", b_frac);

    //
    if (b_frac > 0.0) {
      const Real mag_offset = pin->GetReal("problem/jet", "mag_field_inject_offset");
      const Real mag_thickness =
          pin->GetReal("problem/jet", "mag_field_inject_thickness");
      const Real mag_l_scale = pin->GetReal("problem/jet", "mag_field_inject_l_scale");
      PARTHENON_REQUIRE(mag_offset >= 0.0, "Input Invalid: mag_field_inject_offset < 0");
      PARTHENON_REQUIRE(mag_thickness > 0.0,
                        "Input Invalid: mag_field_inject_thickness <= 0");
      PARTHENON_REQUIRE(
          mag_offset + mag_thickness <= jet_height,
          "Input Invalid: magnetic injection layer extends beyond jet nozzle");
      PARTHENON_REQUIRE(mag_l_scale > 0.0,
                        "Input Invalid: mag_field_inject_l_scale <= 0");
      PARTHENON_REQUIRE(
          pin->GetString("problem/jet", "mag_field_inject_type", {"loop", "tower"}) ==
              "loop",
          "Input Invalid: mag_field_inject_type=tower is not implemented for the Jet "
          "problem. Use mag_field_inject_type=loop.");
      //
      hydro_pkg->AddParam("mag_field_inject_type",
                          MagFieldInjectTypeMap.at(pin->GetString(
                              "problem/jet", "mag_field_inject_type", {"loop"})));
      //
      hydro_pkg->AddParam("mag_field_inject_l_scale", mag_l_scale);
      //
      hydro_pkg->AddParam("mag_field_inject_offset", mag_offset);
      //
      hydro_pkg->AddParam("mag_field_inject_thickness", mag_thickness);
      //
      parthenon::Metadata m({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                             parthenon::Metadata::OneCopy},
                            std::vector<int>({3}));
      hydro_pkg->AddField("mag_field_inject_A", m);
    }
    //
  } else {
    PARTHENON_REQUIRE(
        abs(ke_frac + q_frac - 1.0) < 1e-12,
        "Input Invalid: Jet kinetic and thermal energy fractions must sum to 1.0");
  }
}

void SetInitialConditions(MeshBlock *pmb, ParArrayND<double, parthenon::VariableState> &u,
                          const parthenon::Coordinates_t &coords,
                          const IndexRangeStruct index_ranges,
                          const JetInitStruct &jet_init_struct) {
  // Calculate initial pressure
  const Real p0 = 1.0 / jet_init_struct.gamma;
  // Calculate gamma minus one
  const Real gm1 = jet_init_struct.gamma - 1.0;
  // Calculate relevant constant for input density profile
  Real c;
  if (jet_init_struct.rho_prof_mode == RhoProfileMode::Linear) {
    c = jet_init_struct.rho_delta / jet_init_struct.r_ref;
  } else if (jet_init_struct.rho_prof_mode == RhoProfileMode::Power) {
    c = -log(jet_init_struct.rho_ref / jet_init_struct.rho_0) / log(2.0);
  } else {
    c = log(jet_init_struct.rho_ref / jet_init_struct.rho_0) / jet_init_struct.r_ref;
  }

  // Set initial conditions
  pmb->par_for(
      "Problem Generator: Jet", index_ranges.kb.s, index_ranges.kb.e, index_ranges.jb.s,
      index_ranges.jb.e, index_ranges.ib.s, index_ranges.ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Determine to locations of the cell faces on the z axis
        const Real top_face = coords.Xf<2>(j) + coords.Dxf<2>(j);
        // Calculate the distance from the minimum value of the grid to each cell face
        const Real bottom_offset = coords.Xf<2>(j) - jet_init_struct.x2_min;
        const Real top_offset = top_face - jet_init_struct.x2_min;
        // Calculate cell offset average
        const Real offset_avg = (bottom_offset + top_offset) / 2.0;

        // Create lambda function for density profile based on input profile mode
        auto rho_profile = [=](const Real r) {
          if (jet_init_struct.rho_prof_mode == RhoProfileMode::Constant) {
            return jet_init_struct.rho_0;
          } else if (jet_init_struct.rho_prof_mode == RhoProfileMode::Linear) {
            return jet_init_struct.rho_0 + c * r;
          } else if (jet_init_struct.rho_prof_mode == RhoProfileMode::Power) {
            return jet_init_struct.rho_0 * pow(1.0 + r / jet_init_struct.r_ref, -c);
          } else {
            return jet_init_struct.rho_0 * exp(c * r);
          }
        };
        // Initialize 7-point Gaussian Quadrature class
        parthenon::math::quadrature::gauss<Real, 7> quad;
        // Solve for density and pressure using 7-point Gaussian Quadrature
        const Real rho = quad.integrate(rho_profile, bottom_offset, top_offset) /
                         (top_offset - bottom_offset);
        const Real pressure = p0 + jet_init_struct.const_accel *
                                       quad.integrate(rho_profile, 0.0, offset_avg);
        // Check that real density and pressure were calculated
        PARTHENON_REQUIRE(rho > 0.0, "Jet initialization produced negative density");
        PARTHENON_REQUIRE(pressure > 0.0,
                          "Jet initialization produced negative pressure");

        // Set cell conserved variables
        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = pressure / gm1;

        // Initialize tracer if enabled
        if (jet_init_struct.enable_tracer) {
          u(jet_init_struct.nhydro, k, j, i) = 0.0;
        }

        // Set magnetic fields if enabled
        if (jet_init_struct.fluid == Fluid::glmmhd) {
          u(IB1, k, j, i) = jet_init_struct.b0;
          u(IB2, k, j, i) = 0.0;
          u(IB3, k, j, i) = 0.0;
          // Update energy
          u(IEN, k, j, i) += 0.5 * SQR(jet_init_struct.b0);
        }
      });
}

void CalculateJetVolume(MeshData<Real> *md) {
  MeshBlock *pmb = md->GetBlockData(0)->GetBlockPointer();
  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg = pmb->packages.Get("Hydro");
  const Real inject_radius = hydro_pkg->Param<Real>("jet_inject_radius");
  const Real inject_height = hydro_pkg->Param<Real>("jet_inject_height");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  Real inject_volume = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Calculate Jet Injection Volume",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &vol_sum) {
        const auto &coords = cons_pack.GetCoords(b);
        if (Kokkos::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k))) < inject_radius &&
            coords.Xc<2>(j) < inject_height) {
          vol_sum += coords.CellVolume(k, j, i);
        }
      },
      Kokkos::Sum<Real>(inject_volume));

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &inject_volume, 1, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL
  // Update hydro package value
  hydro_pkg->UpdateParam("jet_inject_volume", inject_volume);
}

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  // Loop over all blocks in the mesh
  for (int b = 0; b < md->NumBlocks(); b++) {
    // Get current mesh block and hydro package
    MeshBlock *pmb = md->GetBlockData(b)->GetBlockPointer();
    std::shared_ptr<parthenon::StateDescriptor> hydro_pkg = pmb->packages.Get("Hydro");

    // Get index ranges for cells and add to struct
    IndexRangeStruct index_ranges;
    index_ranges.ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    index_ranges.jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    index_ranges.kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // Initialize the conserved variables
    ParArrayND<double, parthenon::VariableState> &u =
        pmb->meshblock_data.Get()->Get("cons").data;

    // Get coordinates
    parthenon::Coordinates_t &coords = pmb->coords;

    // Read jet parameters, check, and add to structure
    JetInitStruct jet_init_struct;
    jet_init_struct.fluid = hydro_pkg->Param<Fluid>("fluid");
    jet_init_struct.const_accel = hydro_pkg->Param<Real>("const_accel");
    jet_init_struct.gamma = pin->GetReal("hydro", "gamma");
    jet_init_struct.x2_min = hydro_pkg->Param<Real>("x2_min");
    jet_init_struct.enable_tracer = hydro_pkg->Param<bool>("enable_tracer");
    jet_init_struct.nhydro = hydro_pkg->Param<int>("nhydro");

    // Read and check in density data
    jet_init_struct.rho_0 = pin->GetReal("problem/jet", "rho_0");
    PARTHENON_REQUIRE(jet_init_struct.rho_0 > 0.0, "Input Invalid: rho_0 <= 0");
    jet_init_struct.rho_ref = pin->GetReal("problem/jet", "rho_ref");
    PARTHENON_REQUIRE(jet_init_struct.rho_ref > 0.0, "Input Invalid: rho_ref <= 0");
    jet_init_struct.r_ref = pin->GetReal("problem/jet", "r_ref");
    PARTHENON_REQUIRE(jet_init_struct.r_ref > 0.0, "Input Invalid: r_ref <= 0");
    jet_init_struct.rho_delta = jet_init_struct.rho_ref - jet_init_struct.rho_0;

    // Read in density profile mode and convert to enum. This is needed to safely pass
    // into Kokkos lambda
    jet_init_struct.rho_prof_mode = RhoProfileMap.at(
        pin->GetString("problem/jet", "rho_prof_mode", {"const", "lin", "pow", "expo"}));

    // Read magnetic field information if enabled
    if (jet_init_struct.fluid == Fluid::glmmhd) {
      jet_init_struct.b0 = pin->GetReal("problem/jet", "b0");
    }

    // Set initial conditions
    SetInitialConditions(pmb, u, coords, index_ranges, jet_init_struct);
  }
  // Calculate the jet injection volume
  CalculateJetVolume(md);
}

void HydroInject(
    const Real &dt,
    const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &cons_pack,
    const IndexRangeStruct &index_ranges, const HydroInjectStruct &jet_inject_struct) {
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "JetDriver::HydroInject", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, index_ranges.kb.s, index_ranges.kb.e, index_ranges.jb.s,
      index_ranges.jb.e, index_ranges.ib.s, index_ranges.ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Get cons variable pack
        parthenon::VariablePack<parthenon::Real> &cons = cons_pack(b);
        parthenon::Coordinates_t coords = cons.GetCoords();
        // Check if inside the jet injection volume
        if (Kokkos::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k))) <
                jet_inject_struct.radius &&
            coords.Xc<2>(j) < jet_inject_struct.height) {
          // Inject thermal energy, kinetic energy, and mass
          cons(IEN, k, j, i) += dt * jet_inject_struct.power_density *
                                (jet_inject_struct.q_frac + jet_inject_struct.ke_frac);
          cons(IM2, k, j, i) += dt * Kokkos::sqrt(2 * jet_inject_struct.rho_rate *
                                                  jet_inject_struct.power_density *
                                                  jet_inject_struct.ke_frac);
          cons(IDN, k, j, i) += dt * jet_inject_struct.rho_rate;

          // Update tracer if enabled
          if (jet_inject_struct.enable_tracer) {
            cons(jet_inject_struct.nhydro, k, j, i) = cons(IDN, k, j, i);
          }
        }
      });
}

void ConstructMagInjectPotential(
    const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &cons_pack,
    const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &A_pack,
    const IndexRangeStruct &index_ranges, const MagInjectStruct &mag_inject_struct,
    const Real &field_amp) {
  // Expand index ranges by one in all directions
  IndexRange a_ib = index_ranges.ib;
  a_ib.s -= 1;
  a_ib.e += 1;
  IndexRange a_jb = index_ranges.jb;
  a_jb.s -= 1;
  a_jb.e += 1;
  IndexRange a_kb = index_ranges.kb;
  a_kb.s -= 1;
  a_kb.e += 1;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "JetDriver::ConstructMagInjectPotential",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, a_kb.s, a_kb.e, a_jb.s,
      a_jb.e, a_ib.s, a_ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Get current variable packs and coordinates
        parthenon::VariablePack<Real> &A = A_pack(b);
        parthenon::Coordinates_t coords = cons_pack.GetCoords(b);

        // Calculate current radius and height from the bottom of the simulation
        const Real r = Kokkos::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k)));
        const Real h = coords.Xc<2>(j) - mag_inject_struct.x2_min;
        // Initialize potential in each direction to zero
        Real a1 = 0.0;
        Real a2 = 0.0;
        Real a3 = 0.0;
        // Update potential for loops
        if (mag_inject_struct.type == MagFieldInjectType::Loop) {
          // Check that current height is within inputs
          if (Kokkos::abs(h) >= mag_inject_struct.offset &&
              Kokkos::abs(h) <= mag_inject_struct.offset + mag_inject_struct.thickness) {
            // Update potential along the axis of the jet (x2)
            a2 = field_amp * mag_inject_struct.l_scale *
                 Kokkos::exp(-SQR(r / mag_inject_struct.l_scale));
          }
        }
        // Write final potentials to the A variable pack
        A(0, k, j, i) = a1;
        A(1, k, j, i) = a2;
        A(2, k, j, i) = a3;
      });
}

Real CalculateFieldAmplitude(
    const Real &dt,
    const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &cons_pack,
    const IndexRangeStruct &index_ranges, const HydroInjectStruct &jet_inject_struct,
    const MagInjectStruct &mag_inject_struct) {
  //
  const Real mag_energy = dt * jet_inject_struct.power_density *
                          mag_inject_struct.b_frac * jet_inject_struct.volume;
  //
  Real linear_contrib = 0.0;
  Real quadratic_contrib = 0.0;
  Kokkos::parallel_reduce(
      "JetFieldAmplitude",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          parthenon::DevExecSpace(),
          {0, index_ranges.kb.s, index_ranges.jb.s, index_ranges.ib.s},
          {cons_pack.GetDim(5), index_ranges.kb.e + 1, index_ranges.jb.e + 1,
           index_ranges.ib.e + 1},
          {1, 1, 1, index_ranges.ib.e + 1 - index_ranges.ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &llinear_contrib, Real &lquadratic_contrib) {
        //
        parthenon::VariablePack<Real> &cons = cons_pack(b);
        parthenon::Coordinates_t coords = cons_pack.GetCoords(b);
        //
        const Real r2 = SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k));
        const Real h = coords.Xc<2>(j) - mag_inject_struct.x2_min;
        const Real cell_volume = coords.CellVolume(k, j, i);
        //
        Real b1 = 0.0;
        Real b2 = 0.0;
        Real b3 = 0.0;
        //
        // TODO: Add check for loop and path for tower
        if (Kokkos::abs(h) >= mag_inject_struct.offset &&
            Kokkos::abs(h) <= mag_inject_struct.offset + mag_inject_struct.thickness) {
          const Real exp_r2 = Kokkos::exp(-r2 / SQR(mag_inject_struct.l_scale));
          b1 = 2.0 * coords.Xc<3>(k) / mag_inject_struct.l_scale * exp_r2;
          b2 = 0.0;
          b3 = -2.0 * coords.Xc<1>(i) / mag_inject_struct.l_scale * exp_r2;
          //
        }
        llinear_contrib += (cons(IB1, k, j, i) * b1 + cons(IB2, k, j, i) * b2 +
                            cons(IB3, k, j, i) * b3) *
                           cell_volume;
        lquadratic_contrib += 0.5 * (SQR(b1) + SQR(b2) + SQR(b3)) * cell_volume;
      },
      linear_contrib, quadratic_contrib);

// Sum the linear and quadratic contributions from across all processes
#ifdef MPI_PARALLEL
  Real magnetic_contribs[2] = {linear_contrib, quadratic_contrib};
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, magnetic_contribs, 2,
                                    MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
  linear_contrib = magnetic_contribs[0];
  quadratic_contrib = magnetic_contribs[1];
#endif // MPI_PARALLEL

  PARTHENON_REQUIRE(linear_contrib != 0.0 && quadratic_contrib != 0.0,
                    "Jet magnetic injection linear and quadratic contributions are both "
                    "zero");
  const Real disc =
      linear_contrib * linear_contrib + 4.0 * quadratic_contrib * mag_energy;

  PARTHENON_REQUIRE(disc >= 0.0 || quadratic_contrib != 0.0,
                    "Jet magnetic injection has no viable field amplitude.");

  return (-linear_contrib + Kokkos::sqrt(disc)) / (2.0 * quadratic_contrib);
}

void ApplyMagInjectPotential(
    const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &cons_pack,
    const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &A_pack,
    const IndexRangeStruct &index_ranges) {
  // Take the curl of the potential and apply the new magnetic field
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "JetDriver::ApplyMagInjectPotential",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, index_ranges.kb.s,
      index_ranges.kb.e, index_ranges.jb.s, index_ranges.jb.e, index_ranges.ib.s,
      index_ranges.ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Get current variable packs and coordinates
        parthenon::VariablePack<Real> &cons = cons_pack(b);
        parthenon::VariablePack<Real> &A = A_pack(b);
        parthenon::Coordinates_t coords = cons.GetCoords();

        // Curl potential into magnetic field
        const Real b1 = (A(2, k, j + 1, i) - A(2, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 -
                        (A(1, k + 1, j, i) - A(1, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
        const Real b2 = (A(0, k + 1, j, i) - A(0, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0 -
                        (A(2, k, j, i + 1) - A(2, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
        const Real b3 = (A(1, k, j, i + 1) - A(1, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
                        (A(0, k, j + 1, i) - A(0, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;

        // Add magnetic energy density to overall energy density
        cons(IEN, k, j, i) += cons(IB1, k, j, i) * b1 + cons(IB2, k, j, i) * b2 +
                              cons(IB3, k, j, i) * b3 +
                              0.5 * (SQR(b1) + SQR(b2) + SQR(b3));

        // Add the magnetic field to the conserved variables
        cons(IB1, k, j, i) += b1;
        cons(IB2, k, j, i) += b2;
        cons(IB3, k, j, i) += b3;
      });
}

void JetDriver(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  // Get base variables from mesh data pointer
  const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &cons_pack =
      md->PackVariables(std::vector<std::string>{"cons"});
  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg =
      md->GetMeshPointer()->packages.Get("Hydro");
  // Get bounds and add to structure
  IndexRangeStruct index_ranges;
  index_ranges.ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  index_ranges.jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  index_ranges.kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  // Get variables for hydro injection and add to structure if needed
  const Real m_inject_rate = hydro_pkg->Param<Real>("jet_m_inject_rate");
  const Real power = hydro_pkg->Param<Real>("jet_power");
  HydroInjectStruct jet_inject_struct;
  jet_inject_struct.radius = hydro_pkg->Param<Real>("jet_inject_radius");
  jet_inject_struct.height = hydro_pkg->Param<Real>("jet_inject_height");
  jet_inject_struct.volume = hydro_pkg->Param<Real>("jet_inject_volume");
  jet_inject_struct.ke_frac = hydro_pkg->Param<Real>("jet_ke_frac");
  jet_inject_struct.q_frac = hydro_pkg->Param<Real>("jet_q_frac");
  jet_inject_struct.enable_tracer = hydro_pkg->Param<bool>("enable_tracer");
  jet_inject_struct.nhydro = hydro_pkg->Param<int>("nhydro");

  // Apply a constant acceleration
  const_accel::ConstantAccel(md, tm, dt);

  // Calculate density and energy density injection rates. If the injection volume is
  // zero set the injection rates to zero.
  jet_inject_struct.rho_rate =
      (jet_inject_struct.volume > 0.0) ? (m_inject_rate / jet_inject_struct.volume) : 0.0;
  jet_inject_struct.power_density =
      (jet_inject_struct.volume > 0.0) ? (power / jet_inject_struct.volume) : 0.0;

  // Inject momentum and thermal/kinetic energy
  HydroInject(dt, cons_pack, index_ranges, jet_inject_struct);

  // Check if magnetic fields are enabled and if they are being injected
  if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd) {
    MagInjectStruct mag_inject_struct;
    mag_inject_struct.b_frac = hydro_pkg->Param<Real>("jet_b_frac");
    if (mag_inject_struct.b_frac > 0.0) {
      // Get variables for magnetic field injection and add to structure if needed
      const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &A_pack =
          md->PackVariables(std::vector<std::string>{"mag_field_inject_A"});
      mag_inject_struct.x2_min = hydro_pkg->Param<Real>("x2_min");
      mag_inject_struct.type =
          hydro_pkg->Param<MagFieldInjectType>("mag_field_inject_type");
      mag_inject_struct.l_scale = hydro_pkg->Param<Real>("mag_field_inject_l_scale");
      mag_inject_struct.offset = hydro_pkg->Param<Real>("mag_field_inject_offset");
      mag_inject_struct.thickness = hydro_pkg->Param<Real>("mag_field_inject_thickness");

      // Calculate field amplitude based on target magnetic energy
      Real field_amp = CalculateFieldAmplitude(dt, cons_pack, index_ranges,
                                               jet_inject_struct, mag_inject_struct);
      // Calculate potential with new field amplitude
      ConstructMagInjectPotential(cons_pack, A_pack, index_ranges, mag_inject_struct,
                                  field_amp);
      // Apply potential to magnetic field
      ApplyMagInjectPotential(cons_pack, A_pack, index_ranges);
    }
  }
}

} // namespace jet
