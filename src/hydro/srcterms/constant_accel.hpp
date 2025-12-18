//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file constant_accel.hpp
// \brief Hydro source term for constant acceleration in a given direction
//========================================================================================
#ifndef HYDRO_SRCTERMS_CONSTANT_ACCEL_HPP_
#define HYDRO_SRCTERMS_CONSTANT_ACCEL_HPP_

// AthenaPK headers
#include "../../main.hpp"

using parthenon::IndexDomain;
using parthenon::IndexRange;
using parthenon::MeshData;
using parthenon::Real;

namespace const_accel {
/**
 * @brief Apply a constant acceleration in a given direction
 *
 * @param md Mesh data pointer
 * @param tm Simulation time struct
 * @param dt Time step size
 */
void ConstantAccelSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                          const Real dt);

} // namespace const_accel

#endif // HYDRO_SRCTERMS_CONSTANT_ACCEL_HPP_