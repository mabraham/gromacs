/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2010- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \file
 * \brief
 * Declares common types used in selections.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_selection
 */
#ifndef GMX_SELECTION_SELECTIONENUMS_H
#define GMX_SELECTION_SELECTIONENUMS_H

#include "gromacs/utility/flags.h"

/*! \brief
 * Defines the type of covered fraction.
 *
 * \inpublicapi
 */
typedef enum
{
    CFRAC_NONE,      /**< No covered fraction (everything covered). */
    CFRAC_SOLIDANGLE /**< Fraction of a solid (3D) angle covered. */
} e_coverfrac_t;

namespace gmx
{

/*! \cond internal */
/*! \brief
 * Flags for options.
 *
 * These flags are not part of the public interface, even though they are in an
 * installed header.  They are needed in the implementation of SelectionOption.
 */
enum SelectionFlag : uint64_t
{
    efSelection_OnlyStatic = 1 << 0,
    efSelection_OnlyAtoms  = 1 << 1,
    efSelection_OnlySorted = 1 << 2,
    //! Whether ::POS_MASKONLY should be used for output position evaluation.
    efSelection_DynamicMask = 1 << 3,
    //! If set, unconditionally empty selections result in compilation errors.
    efSelection_DisallowEmpty = 1 << 4,
    //! Whether velocities of output positions should be evaluated.
    efSelection_EvaluateVelocities = 1 << 5,
    //! Whether forces on output positions should be evaluated.
    efSelection_EvaluateForces = 1 << 6,
};

//! Holds a collection of ::SelectionFlag values.
typedef FlagsTemplate<SelectionFlag> SelectionFlags;
//! \endcond

/*! \brief
 * Describes topology properties required for selection evaluation.
 *
 * See SelectionCollection::requiredTopologyProperties().
 *
 * \inpublicapi
 * \ingroup module_selection
 */
struct SelectionTopologyProperties
{
    //! Returns a property object that requires generic topology info.
    static SelectionTopologyProperties topology()
    {
        return SelectionTopologyProperties(true, false);
    }
    //! Returns a property object that requires atom masses.
    static SelectionTopologyProperties masses() { return SelectionTopologyProperties(true, true); }

    //! Initializes properties that does not require anything.
    SelectionTopologyProperties() : needsTopology_(false), needsMasses_(false) {}
    //! Initializes properties with the given flags.
    SelectionTopologyProperties(bool needsTopology, bool needsMasses) :
        needsTopology_(needsTopology), needsMasses_(needsMasses)
    {
    }

    //! Combines flags from another properties object to this.
    void merge(const SelectionTopologyProperties& other)
    {
        needsTopology_ = needsTopology_ || other.needsTopology_;
        needsMasses_   = needsMasses_ || other.needsMasses_;
    }
    //! Whether all flags are `true` (for short-ciruiting logic).
    bool hasAll() const { return needsTopology_ && needsMasses_; }
    //! Whether any flag is `true`.
    bool hasAny() const { return needsTopology_ || needsMasses_; }

    //! Whether topology information is needed for selection evaluation.
    bool needsTopology_;
    //! Whether atom masses are needed for selection evaluation.
    bool needsMasses_;
};

} // namespace gmx

#endif
