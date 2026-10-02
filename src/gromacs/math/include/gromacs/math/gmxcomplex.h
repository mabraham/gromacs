/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 1991- The GROMACS Authors
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
#ifndef GMX_MATH_GMXCOMPLEX_H
#define GMX_MATH_GMXCOMPLEX_H

#include <cmath>

#include "gromacs/utility/real.h"

struct t_complex
{
    real re, im;
};

static t_complex rcmul(real r, t_complex c)
{
    return { r * c.re, r * c.im };
}

static inline t_complex rcexp(real r)
{
    return { std::cos(r), std::sin(r) };
}


static inline t_complex cadd(t_complex a, t_complex b)
{
    return { a.re + b.re, a.im + b.im };
}

static inline t_complex csub(t_complex a, t_complex b)
{
    return { a.re - b.re, a.im - b.im };
}

static t_complex cmul(t_complex a, t_complex b)
{
    return { a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}

static t_complex conjugate(t_complex c)
{
    return { c.re, -c.im };
}

static inline real cabs2(t_complex c)
{
    return (c.re * c.re) + (c.im * c.im);
}

static inline t_complex cdiv(t_complex teller, t_complex noemer)
{
    t_complex anoemer = cmul(conjugate(noemer), noemer);
    t_complex res     = cmul(teller, conjugate(noemer));

    return rcmul(1.0 / anoemer.re, res);
}

inline bool operator==(const t_complex& lhs, const t_complex& rhs)
{
    return (lhs.re == rhs.re) && (lhs.im == rhs.im);
}
inline bool operator!=(const t_complex& lhs, const t_complex& rhs)
{
    return !(lhs == rhs);
}

#endif
