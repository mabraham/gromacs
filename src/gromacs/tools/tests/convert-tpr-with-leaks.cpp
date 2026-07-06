/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2024- The GROMACS Authors
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
/*! \internal \file
 * \brief
 * Tests for gmx convert-tpr that cause memory leaks
 *
 * Some legacy features/functions invoked in convert-tpr have memory leaks,
 * which are picked up in AddressSanitizer runs.
 * To avoid blanket suppression, the tests causing the leaks are isolated here.
 *
 * In particular, the following leaf-level functions are known to leak in
 * these tests: atomcat (mtop_util.cpp), do_atoms (tpxio.cpp), rd_groups
 * (topology/index.cpp).
 *
 * These functions are called and/or cause leaks when slicing the topology
 * according to an index group.
 *
 * \author Eliane Briand <eliane@br.iand.fr>
 */
#include "gmxpre.h"

#include <filesystem>
#include <vector>

#include "gromacs/fileio/tpxio.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/tools/convert_tpr.h"
#include "gromacs/topology/topology.h"

#include "testutils/cmdlinetest.h"
#include "testutils/stdiohelper.h"
#include "testutils/testfilemanager.h"
#include "testutils/tprfilegenerator.h"

#include "convert-tpr-fixture.h"

namespace gmx
{
namespace test
{
namespace
{

TEST_F(ConvertTprTest, selectIndexTest)
{
    gmx_mtop_t top;
    t_inputrec ir;
    t_state    state;
    read_tpx_state(tprFileHandle.tprName(), &ir, &state, &top);

    ASSERT_EQ(top.natoms, 156);

    TestFileManager             fileManager;
    const std::filesystem::path outTprFilename = fileManager.getTemporaryFilePath("extended.tpr");
    const std::string           command[]      = {
        "convert-tpr", "-s", tprFileHandle.tprName(), "-o", outTprFilename.string()
    };
    CommandLine cmdline(command);

    StdioTestHelper stdioHelper(&fileManager);
    stdioHelper.redirectStringToStdin("Protein-H\n");

    gmx::test::CommandLineTestHelper::runModuleFactory(&gmx::ConvertTprInfo::create, &cmdline);

    {
        gmx_mtop_t top_after;
        t_inputrec ir_after;
        t_state    state_after;
        read_tpx_state(outTprFilename, &ir_after, &state_after, &top_after);
        EXPECT_EQ(top_after.natoms, 75);
    }
}

TEST_F(ConvertTprNoVelocityTest, selectIndexTestWithoutVelocity)
{
    gmx_mtop_t top;
    t_inputrec ir;
    t_state    state;
    read_tpx_state(tprFileHandle.tprName(), &ir, &state, &top);

    ASSERT_EQ(top.natoms, 156);

    TestFileManager             fileManager;
    const std::filesystem::path outTprFilename = fileManager.getTemporaryFilePath("extended.tpr");
    const std::string           command[]      = {
        "convert-tpr", "-s", tprFileHandle.tprName(), "-o", outTprFilename.string()
    };
    CommandLine cmdline(command);

    StdioTestHelper stdioHelper(&fileManager);
    stdioHelper.redirectStringToStdin("Protein-H\n");

    gmx::test::CommandLineTestHelper::runModuleFactory(&gmx::ConvertTprInfo::create, &cmdline);

    {
        gmx_mtop_t top_after;
        t_inputrec ir_after;
        t_state    state_after;
        read_tpx_state(outTprFilename, &ir_after, &state_after, &top_after);
        EXPECT_EQ(top_after.natoms, 75);
    }
}

TEST_F(ConvertTprTest, SubsetAtomsTest)
{
    gmx_mtop_t top;
    t_inputrec ir;
    t_state    state;
    read_tpx_state(tprFileHandle.tprName(), &ir, &state, &top);

    TestFileManager             fileManager;
    const std::filesystem::path ndxFilename = TestFileManager::getInputFilePath("subset.ndx");

    const std::filesystem::path outTprFilename = fileManager.getTemporaryFilePath("keep.tpr");
    const std::string command[] = { "convert-tpr",        "-s", tprFileHandle.tprName(), "-n",
                                    ndxFilename.string(), "-o", outTprFilename.string() };
    CommandLine       cmdline(command);

    StdioTestHelper stdioHelper(&fileManager);
    stdioHelper.redirectStringToStdin("0\n");

    gmx::test::CommandLineTestHelper::runModuleFactory(&gmx::ConvertTprInfo::create, &cmdline);

    {
        gmx_mtop_t topAfter;
        t_inputrec irAfter;
        t_state    stateAfter;
        read_tpx_state(outTprFilename, &irAfter, &stateAfter, &topAfter);

        const t_atoms& atomsBefore = top.moltype[0].atoms;
        const t_atoms& atomsAfter  = topAfter.moltype[0].atoms;

        // Atom 40 has original resind 2 (PHE), atoms 0 and 1 have original resind 0 (LYS),
        // and atoms 24 and 25 have original resind 1 (VAL).
        // Under the old bug where atoms->atom[i-1].resind was compared after being overwritten with
        // the remapped nr, atom 0 (resind 0) matched remapped nr 0 of atom 40, falsely merging
        // residue 0 into residue 2, and atom 24 (resind 1) matched remapped nr 1, falsely merging it.
        const std::vector<int> selectedIndices = { 40, 0, 1, 24, 25 };

        EXPECT_EQ(atomsAfter.nr, static_cast<int>(selectedIndices.size()));

        for (size_t i = 0; i < selectedIndices.size(); ++i)
        {
            EXPECT_STREQ(*(atomsAfter.atomname[i]), *(atomsBefore.atomname[selectedIndices[i]]));
            EXPECT_EQ(atomsAfter.atom[i].m, atomsBefore.atom[selectedIndices[i]].m);
            EXPECT_EQ(atomsAfter.atom[i].q, atomsBefore.atom[selectedIndices[i]].q);
            EXPECT_EQ(atomsAfter.atom[i].type, atomsBefore.atom[selectedIndices[i]].type);
            EXPECT_EQ(atomsAfter.atom[i].typeB, atomsBefore.atom[selectedIndices[i]].typeB);
        }

        ASSERT_TRUE(atomsBefore.haveType);
        EXPECT_TRUE(atomsAfter.haveType);
        ASSERT_NE(atomsBefore.atomtype, nullptr);
        ASSERT_NE(atomsAfter.atomtype, nullptr);
        for (size_t i = 0; i < selectedIndices.size(); ++i)
        {
            EXPECT_STREQ(*(atomsAfter.atomtype[i]), *(atomsBefore.atomtype[selectedIndices[i]]));
        }

        ASSERT_TRUE(atomsBefore.haveBState);
        EXPECT_TRUE(atomsAfter.haveBState);
        ASSERT_NE(atomsBefore.atomtypeB, nullptr);
        ASSERT_NE(atomsAfter.atomtypeB, nullptr);
        for (size_t i = 0; i < selectedIndices.size(); ++i)
        {
            EXPECT_STREQ(*(atomsAfter.atomtypeB[i]), *(atomsBefore.atomtypeB[selectedIndices[i]]));
        }

        // Verify residue indices and residue info are correctly remapped across residue boundaries
        EXPECT_EQ(atomsAfter.nres, 3);
        EXPECT_EQ(atomsAfter.atom[0].resind, 0);
        EXPECT_EQ(atomsAfter.atom[1].resind, 1);
        EXPECT_EQ(atomsAfter.atom[2].resind, 1);
        EXPECT_EQ(atomsAfter.atom[3].resind, 2);
        EXPECT_EQ(atomsAfter.atom[4].resind, 2);

        EXPECT_STREQ(*(atomsAfter.resinfo[0].name), *(atomsBefore.resinfo[2].name));
        EXPECT_EQ(atomsAfter.resinfo[0].nr, atomsBefore.resinfo[2].nr);
        EXPECT_EQ(atomsAfter.resinfo[0].chainnum, atomsBefore.resinfo[2].chainnum);
        EXPECT_EQ(atomsAfter.resinfo[0].chainid, atomsBefore.resinfo[2].chainid);

        EXPECT_STREQ(*(atomsAfter.resinfo[1].name), *(atomsBefore.resinfo[0].name));
        EXPECT_EQ(atomsAfter.resinfo[1].nr, atomsBefore.resinfo[0].nr);
        EXPECT_EQ(atomsAfter.resinfo[1].chainnum, atomsBefore.resinfo[0].chainnum);
        EXPECT_EQ(atomsAfter.resinfo[1].chainid, atomsBefore.resinfo[0].chainid);

        EXPECT_STREQ(*(atomsAfter.resinfo[2].name), *(atomsBefore.resinfo[1].name));
        EXPECT_EQ(atomsAfter.resinfo[2].nr, atomsBefore.resinfo[1].nr);
        EXPECT_EQ(atomsAfter.resinfo[2].chainnum, atomsBefore.resinfo[1].chainnum);
        EXPECT_EQ(atomsAfter.resinfo[2].chainid, atomsBefore.resinfo[1].chainid);
    }
}

} // namespace
} // namespace test
} // namespace gmx
