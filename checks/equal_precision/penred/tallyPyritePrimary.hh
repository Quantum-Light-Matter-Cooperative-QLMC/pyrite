// SPDX-License-Identifier: AGPL-3.0-or-later
//
// PyRITE #510 benchmark tally for PenRed (PENELOPE-2018 physics).
// Kept outside PenRed and copied into an isolated source tree by
// prepare_penred.py; it links into AGPL PenRed and carries its license.
//
// One record per event, streamed to `pyrite-primary.dat` in the run directory:
//   T <nhist> <terminal> <E_eV>  primary electron end; terminal 5 forward exit,
//                                4 backward exit, 6 side exit, 7 absorbed (E < EABS)
//   G <nhist> <E_eV>             hard bremsstrahlung photon of the primary, at creation
//   S <nhist> <kpar> <ILB2>      any other secondary that ended (diagnostic count)
// Only a single-threaded run gives one writer.

#ifndef __PYRITE_PRIMARY_TALLY__
#define __PYRITE_PRIMARY_TALLY__

#include <cstdio>

class pen_tallyPyritePrimary : public pen_genericTally<pen_particleState> {
  DECLARE_TALLY(pen_tallyPyritePrimary,pen_particleState,PYRITE_PRIMARY)

  private:
  FILE* fout = nullptr;
  double zmin = 0.0;
  double zmax = 0.0;
  double tolerance = 0.0;

public:

  pen_tallyPyritePrimary() : pen_genericTally(USE_BEGINSIM |
					      USE_BEGINPART |
					      USE_ENDPART |
					      USE_ENDSIM)
  {}

  void tally_beginSim();
  void tally_endSim(const unsigned long long nhist);
  void tally_beginPart(const unsigned long long nhist,
		       const unsigned kdet,
		       const pen_KPAR kpar,
		       const pen_particleState& state);
  void tally_endPart(const unsigned long long nhist,
		     const pen_KPAR kpar,
		     const pen_particleState& state);

  int configure(const wrapper_geometry& geometry,
		const abc_material* const materials[constants::MAXMAT],
		const pen_parserSection& config,
		const unsigned verbose);

  void saveData(const unsigned long long nhist) const;
  void flush();
  int sumTally(const pen_tallyPyritePrimary& tally);
};

#endif
