// SPDX-License-Identifier: AGPL-3.0-or-later
//
// PyRITE #510 benchmark tally for PenRed; see tallyPyritePrimary.hh.

#include "tallyPyritePrimary.hh"

void pen_tallyPyritePrimary::flush(){}

void pen_tallyPyritePrimary::tally_beginSim(){
  fout = fopen("pyrite-primary.dat","w");
  if(fout == nullptr){
    printf("pen_tallyPyritePrimary: unable to open pyrite-primary.dat\n");
    throw std::runtime_error("pyrite-primary.dat");
  }
}

void pen_tallyPyritePrimary::tally_endSim(const unsigned long long /*nhist*/){
  if(fout != nullptr){
    fclose(fout);
    fout = nullptr;
  }
}

void pen_tallyPyritePrimary::tally_beginPart(const unsigned long long nhist,
					     const unsigned /*kdet*/,
					     const pen_KPAR kpar,
					     const pen_particleState& state){
  // Direct child of the primary electron, stacked by hard bremsstrahlung. The
  // body photon EABS exceeds E0, so it is absorbed before any transport.
  if(kpar == PEN_PHOTON && state.ILB[0] == 2 &&
     state.ILB[1] == PEN_ELECTRON && state.ILB[2] == BETAe_HARD_BREMSSTRAHLUNG){
    fprintf(fout,"G %llu %.17g\n",nhist,state.E);
  }
}

void pen_tallyPyritePrimary::tally_endPart(const unsigned long long nhist,
					   const pen_KPAR kpar,
					   const pen_particleState& state){
  if(state.ILB[0] != 1){
    const bool scored = kpar == PEN_PHOTON && state.ILB[0] == 2 &&
      state.ILB[1] == PEN_ELECTRON && state.ILB[2] == BETAe_HARD_BREMSSTRAHLUNG;
    if(!scored)
      fprintf(fout,"S %llu %d %d\n",nhist,static_cast<int>(kpar),state.ILB[2]);
    return;
  }
  if(kpar != PEN_ELECTRON){
    fprintf(fout,"S %llu %d %d\n",nhist,static_cast<int>(kpar),-1);
    return;
  }
  int terminal = 7;
  if(state.MAT == 0){
    if(state.Z >= zmax - tolerance) terminal = 5;
    else if(state.Z <= zmin + tolerance) terminal = 4;
    else terminal = 6;
  }
  fprintf(fout,"T %llu %d %.17g\n",nhist,terminal,state.E);
}

int pen_tallyPyritePrimary::configure(const wrapper_geometry& /*geometry*/,
				      const abc_material* const /*materials*/[constants::MAXMAT],
				      const pen_parserSection& config,
				      const unsigned verbose){
  if(config.read("zmin",zmin) != INTDATA_SUCCESS ||
     config.read("zmax",zmax) != INTDATA_SUCCESS ||
     config.read("tolerance",tolerance) != INTDATA_SUCCESS){
    if(verbose > 0)
      printf("pen_tallyPyritePrimary: 'zmin', 'zmax' and 'tolerance' [cm] are required\n");
    return -1;
  }
  if(!(zmax > zmin) || !(tolerance > 0.0) || !(2.0*tolerance < zmax - zmin)){
    if(verbose > 0)
      printf("pen_tallyPyritePrimary: invalid slab faces or tolerance\n");
    return -2;
  }
  return 0;
}

void pen_tallyPyritePrimary::saveData(const unsigned long long /*nhist*/) const{}
int pen_tallyPyritePrimary::sumTally(const pen_tallyPyritePrimary& /*tally*/){return 0;}

REGISTER_COMMON_TALLY(pen_tallyPyritePrimary)
