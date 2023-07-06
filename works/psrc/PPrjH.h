/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __PPrjH_included
#define __PPrjH_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"

class PrjH;

class PPrjH : public PrjH {

protected:

    int trgHperR;

    std::vector<int> ntpatch,gntpatch;
    
    friend class PPrjR;

    friend class PProjection;

 public:

    PPrjH(int nprj0,int srcH,int srcM,int srcU,float srcspkwgain,
	  int trgH,int trgHperR,int trgM,int trgU,int trghoffs,float trgspkwgain,
	  float pdens,Globals::BCPvar_t bcpvar = Globals::FIXED,
	  Globals::Prjarch_t prjarch = Globals::HPATCHY) ;


    void prnidandseed() ;
	
    int gettrghoffs(int trgh) ;

    int gettrgnoffs(int trgh) ;

    void setdelays(std::vector<std::vector<float> > delaymat) ;

    void fetchntpatch() ;

    std::vector<int> const &getntpatch() ;

    // Fetches gntpatch, name inherited from PrjH
    int getntpatch(int srcn) ;

    void setstrcpl(bool on) ;

    void fliptpatchn(int nflp,int updint) ;

} ;

#endif // __PPrj_included

