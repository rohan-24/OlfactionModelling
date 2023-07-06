/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __PPrj_included
#define __PPrj_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"
#include "Prj.h"

class PPrj : public Prj {


protected:

    friend class PPrjR;

 public:

    PPrj(int srcH,int srcM,int srcU,float srcspkwgain,int trgH,int trgM,int trgU,float trgspkwgain,
	float pdens,Globals::BCPvar_t bcpvar = Globals::FIXED,
    	Globals::Prjarch_t prjarch = Globals::HPATCHY) ;

    void propagate() ;
    
    void updZj() ;

    void fetchntpatch(std::vector<int> &ntpatch) ;

    void update() ;

} ;

#endif // __PPrj_included

