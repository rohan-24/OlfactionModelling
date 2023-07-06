/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __BCC_included
#define __BCC_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"

class BCU;


class BCC : public PrjH {

 protected:

 public:

    BCC(BCU *srcbcu,BCU *trgbcu,float dens = 1,Globals::BCPvar_t bcpvar = Globals::INCR,
	Globals::Prjarch_t bccarch = Globals::HPATCHY) ;

} ;

#endif // __BCC_included

