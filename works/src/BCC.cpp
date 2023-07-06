/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#include <stdlib.h>
#include <vector>
#include <string>
#include <random>
#include <limits>

#include "Globals.h"
#include "Parseparam.h"
#include "BCU.h"
#include "PrjH.h"
#include "Logger.h"
#include "BCC.h"

using namespace std;
using namespace Globals;


BCC::BCC(BCU *srcbcu,BCU *trgbcu,float dens,BCPvar_t bcpvar,Prjarch_t prjarch)
    : PrjH(srcbcu,trgbcu,dens,bcpvar,prjarch) {

}
