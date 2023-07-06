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
#include <cmath>

#include "omp.h"

#include "Globals.h"
#include "PGlobals.h"
#include "Logger.h"
#include "PrjH.h"
#include "Prj.h"
#include "PPrj.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;


PPrj::PPrj(int srcH,int srcM,int srcU,float srcspkwgain,int trgH,int trgM,int trgU,float trgspkwgain,
	   float pdens,BCPvar_t bcpvar,Prjarch_t prjarch)
    : Prj(srcH,srcM,srcU,srcspkwgain,trgH,trgM,trgU,trgspkwgain,pdens,bcpvar,prjarch) {

}


void PPrj::propagate() {

    if (updbw) updBW();

    resetbwsup();    

    for (int srcn=0; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL or connijs[srcn]->silent) continue;

	connijs[srcn]->updbwsup(delact[srcn],srcspkwgain,bwsup);

    }

    for (size_t n=0; n<trgNh; n++) cond[n] += (bwsup[n] - cond[n]) * tauconddt;

}


void PPrj::updZj() {

    for (int trgn=0; trgn<trgNh; trgn++)

	Zj[trgn] += (((1 - eps) * trgact[trgn] + eps) * trgspkwgain - Zj[trgn]) * tauzjdt;

}


void PPrj::fetchntpatch(vector<int> &ntpatch) {

    for (int srcn=0; srcn<srcN; srcn++)

	if (connijs[srcn]!=NULL and not connijs[srcn]->silent) ntpatch[srcn]++;

}


void PPrj:: update() {

    resetbwsup();	

    propagate();

    updP();

    updsdep();

    updsfac();

    if (prn>0) updconnijs();

}
