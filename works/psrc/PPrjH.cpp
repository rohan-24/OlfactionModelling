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
#include "Logger.h"
#include "PrjH.h"
#include "PPrj.h"
#include "PPrjH.h"

using namespace std;
using namespace Globals;


PPrjH::PPrjH(int nprj0,int srcH,int srcM,int srcU,float srcspkwgain,
	     int trgH,int trgHperR,int trgM,int trgU,int trghoffs,float trgspkwgain,
	     float pdens,BCPvar_t bcpvar,Prjarch_t prjarch)
    : PrjH(srcH,srcM,srcU,srcspkwgain,trgH,trgM,trgU,trgspkwgain,pdens,bcpvar,prjarch) {

    ntpatch = vector<int>(srcN,0);

    gntpatch = vector<int>(srcN,0);

    for (int trgh=0; trgh<trgHperR; trgh++) {

	prjs.push_back(new PPrj(srcH,srcM,srcU,srcspkwgain,1,trgM,trgU,trgspkwgain,
				pdens,bcpvar,prjarch));

	prjs.back()->prjh = this;

	prjs.back()->trgH = trgH;

	this->trgHperR = trgHperR;

	prjs.back()->settrghnoffs(trghoffs,(trghoffs + trgh) * trgM * trgU);

	prjs.back()->id = nprj0 + trgh;

	prjs.back()->setseed(nprj0 + trgh + 1);

	prjs.back()->configwon(0);

	((PPrj *)prjs.back())->fetchntpatch(ntpatch);

    }
}


void PPrjH::prnidandseed() {
    
    for (size_t p=0; p<prjs.size(); p++) {

	printf("%4d %9ld\n",prjs[p]->getinfo("id"),prjs[p]->getseed());

	fflush(stdout);

    }
}

int PPrjH::gettrghoffs(int trgh) {

    assertgtelt(trgh,0,trgHperR);

    return prjs[trgh]->trghoffs;

}


int PPrjH::gettrgnoffs(int trgh) {

    assertgtelt(trgh,0,trgHperR);

    return prjs[trgh]->trgnoffs;

}


void PPrjH::setdelays(std::vector<vector<float> > delaymat) {

    for (int trgh=0; trgh<trgHperR; trgh++) prjs[trgh]->setdelays(delaymat);

}


void PPrjH::setstrcpl(bool on) {

    for (int trgh=0; trgh<trgHperR; trgh++) prjs[trgh]->setstrcpl(on);

}


void PPrjH::fetchntpatch() {

    fill(ntpatch.begin(),ntpatch.end(),0);

    for (int trgh=0; trgh<trgHperR; trgh++)

	((PPrj *)prjs[trgh])->fetchntpatch(ntpatch);

}


vector<int> const &PPrjH::getntpatch() { return gntpatch; }


int PPrjH::getntpatch(int srcn) { return gntpatch[srcn]; }


void PPrjH::fliptpatchn(int nflp,int updint) {

    for (int trgh=0; trgh<trgHperR; trgh++) ((PPrj *)prjs[trgh])->qsiltoactn(nflp,updint);

}
