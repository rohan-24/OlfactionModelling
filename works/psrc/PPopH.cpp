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
#include "Pop.h"
#include "PopH.h"
#include "PIO.h"
#include "PPopH.h"

using namespace std;
using namespace Globals;


PPopH::PPopH(int H,int M,int U,Actfn_t actfn_t,Normfn_t normfn_t)
    : PopH(H,M,U,actfn_t,normfn_t) {

}


void PPopH::populate(int H,int M,int U,Actfn_t actfn_t,Normfn_t normfn_t) {

    for (int h=0; h<H; h++) {
	
	switch (actfn_t) {
	case LIN:
	case SPKLIN:
	case LOG:
	case SPKLOG:
	case SIG:
	case SPKSIG:
	case AREG: pops.push_back(new Pop(M,U,actfn_t,normfn_t)); break;

	case SPKEXP:
	case EXP: pops.push_back(new ExpPop(M,U,actfn_t,normfn_t)); break;
	    
	case SPKBCP:
	case BCP: pops.push_back(new BCPPop(M,U,actfn_t,normfn_t)); break;
	    
	case ALIF:
	case AdEx:
	case AdExS: pops.push_back(new SNNPop(M,actfn_t)); break;

	default: error("PPopH::populate","No such actfn_t: " + actfn2str(actfn_t));

	}
	
	pops.back()->poph = this;

    }
}


void PPopH::sethnoffs(int hoffs,int noffs) {

    for (int h=0; h<H; h++) {

    	pops.back()->sethoffs(hoffs + h);

	pops.back()->setnoffs(noffs + h*Nh);

	printf("%d %d\n",pops.back()->hoffs,pops.back()->noffs);

    }
}


void PPopH::setmaxdelay(float maxdelay) {

    for (size_t p=0; p<pops.size(); p++) {

	pops[p]->setmaxidelay(maxdelay);

	pops[p]->setupaxons();
	
    }
}


void PPopH::gatheract(vector<float> &ppact) {

    ppact = getstate("act");

}


void PPopH::bwsuptocurr(float Esyn) {

    if (Esyn>-1) {

    	for (size_t p=0,u=0; p<pops.size(); p++)

     	    for (size_t n=0; n<Nh; n++,u++)

     		pops[p]->bwsup[n] *= (Esyn - pops[p]->dsup[n]);

    }
}


void PPopH::fwritestate(PIO *pio,string statestr) {

    for (size_t p=0; p<pops.size(); p++)

	pio->pfwritestatevec(pops[p]->noffs,pops[p]->getstate(statestr));

}
