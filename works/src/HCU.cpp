#include <stdlib.h>
#include <vector>
#include <string>
#include <random>
#include <limits>

#include "Globals.h"
#include "Logger.h"
#include "Parseparam.h"
#include "HCU.h"

using namespace std;
using namespace Globals;


std::vector<HCU *> HCU::hcus;


HCU::HCU(int M,Actfn_t actfn_t,Normfn_t normfn_t) : Pop(M,actfn_t) {

    if (not (actfn_t==BCP or actfn_t==SPKBCP))
	error("HCU::HCU","Illegal 'actfn_t'");

    this->normfn_t = normfn_t;

    initparams();

    allocatestate();

    id = hcus.size();

    hcus.push_back(this);

}


HCU *HCU::gethcu(int id) {

    for (size_t h=0; h<hcus.size(); h++) if (hcus[h]->id==id) return hcus[h];

    return NULL;

}


void HCU::allocatestate() {

    Pop::allocatestate();
    
    expdsup = vector<float>(N,0);

    pact = vector<float>(N,0);

    dsupmax = 0;

    expdsupsum = 0;

}


int HCU::getinfo(string infostr) { return -1; }


float HCU::getstate1(string statestr) {

    if (statestr=="dsupmax") 
	return dsupmax;
    else if (statestr=="expdsupsum") 
	return expdsupsum;
    else error("HCU::getstate1","Illegal statestr: " + statestr);

    return dsupmax;

}


vector<float> const& HCU::getstate(string statestr) {

    if (statestr=="expdsup") 

	return expdsup;

    else if (statestr=="pact") 

	return pact;

    else return Pop::getstate(statestr);

    error("HCU::getstate","Illegal statestr: " + statestr);

    return act;

}


void HCU::prnstate(string statestr) {

    if (statestr=="dsupmax" or statestr=="expdsupsum") {

	prnval(getstate1(statestr));

	return;

    }

    prnvec(getstate(statestr));

}


void HCU::fwritestate(string statestr,FILE *outf) {

    if (statestr=="expdsup") {

	fwriteval(getstate(statestr),outf);

	return;

    } else if (statestr=="dsupmax" or statestr=="expdsupsum") {

	fwriteval(getstate1(statestr),outf);

	return;

    }
    
    fwriteval(getstate(statestr),outf);

}


Logger *HCU::logstate(std::string statestr,std::string filename,int logstep) {

    return new Logger(this,statestr,filename,logstep);

}


void HCU::setinp(float inpval,int n) {

    if (inpval<0) error("HCU::setinp","Illegal inpval<0: " + to_string(inpval));

    if (n<0) 

	fill(inp.begin(),inp.end(),log(inpval + epsfloat));

    else if (0<=n and n<M)

	inp[n] = log(inpval + epsfloat);

    else

	error("Pop::setinp","Illegal n not in [0,M[: " + to_string(n));	
    

}


void HCU::setinp(vector<float> inpvec) {

    if (inpvec.size()<noffs+N) error("HCU::setinp","inpvec length mismatch: " + to_string(inpvec.size()));

    for (int n=0; n<N; n++) {

    	if (inpvec[noffs+n]<0) error("HCU::setinp","Illegal inpvec[i]<0: " + to_string(inpvec[noffs+n]));

    	inp[n] = log(inpvec[noffs + n] + epsfloat);

    }
}


void HCU::resetstate() {

    Pop::resetstate();

    fill(expdsup.begin(),expdsup.end(),0);

    fill(pact.begin(),pact.end(),0);

    fill(dsup.begin(),dsup.end(),0);

    fill(bwsup.begin(),bwsup.end(),0);

    dsupmax = 0;
    
    expdsupsum = 0;

}


void HCU::upddsupmax() {

    dsupmax = again * dsup[0];

    for (int n=1; n<N; n++)
	
	if (again * dsup[n]>dsupmax) dsupmax = again * dsup[n];

}


void HCU::updexpdsup() {

    for (int n=0; n<N; n++) expdsup[n] = exp(again * dsup[n] - dsupmax);

}


void HCU::updexpdsupsum() {

    expdsupsum = expdsup[0];

    for (int n=1; n<N; n++) expdsupsum += expdsup[n];

}


void HCU::updBCP() {

    fill(act.begin(),act.end(),0);

    for (int n=0; n<N; n++) {

	if (pact[n]>thres) act[n] = pact[n];

	/// if (simstep>200) printf("%4d %2d %.2f %.2f %.2f %.2f %.2f %.2f\n",
	// 			simstep,n,bwsup[n],bwgain,dsup[n],expdsup[n],pact[n],act[n]);
    }

}

///// OLD ////////

// void HCU::normalize() {

//     fill(pact.begin(),pact.end(),0);

//     if (normfn_t==FULL) {

// 	if (expdsupsum>0) {

// 	    for (int n=0; n<N; n++) {

// 		expdsup[n] /= expdsupsum;

// 		pact[n] = expdsup[n];

// 	    }
// 	}

//     } else if (normfn_t==HALF) {

// 	if (dsupmax>0 or expdsupsum>1)

// 	    for (int n=0; n<N; n++) expdsup[n] /= expdsupsum;

// 	else {

// 	    float kdsupmax = exp(dsupmax);

// 	    for (int n=0; n<N; n++) expdsup[n] *= kdsupmax;

// 	}

// 	for (int n=0; n<N; n++) pact[n] = expdsup[n];

//     } else if (normfn_t==CAP) {

// 	for (int n=0; n<N; n++) {

// 	    if (dsupmax>1) expdsup[n] *= exp(dsupmax);
	
// 	    pact[n] = expdsup[n];

// 	    if (pact[n]>1) pact[n] = 1;

// 	}
	
//     } else error("HCU::normalize","Illegal normfn_t");

// }

void HCU::normalize() {

    fill(pact.begin(),pact.end(),0);

    if (normfn_t==FULL) {

    if (expdsupsum>0) {

        for (int n=0; n<N; n++) {

        expdsup[n] /= expdsupsum;

        pact[n] = expdsup[n];

        }
    }

/*

  halfnorm(logX) is mathematically normalization only when when
  sum(exp(logX)) is >1, otherwise the exp(logX) should be used. The
  only complication in the code is that it sometimes happens that
  exp(logX) overflows, i.e. gives 'inf'. This needs to be avoided,
  hence the use of 'dsupmax' in the code below.

  Note that exp(logX - dsupmax) * exp(dsupmax) gives the original
  exp(logX). But as stated above, this needs that exp(dsupmax) does
  not overflow, which is secured in the below code.

  We use this to compute halfnorm(dsup), given that we have already
  computed dsupmax, expdsup = exp(dsup - dsupmax), and expdsupsum =
  sum(expdsup).

*/

    } else if (normfn_t==HALF) {

    if (dsupmax>0)

        for (int n=0; n<N; n++) expdsup[n] /= expdsupsum;

    else {

        float expdsupmax = exp(dsupmax);

        if (expdsupsum * expdsupmax > 1)

            for (int n=0; n<N; n++) expdsup[n] /= expdsupsum;

        else

            for (int n=0; n<N; n++) expdsup[n] *= expdsupmax;

    }

    pact = expdsup;

/*

  cap(logX) is mathematically min(1,exp(logX)). Again exp(log(X))
  could overflow. But by setting logX to 0 and X = 1 if logX>0 this is
  handled.

*/

    } else if (normfn_t==CAP) {

    for (int n=0; n<N; n++) {

        if (dsup[n]>0) expdsup[n] = 1; else expdsup[n] = exp(dsup[n]);

        pact[n] = expdsup[n];

    }
    
    } else error("HCU::normalize","Illegal normfn_t");
}

void HCU::update() {

    updada();

    upddsup();

    upddsupmax();

    updexpdsup();

    updexpdsupsum();

    normalize();

    switch (actfn_t) {

    case SPKBCP:
    case BCP: updBCP();	break;

    default: error("HCU::update","Illegal actfn_t");
    }

    updaxons();

}
