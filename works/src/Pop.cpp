/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#include <stdlib.h>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <limits>

#include "Globals.h"
#include "Parseparam.h"
#include "PopH.h"
#include "Logger.h"
#include "Prj.h"
#include "Pop.h"

using namespace std;
using namespace Globals;


#define DSUPMAXVAL 16


std::vector<Pop *> Pop::pops;


Pop::Pop(int M,Actfn_t actfn_t,Normfn_t normfn_t) {

    if (setupdone) error("Pop::Pop","Not allowed when setupdone");

    if (actfn_t!=LIN and actfn_t!=LOG and actfn_t!=BCP and actfn_t!=SIG and
	actfn_t!=SPKLIN and actfn_t!=SPKLOG and actfn_t!=SPKSIG and actfn_t!=SPKBCP and actfn_t!=AREG)
	error("Pop::Pop","Illegal actfn_t: " + to_string(actfn_t));

    if (M==0) error("Pop::Pop","Illegal M==0");

    id = pops.size();

    poph = NULL;

    this->M = M;

    U = 1;

    N = M * U;

    this->actfn_t = actfn_t;

    this->normfn_t = normfn_t;

    init();

    pops.push_back(this);

}


Pop::Pop(int M,int U,Actfn_t actfn_t,Normfn_t normfn_t) {

    if (setupdone) error("Pop::Pop","Not allowed when setupdone");

    if (actfn_t!=LIN and actfn_t!=LOG and actfn_t!=BCP and actfn_t!=SIG and
	actfn_t!=SPKLIN and actfn_t!=SPKLOG and actfn_t!=SPKSIG and actfn_t!=SPKBCP and actfn_t!=AREG)
	error("Pop::Pop","Illegal actfn_t: " + to_string(actfn_t));

    if (M==0) error("Pop::Pop","Illegal M==0");
    if (U==0) error("Pop::Pop","Illegal U==0");

    id = pops.size();

    poph = NULL;

    this->M = M;

    this->U = U;

    N = M * U;

    this->actfn_t = actfn_t;

    this->normfn_t = normfn_t;

    init();

    pops.push_back(this);

}


Pop::Pop(int M,int U,bool noinit) {

    if (setupdone) error("Pop::Pop","Not allowed when setupdone");

    if (M==0) error("Pop::Pop","Illegal M==0");
    if (U==0) error("Pop::Pop","Illegal U==0");

    id = pops.size();

    poph = NULL;

    this->M = M;

    this->U = U;

    N = M * U;

    this->actfn_t = actfn_t;

    pops.push_back(this);

}


void Pop::init() {

    noffs = 0;

    hoffs = 0;

    setseed(id + 1);

    axons = NULL;

    initparams();

    allocatestate();

    poissondistr = poisson_distribution<int> (1.0);

    uniformdistr = uniform_real_distribution<float> (0.0,1.0);

}

Pop *Pop::getpop(int id) {

    for (size_t h=0; h<pops.size(); h++) if (pops[h]->id==id) return pops[h];

    return NULL;

}


int Pop::getnpop() { return pops.size(); }


void Pop::initparams() {

    setparam("igain",1);
    setparam("again",1);
    setparam("bwgain",1);
    setparam("taum",1.1 * timestep);  // taum == timestep may not work well.

    setparam("adgain",0);
    setparam("taua",1);
    setparam("namp",0);
    setparam("nmean",1);
    setparam("thres",0);

    Esyn = -1;
    spkwgain = 1;
    maxfq = 0;

    switch (actfn_t) {

    case SPKBCP:

    case SPKLIN:

    case SPKLOG:

    case SPKSIG:

    case SPKEXP:

    case AREG:
	setparam("maxfq",100);
	break;

    }

    switch (actfn_t) {

    case LIN:

    case SPKLIN:

    case AREG:
	setparam("thres",0);

	break;

    case SPKBCP:

    case SPKEXP:

    case BCP:

    case EXP:
	setparam("thres",log(epsfloat));
	break;

    case SIG:

    case LOG:

    case SPKSIG:

    case SPKLOG:

	setparam("thres",1);
	break;

    }
}


void Pop::allocatestate() {

    bwsup = vector<float>(N,0);
    dsup = vector<float>(N,0);
    act = vector<float>(N,0);
    H_en = vector<float>(N,0);
    pact = vector<float>(N,0);
    ada = vector<float>(N,0);
    xyz = vector<float> (3,0);

    maxidelay = 0;

    inp = vector<float>(N,0);

}


int Pop::getinfo(string infostr) {

    if (infostr=="M") return M;
    else if (infostr=="N") return N;
    else if (infostr=="U") return U;
    else if (infostr=="Actfn") return actfn_t;
    else if (infostr=="id") return id;
    else if (infostr=="hoffs") return hoffs;
    else if (infostr=="noffs") return noffs;
    else error("Pop::getinfo","No such 'infostr'");

    return 0;

}


void Pop::setxyz(vector<float> xyz) { this->xyz = xyz; }


vector<float> const &Pop::getxyz() { return xyz; }


void Pop::setupaxons() {

    now = 0;

    axons = new float[N*maxidelay];

}


void Pop::setupaxonsall() {

    for (size_t h=0; h<pops.size(); h++) pops[h]->setupaxons();

}


void Pop::setnmean(float nmean) {

    this->nmean = nmean * timestep;

    generator.seed(seed);

    poissondistr = poisson_distribution<int> (this->nmean);

}


bool Pop::checkparam(string paramstr) {

    if (paramstr=="igain" or paramstr=="again" or paramstr=="bwgain" or paramstr=="taum" or
	paramstr=="taumdt" or paramstr=="adgain" or paramstr=="taua" or paramstr=="tauadt" or paramstr=="namp" or
	paramstr=="nmean" or paramstr=="thres" or paramstr=="spkwgain") return true;

    switch (actfn_t) {

    case SPKBCP:
    case SPKLIN:
    case SPKSIG:
    case SPKLOG:
    case SPKEXP:
    case AREG:

	if (paramstr=="maxfq") return true;
	break;

    }

    return false;

}


void Pop::fixupspkwgain() {

    for (size_t p=0; p<inprjs.size(); p++) inprjs[p]->trgspkwgain = spkwgain;

    for (size_t p=0; p<utprjs.size(); p++) utprjs[p]->srcspkwgain = spkwgain;

}


void Pop::setparam(string paramstr,float parval) {

    checkparam(paramstr);

    if (paramstr=="taum") {
	// if (parval<=timestep) warning("Pop::setparam","taum == timestep may not work well");
	taumdt = calctaudt(parval);
    } else if (paramstr=="igain")
	igain = parval;
    else if (paramstr=="again")
	again = parval;
    else if (paramstr=="bwgain")
	bwgain = parval;
    else if (paramstr=="namp")
	namp = parval;
    else if (paramstr=="nmean")
	setnmean(parval);
    else if (paramstr=="adgain")
	adgain = parval;
    else if (paramstr=="Esyn")
	Esyn = parval;
    else if (paramstr=="taua") {
	tauadt = calctaudt(parval);
    } else if (paramstr=="maxfq") {
	maxfq = parval;
	spkwgain = 1/(maxfq*timestep);
	fixupspkwgain();
    } else if (paramstr=="thres")
	thres = parval;
    else if (paramstr=="spkper") {
	spkiper = int(parval/timestep + 0.5);
	setparam("maxfq",1/parval);
    } else if (paramstr=="spkpha")
	spkipha = int(parval/timestep + 0.5);

    else error("Pop::setparam","Illegal paramstr:" + paramstr);

}


void Pop::setparam(Popparam_t param_t,float parval) {

    switch (param_t) {

    case TAUM:
	taumdt = calctaudt(parval);
	break;
    case IGAIN: setparam("igain",parval); break;
    case AGAIN: setparam("again",parval); break;
    case BWGAIN: setparam("bwgain",parval); break;
    case NAMP: setparam("namp",parval); break;
    case NMEAN: setparam("nmean",parval); break;
    case ADGAIN: setparam("adgain",parval); break;
    case TAUA: setparam("taua",parval); break;
    case MAXFQ: setparam("maxfq",parval); break;
    case THRES: setparam("thres",parval); break;
    case SPKPER: setparam("spkper",parval); break;
    case SPKPHA: setparam("spkipha",parval); break;

    default: error("Pop::setparam","No such param_t: " + popparam2str(param_t));

    }

}


void Pop::setparamsfromfile(string paramfilename) {}


float Pop::getparam(string paramstr) {

    if (paramstr=="igain")
	return igain;
    else if (paramstr=="again")
	return again;
    else if (paramstr=="taumdt")
    	return taumdt;
    else if (paramstr=="taum")
    	return timestep/taumdt;
    else if (paramstr=="bwgain")
    	return bwgain;
    else if (paramstr=="namp")
	return namp;
    else if (paramstr=="nmean")
    	return nmean;
    else if (paramstr=="adgain")
    	return adgain;
    else if (paramstr=="tauadt")
    	return tauadt;
    else if (paramstr=="taua")
    	return timestep/tauadt;
    else if (paramstr=="maxfq")
    	return maxfq;
    else if (paramstr=="thres")
    	return thres;
    else if (paramstr=="spkwgain")
    	return spkwgain;
    else error("Pop::getparam","Illegal paramstr: " + paramstr);

    return again;

}


void Pop::sethoffs(int hoffs) {

    this->hoffs = hoffs;

}


void Pop::setnoffs(int noffs) {

    this->noffs = noffs;

}


void Pop::sethnoffs(int hoffs) {

    sethoffs(hoffs);

    setnoffs(hoffs * N);

}


vector<float> const& Pop::getstate(string statestr) {

    if (statestr=="inp")
	return inp;
    else if (statestr=="bwsup")
	return bwsup;
    else if (statestr=="dsup")
	return dsup;
    else if (statestr=="pact")
	return act;
    else if (statestr=="act")
	return act;
    else if (statestr=="ada")
	return ada;
    else if (statestr=="H_en")
    return H_en;
    else error("Pop::getstate","Illegal statestr: " + statestr);

    return act;

}


void Pop::cpystatetovec(string statestr,vector<float> &vec) {

    if (noffs<0) error("Pop::cpystatetovec","Illegal noffs<0");

    vector<float> stavec = getstate(statestr);

    for (int m=0; m<N; m++) vec[noffs + m] = stavec[m];

}


void Pop::cpyvectostate(string statestr,vector<float> &vec) {

    if (noffs<0) error("Pop::cpyvectostate","Illegal noffs<0");

    vector<float> stavec = getstate(statestr);

    for (int m=0; m<N; m++) stavec[m] = vec[noffs + m];

}


long Pop::setseed(long seed) {

    if (seed==0) seed = random_device{}();

    if (seed<0) seed = -seed;

    seed = 10000000 + (seed + 11*id)%10000000;

    this->seed = seed;

    generator.seed(seed);

    return seed;

}


long Pop::getseed() { return this->seed; }


float Pop::nextfloat() { return uniformdistr(generator); }


int Pop::nextpoisson() { return poissondistr(generator); }


void Pop::prnparam(string paramstr) {

    prnval(getparam(paramstr));

}


void Pop::prnstate(string statestr) {

    prnvec(getstate(statestr));

}


void Pop::fwritestate(string statestr,FILE *outf) {

        fwriteval(getstate(statestr),outf);

}


void Pop::fwritestate(string statestr,string filename) {

    FILE *outf = fopen(filename.c_str(),"w");

    if (outf==NULL) error("Pop::fwritestate","Could not open file" + filename);

    fwritestate(statestr,outf);

    fclose(outf);

}


Logger *Pop::logstate(std::string statestr,std::string filename,int logstep) {

    return new Logger(this,statestr,filename,logstep);

}


void Pop::setinp(float inpval,int n) {

    if (n<0)

	fill(inp.begin(),inp.end(),inpval);

    else if (0<=n-noffs and n-noffs<M)

	inp[n] = igain * inpval;

    else

	error("Pop::setinp","Illegal n not in [0,M[: " + to_string(n));

}


void Pop::setinp(vector<float> inpvec) {

    bool stimM = (M<=inpvec.size()),stimN = (N<=inpvec.size()); /// This is shaky, why not remove?

    if (not (stimM or stimN)) error("Pop::setinp","inpvec length mismatch: " + to_string(inpvec.size()));

    if (stimM) {

	for (int m=0; m<M; m++) {

	    for (int u=0,n; u<U; u++) {

		n = m * U + u;

		inp[n] = igain * inpvec[hoffs*M + m];

	    }
	}

    } else if (stimN)

	for (int n=0; n<N; n++) inp[n] = igain * inpvec[noffs + n];

}


void Pop::resetstate() {

    fill(bwsup.begin(),bwsup.end(),0);

    fill(act.begin(),act.end(),0);
    fill(ada.begin(),ada.end(),0);

    if (actfn_t==BCP or actfn_t==LIN or actfn_t==LOG or actfn_t==EXP or
	actfn_t==SPKBCP or actfn_t==SPKLIN or actfn_t==SPKLOG or actfn_t==SPKEXP or actfn_t==AREG) {
	fill(inp.begin(),inp.end(),0);
	fill(dsup.begin(),dsup.end(),0);
    }

    if (axons!=NULL) memset(axons,0,N*maxidelay*sizeof(float));

}


void Pop::resetstateall() {

    for (size_t h=0; h<pops.size(); h++) pops[h]->resetstate();

}


void Pop::setmaxidelay(float maxdelay) {

    if (maxdelay<=timestep) error("Pop::setmaxdelay","Illegal maxdelay<=0");

    maxidelay = maxdelay/timestep;

}


void Pop::updaxons() {

    if (maxidelay<=0 or axons==NULL) return;

    for (int m=0; m<N; m++) axons[m * maxidelay + now] = act[m];

    now = (now + 1) % maxidelay;

}


int Pop::getdslot(int idelay) {

    return (maxidelay + now - idelay)%maxidelay;

}


float Pop::getdelact(int m,int idelay) {

    // std::cout<<"\n idelay: "<<idelay<<" "<<" maxidelay: "<<maxidelay;
    return axons[m * maxidelay + getdslot(idelay)];

}


void Pop::updada() {

    for (int n=0; n<N; n++) {

	ada[n] += (adgain * spkwgain * act[n] - ada[n]) * tauadt;

    }

}


void Pop::upddsup() {

    if (actfn_t==ALIF or actfn_t==AdEx or actfn_t==AdExS) return;

    for (int n=0; n<N; n++) 

    	dsup[n] += (inp[n] + bwgain * bwsup[n] - ada[n] - dsup[n]) * taumdt;

    if (namp>0) for (int n=0; n<N; n++) dsup[n] += namp * (nextpoisson() - nmean); 
    // std::cout<<"namp: "<<namp<<" nextpoisson(): "<<nextpoisson()<<" nmean: "<<nmean<<" del_dsup: "<<namp * (nextpoisson() - nmean)<<std::endl;} // nbrav


}


void Pop::normalize() {

    fill(pact.begin(),pact.end(),1./N);

    if (normfn_t==NONORMFN) {

	for (int n=0; n<N; n++) pact[n] = dsup[n];

    } else if (normfn_t==FULL) {

	float dsupsum = 0; for (int n=0; n<N; n++) dsupsum += dsup[n];

	if (dsupsum>0) for (int n=0; n<N; n++) pact[n] = dsup[n]/dsupsum;

    } else if (normfn_t==HALF) {

	float dsupsum = 0; for (int n=0; n<N; n++) dsupsum += dsup[n];

	if (dsupsum>1) for (int n=0; n<N; n++) pact[n] = dsup[n]/dsupsum;

    } else if (normfn_t==CAP) {

	for (int n=0; n<N; n++) if (dsup[n]<1) pact[n] = dsup[n]; else pact[n] = 1;

    } else error("Pop::normalize","Illegal normfn_t");

}

void Pop::compute_energy() {

    for (int n=0; n<N; n++) {
        H_en[n] = dsup[n] * act[n];
    }

}

void Pop::updLIN() {

    fill(act.begin(),act.end(),0);

    if (thres<0) error("Pop::updLIN","Illegal thres<0");

    for (int n=0; n<N; n++) if (pact[n]>thres) act[n] = again * pact[n];

    normalize();

}


void Pop::updLOG() {

    fill(act.begin(),act.end(),0);

    if (thres<1) error("Pop::updLOG","Illegal thres<1");

    for (int n=0; n<N; n++) if (pact[n]>thres) act[n] = log(again*pact[n]);

    normalize();

}


void Pop::updSIG() {

    fill(act.begin(),act.end(),0);

    for (int n=0; n<N; n++) act[n] = 1/(1 + exp(-again * (pact[n] - thres)));

    normalize();

}




void Pop::updspk() {

    int effspkiper;

    switch (actfn_t) {

    case SPKLIN:
    case SPKLOG:
    case SPKSIG:
    case SPKEXP:
	for (int n=0; n<N; n++)	act[n] = nextfloat()<(act[n] * maxfq * timestep);
	break;
    case AREG:
	for (int n=0; n<N; n++) {
	    if (dsup[n]>thres) effspkiper = spkiper/pact[n]; else effspkiper = 1e6;

	    if (effspkiper<=0)
		act[n] = 0;
	    else
		act[n] = ((2*spkiper + simstep - spkipha)%effspkiper==0);

	}
	break;

    default: error("Pop::updspk","Illegal Actfn_t: " + to_string(actfn_t));

    }
}


void Pop::update() {

    updada();

    upddsup();

    switch (actfn_t) {

    case LIN: updLIN(); break;
    case LOG: updLOG(); break;
    case SIG: updSIG(); break;
    case SPKLIN: updLIN(); updspk(); break;
    case SPKLOG: updLOG(); updspk(); break;
    case SPKSIG: updSIG(); updspk(); break;
    case AREG: normalize(); updspk(); break;

    default: error("Pop::update","Illegal Actfn_t: " + to_string(actfn_t));

    }

    updaxons();

}


void Pop::updateall() {

    for (size_t h=0; h<pops.size(); h++) pops[h]->update();

}


void Pop::resetbwsup() {

    fill(bwsup.begin(),bwsup.end(),0);

}


void Pop::resetbwsupall() {

    for (size_t h=0; h<pops.size(); h++) pops[h]->resetbwsup();

}


void Pop::contribute(vector<float> &cond) {

    if (Esyn<=-1)

	for (int n=0; n<N; n++) bwsup[n] += cond[n];

    else

       for (int n=0; n<N; n++) bwsup[n] += cond[n] * (Esyn - dsup[n]);

}


ExpPop::ExpPop(int M,int U,Actfn_t actfn_t,Normfn_t normfn_t) : Pop(M,U,LIN,normfn_t) {

    // Need to be special due to high overflow tendency

    if (actfn_t!=EXP and actfn_t!=SPKEXP) error("ExpPop::ExpPop","Illegal actfn_t");

    this->actfn_t = actfn_t;

    expallocatestate();

}


ExpPop::ExpPop(int M,Actfn_t actfn_t,Normfn_t normfn_t) : Pop(M,LIN,normfn_t) {

    // Need to be special due to high overflow tendency

    if (actfn_t!=EXP and actfn_t!=SPKEXP) error("ExpPop::ExpPop","Illegal actfn_t");

    this->actfn_t = actfn_t;

    expallocatestate();

}


void ExpPop::expallocatestate() {

    expdsup = vector<float>(N,0);

    pact = vector<float>(N,0);

    dsupmax = 0;

    expdsupsum = 0;

}


float ExpPop::getstate1(string statestr) {

    if (statestr=="dsupmax")
	return dsupmax;
    else if (statestr=="expdsupsum")
	return expdsupsum;
    else error("ExpPop::getstate1","Illegal statestr: " + statestr);

    return dsupmax;

}


vector<float> const& ExpPop::getstate(string statestr) {

    if (statestr=="expdsup")

	return expdsup;

    else if (statestr=="pact")

	return pact;

    else return Pop::getstate(statestr);

    error("ExpPop::getstate","Illegal statestr: " + statestr);

    return act;

}


void ExpPop::prnstate(string statestr) {

    if (statestr=="dsupmax" or statestr=="expdsupsum") {

	prnval(getstate1(statestr));

	return;

    }

    prnvec(getstate(statestr));

}


void ExpPop::fwritestate(string statestr,FILE *outf) {

    if (statestr=="expdsup") {

	fwriteval(getstate(statestr),outf);

	return;

    } else if (statestr=="dsupmax" or statestr=="expdsupsum") {

	fwriteval(getstate1(statestr),outf);

	return;

    }

    fwriteval(getstate(statestr),outf);

}


void ExpPop::resetstate() {

    Pop::resetstate();

    fill(expdsup.begin(),expdsup.end(),1./M);

    fill(pact.begin(),pact.end(),0);

    fill(dsup.begin(),dsup.end(),log(1./M));

    fill(bwsup.begin(),bwsup.end(),log(1./M));

    dsupmax = 0;

    expdsupsum = 0;

}


void ExpPop::upddsupmax() {

    dsupmax = again * dsup[0]; // 0

    for (int n=1; n<N; n++)

	if (again * dsup[n]>dsupmax) dsupmax = again * dsup[n];

}


void ExpPop::updexpdsup() {

    for (int n=0; n<N; n++) expdsup[n] = exp(again * dsup[n] - dsupmax); 

}


void ExpPop::updexpdsupsum() {

    expdsupsum = expdsup[0];

    for (int n=1; n<N; n++) expdsupsum += expdsup[n];

}


 
void ExpPop::normalize() {

    fill(pact.begin(),pact.end(),1./N);

    switch (normfn_t) {

    case NONORMFN:

    if (dsupmax>DSUPMAXVAL) error("ExpPop::normalize","Too high dsup values");

    for (int n=0; n<N; n++) pact[n] = expdsup[n] * exp(dsupmax);

    break;

    case FULL:

    if (expdsupsum>0) {

        for (int n=0; n<N; n++) {

        expdsup[n] /= expdsupsum;

        pact[n] = expdsup[n];

        }
    }
    break;

    case HALF:

    // // **** Older Version ****

    // if (dsupmax>0 or expdsupsum>1)

    //     for (int n=0; n<N; n++) expdsup[n] /= expdsupsum;

    // else {

    //     float kdsupmax = exp(dsupmax);

    //     for (int n=0; n<N; n++) expdsup[n] *= kdsupmax;

    // }

    // for (int n=0; n<N; n++) pact[n] = expdsup[n];


    // // **** Updated 12/11/2021 **** // //

    if (dsupmax>0)
        for (int n=0; n<N; n++) expdsup[n] /= expdsupsum;

    else {
        float kdsupmax = exp(dsupmax);
        if (kdsupmax*expdsupsum>1)
            for (int n=0; n<N; n++) expdsup[n] /= expdsupsum;
        else
            for (int n=0; n<N; n++) expdsup[n] *= kdsupmax;     

    }
    for (int n=0; n<N; n++) pact[n] = expdsup[n];

    break;


    case CAP:

    for (int n=0; n<N; n++) {

        if (dsup[n]>0) expdsup[n] = 1; else expdsup[n] = exp(dsup[n]); //dsup[n] = 0;
    
        pact[n] = expdsup[n];

    }

    break;

    default: error("ExpPop::normalize","Illegal normfn_t");

    }

}


void ExpPop::updEXP() {

    fill(act.begin(),act.end(),0);

    for (int n=0; n<N; n++) {
        if (pact[n]>thres) act[n] = pact[n];
    }

}


void ExpPop::update() {

    updada();

    upddsup();

    upddsupmax();

    updexpdsup();

    updexpdsupsum();

    normalize();

    updEXP();

    compute_energy();
    switch (actfn_t) {

    case EXP: break;
    case SPKEXP: updspk(); break;

    default: error("ExpPop::update","Illegal actfn_t");
    }

    updaxons();

}



BCPPop::BCPPop(int M,int U,Actfn_t actfn_t,Normfn_t normfn_t) : ExpPop(M,U,EXP,normfn_t) {

    if (actfn_t==SPKBCP) this->actfn_t = SPKEXP;

}


BCPPop::BCPPop(int M,Actfn_t actfn_t,Normfn_t normfn_t) : ExpPop(M,EXP,normfn_t) {

    if (actfn_t==SPKBCP) this->actfn_t = SPKEXP;

}


void BCPPop::setinp(float inpval,int n) {

    if (inpval<0) error("BCPPop::setinp","Illegal inpval<0: " + to_string(inpval));

    if (n<0)

	fill(inp.begin(),inp.end(),igain *log(inpval + epsfloat)); //  fill(inp.begin(),inp.end(),log(igain * inpval + epsfloat));

    else if (0<=n and n<M)

	inp[n] = igain * log(inpval + epsfloat); //log((igain * inpval) + epsfloat); 

    else

	error("Pop::setinp","Illegal n not in [0,M[: " + to_string(n));


}


void BCPPop::setinp(vector<float> inpvec) {


    if (inpvec.size()<noffs+N)
	error("BCPPop::setinp","inpvec length mismatch: " +
	      to_string(inpvec.size()) + " " + to_string(noffs) + " " + to_string(N));

    for (int n=0; n<N; n++) {

    	if (inpvec[noffs+n]<0) error("BCPPop::setinp","Illegal inpvec[i]<0: " + to_string(inpvec[noffs+n]));
 
        inp[n] = igain * log(inpvec[noffs + n] + epsfloat); //inp[n] = log((igain * inpvec[noffs + n]) + epsfloat);

    }
}


void BCPPop::fwritestate(string statestr,FILE *outf) {

    ExpPop::fwritestate(statestr,outf);

}


SNNPop::SNNPop(int M,Actfn_t actfn_t) : Pop(M,1,true) {

    if (setupdone) error("Pop::Pop","Not allowed when setupdone");

    this->actfn_t = actfn_t;

    init();

}


SNNPop::SNNPop(int M,int U,Actfn_t actfn_t) : Pop(M,U,true) {

    if (setupdone) error("Pop::Pop","Not allowed when setupdone");

    this->actfn_t = actfn_t;

    init();

}


void SNNPop::setparamsfromfile(string paramfilename) {

    Parseparam *parseparam = new Parseparam(paramfilename);

    parseparam->postparam("C",&C,Float);
    parseparam->postparam("gL",&gL,Float);
    parseparam->postparam("EL",&EL,Float);
    parseparam->postparam("DT",&DT,Float);
    parseparam->postparam("VR",&VR,Float);
    parseparam->postparam("VT",&VT,Float);
    parseparam->postparam("spkreft",&spkreft,Float);
    parseparam->postparam("taua",&taua,Float);
    parseparam->postparam("adgain",&adgain,Float);

    parseparam->doparse();

    setparam("C",C);
    setparam("gL",gL);
    setparam("EL",EL);
    setparam("DT",DT);
    setparam("VR",VR);
    setparam("VT",VT);
    setparam("spkreft",spkreft);
    setparam("taua",taua);
    setparam("adgain",adgain);

}


void SNNPop::initparams() {

    Pop::initparams();

    setparam("C",1.5e-11);
    setparam("gL",2e-9);
    setparam("EL",-76e-3);
    if (not actfn_t==ALIF) setparam("DT",1e-3);
    setparam("VR",-60e-3);
    setparam("VT",-44e-3);
    setparam("spkreft",1e-3);
    setparam("taua",0.100);
    setparam("adgain",0);
    setparam("nmean",2000);
    setparam("namp",2e-3);

}


bool SNNPop::checkparam(string paramstr) {

    if (paramstr=="C" or paramstr=="gL" or paramstr=="EL" or paramstr=="VR" or paramstr=="VT" or
	paramstr=="spkreft" or paramstr=="spkireft")
	return true;

    switch (actfn_t) {

    case AdExS:

    case AdEx:
	if (paramstr=="DT") return true;
	break;

    case ALIF:
	break;

    default: error("SNNPop::checkparam","No such 'actfn_t'");

    }

    return false;

}


void SNNPop::setparam(string paramstr,float parval) {

    if (checkparam(paramstr)) {

	if (paramstr=="spkireft") {
	    if (parval<1) error("Pop::setparam","Illegal spkireft<1: " + to_string(parval));
	    spkireft = parval;
	} else if (paramstr=="spkreft") {
	    if (parval<timestep) error("Pop::setparam","Illegal spkreft<timestep: " + to_string(parval));
	    setparam("spkireft",parval/timestep);
	} else if (paramstr=="spkwgain") {
	    spkwgain = parval;
	    fixupspkwgain();
	} else if (paramstr=="C")
	    C = parval;
	else if (paramstr=="gL")
	    gL = parval;
	else if (paramstr=="EL")
	    EL = parval;
	else if (paramstr=="DT")
	    DT = parval;
	else if (paramstr=="VR")
	    VR = parval;
	else if (paramstr=="VT")
	    VT = parval;

    } else

	Pop::setparam(paramstr,parval);

}


float SNNPop::getparam(string paramstr) {

    if (paramstr=="spkireft")
	return spkireft;
    if (paramstr=="spkreft")
	return spkireft * timestep;
    else if (paramstr=="spkwgain")
    	return spkwgain;
    else if (paramstr=="taum")
    	return C/gL;
    else if (paramstr=="C")
    	return C;
    else if (paramstr=="gL")
	return gL;
    else if (paramstr=="EL")
    	return EL;
    else if (paramstr=="DT") {
	if (actfn_t==ALIF) warning("SNNPop::getparam","'DT' is not a paramter of an ALIF neuron");
    	return DT;
    } else if (paramstr=="VR")
    	return VR;
    else if (paramstr=="VT")
    	return VT;

    else return Pop::getparam(paramstr);

}


void SNNPop::allocatestate() {

    Pop::allocatestate();

    spkstep = vector<int>(N,0);
    fill(dsup.begin(),dsup.end(),EL);

}


void SNNPop::resetstate() {

    Pop::resetstate();

    fill(dsup.begin(),dsup.end(),EL);

}


void SNNPop::updada() {

    for (int n=0; n<N; n++) {

	if (actfn_t==ALIF or actfn_t==AdExS)

	    ada[n] += (adgain * act[n] - ada[n]) * tauadt;

	else

	    ada[n] += (adgain * (dsup[n] - EL) - ada[n]) * tauadt;

    }
}


void SNNPop::updspk() {

    float tmp;

    for (int n=0; n<N; n++) {

	if (spkstep[n]>0) {

	    spkstep[n]--;

	    if (spkstep[n]==0) {

		dsup[n] = VR;

		act[n] = 0;

		spkstep[n] = -spkireft;

	    }

	} else if (spkstep[n]<0)

	    spkstep[n]++;

	else if (VT<=dsup[n]) {

	    dsup[n] = 0.010;

	    act[n] = 1;

	    spkstep[n] = 2;

	} else {

	    if (namp>0) dsup[n] += namp * (nextpoisson() - nmean);

	    switch (actfn_t) {

	    case ALIF:
		dsup[n] += -(gL * (dsup[n] - EL) - bwgain * bwsup[n] + ada[n]) * timestep/C

		    + inp[n] * timestep/C ;

		break;

	    case AdEx:

	    case AdExS:

		tmp = gL* DT * exp((dsup[n] - VT)/DT);

		if (tmp>1e-10) tmp = C/timestep*0.1;

		dsup[n] += -(gL * (dsup[n] - EL)
			     - tmp
			     - bwgain * bwsup[n]
			     + ada[n]
			     ) * timestep/C

		    + inp[n] * timestep/C;

		if (VT<=dsup[n]) dsup[n] = VT;

		break;

	    default: error("SNNPop::updspk","No such 'actfn_t'");

	    }
	}
    }
}


void SNNPop::update() {

    updada();

    upddsup();

    switch (actfn_t) {

    case ALIF:
    case AdExS:
    case AdEx: updspk(); break;

    default: error("SNNPop::update","Illegal Actfn_t: " + to_string(actfn_t));
    }

    updaxons();

}
