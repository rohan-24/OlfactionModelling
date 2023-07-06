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

#include "Globals.h"
#include "Parseparam.h"
#include "Pop.h"
#include "PopH.h"
#include "PrjH.h"
#include "Logger.h"
#include "Prj.h"

using namespace std;
using namespace Globals;


Connij::Connij(Prj *prj,int srcn,int trgM,float psilent,float eps,BCPvar_t bcpvar) {

    this->prj = prj;

    this->srcn = srcn;

    this->trgM = trgM;

    this->eps = eps;

    this->tausddt = 0;

    this->tausfdt = 0;

    this->sp0 = 1;

    Wij = vector<float> (trgM,0);

    won = vector<float>(trgM,1);

    if (bcpvar!=FIXED)

	Pij = vector<float> (trgM,eps*eps);

    if (bcpvar==INCR_E) Eij = vector<float> (trgM,eps*eps);

    initstate();

    resetstate();

    setsilent(psilent);

}


void Connij::initstate() {

    for (int trgm=0; trgm<trgM; trgm++)	won[trgm] = prj->nextfloat()<prj->wdens;

    fill(Wij.begin(),Wij.end(),0);

    silent = false;

    Zi = Ei = Pi = eps;

    if (Pij.size()>0) fill(Pij.begin(),Pij.end(),eps*eps);

    if (Eij.size()>0) fill(Eij.begin(),Eij.end(),eps*eps);

    resetstate();

}


void Connij::resetstate() {

    sdep = 1;

    sfac = sp0;

    Zi = eps;

    if (Eij.size()>0) {

	Ei = eps;

	fill(Eij.begin(),Eij.end(),eps*eps);

    }
}


int Connij::getinfo(string infostr) {

    int infoval;

    if (infostr=="W_n") {
	infoval = 0;
	for (int m=0; m<trgM; m++) infoval += won[m];
	return infoval;
    } else if (infostr=="bstep")
	return bstep;
    else if (infostr=="age")
	return simstep - bstep;

    else error("Connij::getinfo","No such 'infostr'");

    return 0;

}


int Connij::getwon(int trgm) {

    if (trgm<0 or trgM<=trgm) error("Connij::getwon","'trgm' not in [0,trgm[");

    return won[trgm];

}


void Connij::updsdep(float srcact,float spkwgain) {

    /* After Mongillo et al. 2008, Synaptic working memory. Eqn 5-7 */

    /* "x" is here called "sdep" */

    if (tausddt<=0) return;

    sdep += (1 - sdep) * tausddt;

    sdep -= sfac * sdep * srcact * spkwgain * tausddt;

    // if (srcact>0.99)
    //   std::cout<<"sdep="<<sdep<<",sfac="<<sfac<<" ";

    if (sdep<0) sdep = 0;

}


void Connij::updsfac(float srcact,float spkwgain) {

    /* After Mongillo et al. 2008, Synaptic working memory. Eqn 5-7 */
    /* "u" is here called "sfac" */

    if (tausfdt<=0) return;

    sfac += (sp0 - sfac) * tausfdt; // sfac == sp0 if no facilitation

    sfac += sp0 * (1 - sfac) * srcact * spkwgain * tausfdt;

    if (sfac>1) sfac = 1; else if (sfac<0) sfac = 0;

}


void Connij::updbwsup(float srcact,float spkwgain,vector<float> &bwsup) {

    if (silent) return;

    for (int trgm=0; trgm<trgM; trgm++)

    	bwsup[trgm] += srcact * Wij[trgm] * sdep * sfac * spkwgain;

}


void Connij::setactive() {

    silent = false;

    for (int trgm=0; trgm<trgM; trgm++)	won[trgm] = prj->nextfloat()<prj->wdens;

    bstep = simstep;

    mic = -1;

    resetstate();

}


void Connij::setsilent() {

    silent = true;

    bstep = simstep;

    mic = -1;

    fill(Wij.begin(),Wij.end(),0);
    fill(won.begin(),won.end(),0);

    resetstate();

}


void Connij::setsilent(float psilent) {

    if (psilent==0) setactive();

    else if (psilent==1) setsilent();

    else if (prj->nextfloat()<psilent) setsilent(); else setactive();

}


void Connij::applywon() {

    if (won.size()==0) return;

    for (int trgm=0; trgm<trgM; trgm++) Wij[trgm] *= won[trgm];

}


void Connij::updPc(float srcact,vector<float> &trgact,float ki,float kij) {

    Pi += srcact * ki;

    for (int trgm=0; trgm<trgM; trgm++) Pij[trgm] += srcact * trgact[trgm] * kij;

}


void Connij::updPz(float srcact,vector<float> &Zj,float ki,float tauzidt,float kp,float P) {

    // Zi += (((1 - eps) * srcact + eps) * ki - Zi) * tauzidt; // ala

    Zi += (srcact * ki - Zi) * tauzidt; // ala

    if (kp<=0) return;

    Pi += (Zi - Pi) * kp;

    for (int trgm=0; trgm<trgM; trgm++)

	Pij[trgm] += (Zi * Zj[trgm] - Pij[trgm]) * kp;

}


void Connij::updPze(float srcact,vector<float> &Zj,float ki,float tauzidt,float ke,float kp) {

    // Zi += (((1 - eps) * srcact + eps) * ki - Zi) * tauzidt; // ala

    Zi += (srcact * ki - Zi) * tauzidt; // ala

    Ei += (Zi - Ei) * ke;

    for (int trgm=0; trgm<trgM; trgm++) Eij[trgm] += (Zi * Zj[trgm] - Eij[trgm]) * ke;

    if (kp<=0) return;

    Pi += (Ei - Pi) * kp;

    for (int trgm=0; trgm<trgM; trgm++)

	Pij[trgm] += (Eij[trgm] - Pij[trgm]) * kp;

}


void Connij::updBW(vector<float> &Pj,float P,float ewgain,float iwgain) {

    float logarg;

    for (int trgm=0; trgm<trgM; trgm++) {

	logarg = Pij[trgm] * P/(Pi * Pj[trgm]);

	if (logarg<=0)

	    error("Connij::updBW","logarg<=0");

	else if (logarg>1)

	    Wij[trgm] = ewgain * log(logarg);

	else

	    Wij[trgm] = iwgain * log(logarg);

    }

    applywon();

}


void Connij::updMic1(vector<float> &Pj,float P) {

    float Wij0,Wij0c,logarg1,logarg2;

    mic = 0;

    for (int trgm=0; trgm<trgM; trgm++) {

	logarg1 = Pij[trgm] * P/(Pi * Pj[trgm]);

	if (logarg1<=0) error("Connij::updMic1","logarg1<=0");

	Wij0 = log(logarg1);

	logarg2 = ((Pj[trgm] - Pij[trgm]) * P)/((P - Pi) * Pj[trgm]);

	if (logarg1<=0) error("Connij::updMic1","logarg1<=0");

	Wij0c = log(logarg2);

	mic += Pij[trgm] * Wij0 + (Pj[trgm] - Pij[trgm]) * Wij0c;

    }
}


void Connij::updMic2(vector<float> &Pj,float P) {

    float Wij0;

    mic = 0;

    for (int trgm=0; trgm<trgM; trgm++) {

	Wij0 = log(Pij[trgm] * P/(Pi * Pj[trgm]));

	mic += Pij[trgm] * Wij0;

    }
}


void Connij::updMic3() {

    mic = 0;

    for (int trgm=0; trgm<trgM; trgm++) mic += fabs(Wij[trgm]);

}


void Connij::prnp(FILE *outf) {

    fprintf(outf,"Pi = %.2e bstep = %6d silent = %d\n",Pi,bstep,silent);
    prnevec(Pij,2);
    prnevec(prj->Pj,2);
    prnvec(won);

}


Prj *Prj::getprj(int id) {

    if (id<0 or prjs.size()<=id) error("Prj::getprj","Illegal 'id'");

    return prjs[id];

}


vector<Prj *> Prj::prjs;

Prj::Prj(Pop *srcpop,Pop *trgpop,float pdens,BCPvar_t bcpvar,Prjarch_t prjarch) {

    if (setupdone) error("Prj::Prj","Not allowed when setupdone");

    id = prjs.size();

    setseed(id + 1);

    this->srcpop = srcpop;

    this->srcpoph = NULL;

    this->trgpop = trgpop;

    trgpoph = trgpop->poph;

    prjh = NULL;

    cprj = NULL;

    this->pdens = pdens;

    this->bcpvar = bcpvar;

    this->prjarch = prjarch;

    pupdfn_t = PNORMAL;

    srchoffs = 0;

    srcnoffs = 0;

    trghoffs = 0;

    trgnoffs = 0;

    setparam("taup",1);

    initstate();

    initparam();

    allocatestate();

    updidelay();

    updBW(true);

    uniformfloatdistr = uniform_real_distribution<float> (0.0,1.0);

    uniformintdistr = uniform_int_distribution<int> ();

    trgpop->inprjs.push_back(this);

    prjs.push_back(this);

}


Prj::Prj(PopH *srcpoph,Pop *trgpop,float pdens,BCPvar_t bcpvar,Prjarch_t prjarch) {

    if (setupdone) error("Prj::Prj","Not allowed when setupdone");

    id = prjs.size();

    setseed(id + 1);

    this->srcpop = NULL;

    this->srcpoph = srcpoph;

    this->trgpop = trgpop;

    trgpoph = trgpop->poph;

    prjh = NULL;

    cprj = NULL;

    this->pdens = pdens;

    this->bcpvar = bcpvar;

    this->prjarch = prjarch;

    srchoffs = 0;

    srcnoffs = 0;

    trghoffs = 0;

    trgnoffs = 0;

    setparam("taup",1);

    initstate();

    initparam();

    allocatestate();

    updidelay();

    updBW(true);

    uniformfloatdistr = uniform_real_distribution<float> (0.0,1.0);

    uniformintdistr = uniform_int_distribution<int> ();

    srcpoph->updutprjs(this);

    trgpop->inprjs.push_back(this);

    prjs.push_back(this);

}


Prj::Prj(PopH *srcpoph,Pop *trgpop,Prj *cprj,BCPvar_t bcpvar,Prjarch_t prjarch) {

    if (setupdone) error("Prj::Prj","Not allowed when setupdone");

    id = prjs.size();

    setseed(id + 1);

    this->srcpop = NULL;

    this->srcpoph = srcpoph;

    this->trgpop = trgpop;

    this->trgpoph = trgpop->poph;

    prjh = NULL;

    this->cprj = cprj;

    this->bcpvar = bcpvar;

    this->prjarch = prjarch;

    srchoffs = 0;

    srcnoffs = 0;

    trghoffs = 0;

    trgnoffs = 0;

    setparam("taup",1);

    initstate();

    initparam();

    allocatestate();

    updidelay();

    updBW(true);

    uniformfloatdistr = uniform_real_distribution<float> (0.0,1.0);

    uniformintdistr = uniform_int_distribution<int> ();

    srcpoph->updutprjs(this);

    trgpop->inprjs.push_back(this);

    prjs.push_back(this);

}


// This is for use in psrc
Prj::Prj(int srcH,int srcM,int srcU,int srcspkwgain,int trgH,int trgM,int trgU,int trgspkwgain,
	 float pdens,BCPvar_t bcpvar,Prjarch_t prjarch) {

    if (setupdone) error("Prj::Prj","Not allowed when setup is done");

    id = prjs.size();

    setseed(id + 1);

    this->srcpop = NULL;

    this->srcpoph = NULL;

    this->srcH = srcH;

    this->srcM = srcM;

    this->srcU = srcU;

    this->srcspkwgain = srcspkwgain;

    srcNh = srcM * srcU;

    srcN = srcH * srcNh;

    trgpop = NULL;

    trgpoph = NULL;

    this->trgH = trgH;

    this->trgM = trgM;

    this->trgU = trgU;

    this->trgspkwgain = trgspkwgain;

    trgNh = trgM * trgU;

    trgN = trgH * trgNh;

    trgact = vector<float>(trgN,0);

    prjh = NULL;

    cprj = NULL;

    this->pdens = pdens;

    this->bcpvar = bcpvar;

    this->prjarch = prjarch;

    srchoffs = 0;

    srcnoffs = 0;

    trghoffs = 0;

    trgnoffs = 0;

    setparam("taup",1);

    initstate();

    initparam();

    allocatestate();

    updBW(true);

    uniformfloatdistr = uniform_real_distribution<float> (0.0,1.0);

    uniformintdistr = uniform_int_distribution<int> ();

    prjs.push_back(this);

}


void Prj::initparam() {

    tauconddt = calctaudt(timestep);

    wdens = 1;

    cspeed = 1;

    tauzidt = calctaudt(timestep);

    tauzjdt = calctaudt(timestep);

    tauedt = calctaudt(timestep);

    taupdt = calctaudt(1);

    setparam(MINAGE,0);

    taubdt = calctaudt(timestep);

    prn = 0;

    pupdfn_t = PNORMAL;

    psilent = 0;

    kbjhalf = 0;

    flpscr = 1;

    bgain = ewgain = iwgain = 1;

    sp0 = 1;

    tausddt = 0;

    tausfdt = 0;

    strcplon = false;

    updint = 0;

    pspa = pspc = 0;

    if (srcpoph!=NULL) srcspkwgain = srcpoph->getparam("spkwgain");
    else if (srcpop!=NULL) srcspkwgain = srcpop->getparam("spkwgain");

    if (trgpop!=NULL) trgspkwgain = trgpop->getparam("spkwgain");

}


void Prj::configwon(float psilent,bool pfixed) {

    if (not pfixed) for (int srcn=0; srcn<srcN; srcn++) connijs[srcn] = NULL;

    switch (prjarch) {

    case USELONE:

	for (int srcn=0,trgm,trgn0; srcn<srcN; srcn++) {

	    trgm = -1;

	    if (not pfixed and nextfloat()<pdens) {

		trgm = nextint()%trgM;

		connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);

		trgn0 = trgm * trgU;

	    }

	    if (connijs[srcn]==NULL) continue;

	    if (trgm<0) {

	    	for (trgn0=0; trgn0<trgNh; trgn0++)

	    	    if (connijs[srcn]->won[trgn0]==1) break;

	    }

	    for (int trgn=trgn0; trgn<trgn0 + trgU; trgn++)

	    	connijs[srcn]->won[trgn] = nextfloat()<wdens;

	}

	return;

    case UEXCLONE:

	if (cprj!=NULL) {

	    if (cprj->getinfo("srcN")!=srcN)
		error("Prj::configwon","srcN mismatch: " + to_string(cprj->getinfo("srcN")) +
		      " -- " + to_string(srcN));

	} else

	    error("Prj::configwon","cprj may not be NULL");

	float cwdens = wdens * (1.0 + trgU/trgN);

	for (int srcn=0,trgn0,trgu; srcn<srcN; srcn++) {

	    if (cprj->connijs[srcn]==NULL) continue;

	    connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);

	    // Find which unit is on in cprj->connijs

	    for (trgn0=0; trgn0<cprj->trgNh; trgn0++)

		if (cprj->connijs[srcn]->won[trgn0]==1) break;

	    trgu = trgn0/cprj->trgU;

	    for (int trgm=0; trgm<trgM; trgm++) {

		if (trgm==trgu) continue;

		for (int trgn=trgm*trgU; trgn<(trgm+1)*trgU; trgn++)

		    connijs[srcn]->won[trgn] = nextfloat()<cwdens;

	    }
	}

	return;

    }

    switch (prjarch) {

    case HPATCHY:

	for (int srcn=0; srcn<srcN; srcn++) {

	    if (cprj!=NULL and cprj->connijs[srcn]!=NULL)

		connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);

	    else if (not pfixed and nextfloat()<pdens)

		connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgNh; trgn++) connijs[srcn]->won[trgn] = nextfloat()<wdens;

	}

	break;

  case BC_MC:

    for (int srcn=0;srcn<srcN;srcn++) {
      if (not pfixed and nextfloat()<pdens)
        if(trghoffs == srcn/srcNh) // only connect Basket Cells and Pyr MCs that are in same HC
          connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);


      if (connijs[srcn]==NULL) continue;
      for (int trgnh=0; trgnh<trgNh; trgnh++) {/**std::cout<<"\nconfigwon trgnh: "<<trgnh;**/ connijs[srcn]->won[trgnh] = nextfloat()<wdens;}

    }

break;

    case HNDIAG:

	if (srcH!=trgH) error("Prj::configwon","srcpop(h) -- trgpop H-mismatch");

	for (int srcn=0; srcn<srcN; srcn++) {

	    if (not pfixed) {

		if (trghoffs!=srcn/srcNh and nextfloat()<pdens)

		    connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);

	    }

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgNh; trgn++)

		connijs[srcn]->won[trgn] = nextfloat()<wdens;

	}

	break;

    case HDIAG:

	if (srcH!=trgH) error("Prj::configwon","srcpop(h) -- trgpop H-mismatch");

	for (int srcn=0; srcn<srcN; srcn++) {

	    if (not pfixed) {

		if (trghoffs==srcn/srcNh and nextfloat()<pdens)

		    connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);

	    }

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgNh; trgn++)

		connijs[srcn]->won[trgn] = nextfloat()<wdens;

	}

	break;

    case UDIAG:

	if (srcM!=trgM) error("Prj::configwon","srcpop(h) -- trgpop U-mismatch");

	for (int srcn=0; srcn<srcN; srcn++) {

	    if (not pfixed) {

		if (trghoffs==srcn/srcNh and nextfloat()<pdens)

		    connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);

	    }

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgNh; trgn++)

		if ((trgnoffs+trgn)/trgU==srcn/srcU)

		    connijs[srcn]->won[trgn] = nextfloat()<wdens;

		else

		    connijs[srcn]->won[trgn] = 0;
	}

	break;

    case UNDIAG:

	if (srcM!=trgM) error("Prj::configwon","srcpop(h) -- trgpop U-mismatch");

	for (int srcn=0; srcn<srcN; srcn++) {

	    if (not pfixed) {

		if (nextfloat()<pdens)

		    connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);

	    }

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgNh; trgn++) {

		    if ((trgnoffs+trgn)/trgU!=srcn/srcU)

			connijs[srcn]->won[trgn] = nextfloat()<wdens;

		}
	    }

	break;

    case HDIAG_UNDIAG:

	if (srcM!=trgM) error("Prj::configwon","srcpop(h) -- trgpop U-mismatch");

	for (int srcn=0; srcn<srcN; srcn++) {

	    if (not pfixed) {

		if (trghoffs==srcn/srcNh and nextfloat()<pdens)

		    connijs[srcn] = new Connij(this,srcn,trgNh,psilent,eps,bcpvar);

	    }

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgNh; trgn++) {

		if ((trgnoffs+trgn)/trgU!=srcn/srcU)

		    connijs[srcn]->won[trgn] = nextfloat()<wdens;

		else

		    connijs[srcn]->won[trgn] = 0;

		}
	    }

	break;

    }

    setparam("p0",sp0);

    setparam("tausddt",tausddt);

    setparam("tausfdt",tausfdt);

}


bool Prj::isconn(int srcn) { return connijs[srcn]!=NULL; }


int Prj::getwon(int srcn,int trgm) {

    if (srcn<0 or srcN<=srcn) error("Prj::getwon","Illegal 'srcn'");

    if (trgm<0 or trgN<=trgm) error("Prj::getwon","Illegal 'trgm'");

    if (connijs[srcn]!=NULL) return connijs[srcn]->getwon(trgm);

    return 0;

}


void Prj::initstate() {

    if (srcpop!=NULL) {

	srcH = 1;

	srcM = srcpop->getinfo("M");

	srcN = srcpop->getinfo("N");

	srcU = srcpop->getinfo("U");

	srcNh = srcM * srcU;

    } else if (srcpoph!=NULL) {

	srcH = srcpoph->getinfo("H");

	srcM = srcpoph->getinfo("M");

	srcN = srcpoph->getinfo("N");

	srcNh = srcpoph->getinfo("Nh");

	srcU = srcpoph->getinfo("U");

    }

    if (trgpoph!=NULL) {

	trgH = trgpoph->getinfo("H");

	trgM = trgpoph->getinfo("M");

	trgU = trgpoph->getinfo("U");

	trgNh = trgM * trgU;

	trgN = trgH * trgNh;

    } else if (trgpop!=NULL) {

	trgH = 1;

	trgM = trgpop->getinfo("M");

	trgU = trgpop->getinfo("U");

	trgNh = trgM * trgU;

	trgN = trgH * trgNh;

    }


    if (bcpvar==FIXED) 
    	eps = epsfloat;
    else if (bcpvar==COUNT)
	eps = 1;
    else 
    	eps = 1./(1./taupdt + 1) ; // timestep/(timestep/taupdt + timestep);

    P = eps;

    updbw = true;

    nstoa = nntos = 0;

}


void Prj::updidelay() {

    float dist;

    Pop *tmpsrcpop = srcpop;

    for (int srcn=0; srcn<srcN; srcn++) {

	if (srcpoph!=NULL) tmpsrcpop = srcpoph->getpop(srcn/srcNh);

	dist = distance(tmpsrcpop->getxyz(),trgpop->getxyz());

	idelay[srcn] = dist/cspeed/timestep + 1;


	if (tmpsrcpop->maxidelay>1/taupdt)

	    warning("Prj::updidelay","Excessive delays: " + to_string(tmpsrcpop->maxidelay));

	if (tmpsrcpop->maxidelay<idelay[srcn]) tmpsrcpop->maxidelay = idelay[srcn];

	tmpsrcpop->setupaxons();

    }
}


void Prj::setdelays(float delay) {

    Pop *tmpsrcpop = srcpop;

    for (int srcn=0; srcn<srcN; srcn++) {

	if (srcpoph!=NULL) tmpsrcpop = srcpoph->getpop(srcn/srcNh);

	idelay[srcn] = delay/timestep + 0.5;

	if (idelay[srcn]>1/taupdt)

	    warning("Prj::setdelays","Excessive delays: " + to_string(idelay[srcn]));

	if (tmpsrcpop->maxidelay<idelay[srcn]) tmpsrcpop->maxidelay = idelay[srcn];

	tmpsrcpop->setupaxons();

    }
}


void Prj::setdelays(vector<vector<float> > delaymat) {

    Pop *tmpsrcpop = srcpop;

    for (int srcn=0; srcn<srcN; srcn++) {

	if (srcpoph!=NULL) tmpsrcpop = srcpoph->getpop(srcn/srcNh);

	idelay[srcn] = delaymat[srcn/srcNh][trghoffs]/timestep + 0.5;

	if (idelay[srcn]>1/taupdt)

	    warning("Prj::setdelays","Excessive delays: " + to_string(idelay[srcn]));

	if (tmpsrcpop->maxidelay<idelay[srcn]) tmpsrcpop->maxidelay = idelay[srcn];

	tmpsrcpop->setupaxons();

    }
}


void Prj::allocatestate() {

    bwsup = vector<float> (trgNh,0);

    cond = vector<float> (trgNh,0);

    Bj = vector<float> (trgNh,0);

    idelay = vector<int>(srcN);

    delact = vector<float>(srcN);

    connijs = vector<Connij *>(srcN,NULL);

    switch (bcpvar) {

    case INCR_E:

    	Ej = vector<float> (trgNh,eps);

    case INCR:

    	Zj = vector<float> (trgNh,eps);

    case COUNT:

    	Pj = vector<float> (trgNh,eps);

    	break;

    case FIXED:

    	break;

    default: error("Prj::allocatestate","No such BCP variant");

    }
}


int Prj::getinfo(string infostr) {

    if (infostr=="id") return id;
    else if (infostr=="trgid" and trgpop!=NULL) return trgpop->getinfo("id");
    else if (infostr=="bcpvar") return bcpvar;
    else if (infostr=="Patch_n") {
	int npatch = 0;
	for (size_t srcn=0; srcn<connijs.size(); srcn++)
	    if (connijs[srcn]!=NULL) npatch++;
	return npatch;
    } else if (infostr=="W_n") {
	int nw = 0;
	for (size_t srcn=0; srcn<connijs.size(); srcn++)
	    if (connijs[srcn]!=NULL)
		nw += connijs[srcn]->getinfo("W_n");
	return nw;
    } else if (infostr=="srcN") return srcN;
    else if (infostr=="srcNh") return srcNh;
    else if (infostr=="trgM") return trgM;
    else if (infostr=="trgN") return trgN;
    else if (infostr=="trgNh") return trgNh;
    else error("Prj::getinfo","No such 'infostr'");

    return 0;

}


void Prj::setsrchnoffs(int srchoffs,int srcnoffs) {

    this->srchoffs = srchoffs;

    this->srcnoffs = srcnoffs;

}


void Prj::settrghnoffs(int trghoffs,int trgnoffs) {

    this->trghoffs = trghoffs;

    this->trgnoffs = trgnoffs;

}


void Prj::setparam(string paramstr,float paramval) {

    if (paramstr=="pdens")
	error("Prj::setparam","Parameter 'pdens' cannot be set");
    else if (paramstr=="wdens") {
	wdens = paramval;
	configwon(psilent,true);
    } else if (paramstr=="psilent") {
	if (1<=paramval)
	    error("Prj::setparam","Illegal: 'psilence'<=0");
	psilent = paramval;
	for (int srcn=0; srcn<srcN; srcn++)
	    if (connijs[srcn]!=NULL)
		connijs[srcn]->setsilent(psilent);
    } else if (paramstr=="bcpvar")
	error("Prj::setparam","Parameter 'bcpvar' cannot be set");
    else if (paramstr=="taucond")
	tauconddt = calctaudt(paramval);
    else if (paramstr=="taup") {
	taupdt = calctaudt(paramval);
	eps = 1./(1./taupdt + 1);
    } else if (paramstr=="taub")
	taubdt = calctaudt(paramval);
    else if (paramstr=="eps")
	error("Prj::setparam","Parameter 'eps' cannot be set");
    else if (paramstr=="taue")
	tauedt = calctaudt(paramval);
    else if (paramstr=="tauzi")
	tauzidt = calctaudt(paramval);
    else if (paramstr=="tauzj")
	tauzjdt = calctaudt(paramval);
    else if (paramstr=="prn")
	prn = paramval;
    else if (paramstr=="bgain") {
	bgain = paramval;
	updBW(true);
    } else if (paramstr=="kbjhalf") {
	kbjhalf = paramval;
	if (kbjhalf!=0 and kBj.size()==0) kBj = vector<float>(trgNh,0);
	updBW(true);
    } else if (paramstr=="flpscr")
	flpscr = paramval;
    else if (paramstr=="minage")
	minage = paramval/timestep;
    else if (paramstr=="pspa") {
	pspa = paramval;
    } else if (paramstr=="pspc") {
	pspc = paramval;
    } else if (paramstr=="ewgain") {
	ewgain = paramval;
	updBW(true);
    } else if (paramstr=="iwgain") {
	iwgain = paramval;
	updBW(true);
    } else if (paramstr=="wgain") {
	setparam("ewgain",paramval); setparam("iwgain",paramval);
    } else if (paramstr=="cspeed") {
	cspeed = paramval;
	updidelay();
    } else if (paramstr=="p0") {
	sp0 = paramval;
	for (int srcn=0; srcn<srcN; srcn++)
	    if (connijs[srcn]!=NULL)
		connijs[srcn]->sp0 = paramval;
    } else if (paramstr=="tausddt") {
	tausddt = paramval;
	for (int srcn=0; srcn<srcN; srcn++)
	    if (connijs[srcn]!=NULL)
		connijs[srcn]->tausddt = tausddt;
    } else if (paramstr=="tausd") {
	if (paramval<=0)
	    tausddt = 0;
	else
	    tausddt = calctaudt(paramval);
	setparam("tausddt",tausddt);
    } else if (paramstr=="tausfdt") {
	tausfdt = paramval;
	for (int srcn=0; srcn<srcN; srcn++)
	    if (connijs[srcn]!=NULL)
		connijs[srcn]->tausfdt = tausfdt;
    } else if (paramstr=="tausf") {
	if (paramval<=0)
	    tausfdt = 0;
	else
	    tausfdt = calctaudt(paramval);
	setparam("tausfdt",tausfdt);

    }

    else error("Prj::setparam","Illegal paramstr: " + paramstr);

}

void Prj::setparam(Prjparam_t param_t,float parval) {

    switch (param_t) {

    case PDENS: setparam("pdens",parval); break;
    case EPS: setparam("eps",parval); break;
    case WDENS: setparam("wdens",parval); break;
    case PSILENT: setparam("psilent",parval); break;
    case BCPVAR: setparam("bcpvar",parval); break;
    case TAUCOND: setparam("taucond",parval); break;
    case TAUP: setparam("taup",parval); break;
    case TAUB: setparam("taub",parval); break;
    case TAUE: setparam("taue",parval); break;
    case TAUZI: setparam("tauzi",parval); break;
    case TAUZJ: setparam("tauzj",parval); break;
    case PRN: setparam("prn",parval); break;
    case BGAIN: setparam("bgain",parval); break;
    case KBJHALF: setparam("kbjhalf",parval); break;
    case FLPSCR: setparam("flpscr",parval); break;
    case MINAGE: setparam("minage",parval); break;
    case PSPA: setparam("pspa",parval); break;
    case PSPC: setparam("pspc",parval); break;
    case EWGAIN: setparam("ewgain",parval); break;
    case IWGAIN: setparam("iwgain",parval); break;
    case WGAIN: setparam("wgain",parval); break;
    case CSPEED: setparam("cspeed",parval); break;
    case P0: setparam("p0",parval); break;
    case TAUSD: setparam("tausd",parval); break;
    case TAUSF: setparam("tausf",parval); break;

    default:
	printf("%d\n",param_t);
	error("Prj::setparam","No such paramstr: " + prjparam2str(param_t));
	break;

    }
}


float Prj::getparam(Prjparam_t param_t) {

    switch (param_t) {

    case PDENS: return pdens; break;
    case WDENS: return wdens; break;
    case TAUCOND: return timestep/tauconddt; break;
    case TAUZI: return timestep/tauzidt; break;
    case TAUZJ: return timestep/tauzjdt; break;
    case TAUE: return timestep/tauedt; break;
    case TAUP: return timestep/taupdt; break;
    case TAUB: return timestep/taubdt; break;
    case P0: if (connijs.size()!=0) return connijs[0]->sp0; break;
    case TAUSD: return timestep/tausddt; break;
    case TAUSF: return timestep/tausfdt; break;
    case PRN: return prn; break;
    case BGAIN: return bgain; break;
    case EWGAIN: return ewgain; break;
    case IWGAIN: return iwgain; break;
    case KBJHALF: return kbjhalf; break;
    case CSPEED: return cspeed; break;

    default: error("Prj::getparam","No such paramstr: " + prjparam2str(param_t));

    }

    return 0;

}


void Prj::setstate(string statestr,vector<float> statevec) {

    if (statestr=="Pi") {

	if (statevec.size()!=srcN)

		error("Prj::setstate(Pi)","statevec size mismatch : " +
		      to_string(srcN));

	for (int srcn=0; srcn<srcN; srcn++) {

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgNh; trgn++) connijs[srcn]->Pi = statevec[srcn];

	}

    } else if (statestr=="Bj" or statestr=="Pj") {

	if (statevec.size()<trgnoffs + trgNh) error("Prj::setstate(Bj/Pj)","statevec size mismatch");

	for (int n=0; n<trgNh; n++)

	    if (statestr=="Bj")

		Bj[n] = statevec[trgnoffs + n];

	    else if (statestr=="Pj")

		Pj[n] = statevec[trgnoffs + n];

    } else

	error("Prj::setstate","Cannot set this state: " + statestr);

    if (statestr=="Pi" or statestr=="Pj") updBW(true);

}


void Prj::setstate(string statestr,vector<vector<float> > statemat,bool pfixed) {

    if (statestr=="Wij" or statestr=="Pij") {

	configwon(psilent,pfixed);

	if (srcN<statemat.size())

		error("Prj::setstate","statemat r-dim mismatch : " +
		      to_string(srcN) + " -- " + to_string(statemat.size()));

	    else if (statemat[0].size()<trgnoffs + trgNh)

		error("Prj::setstate","statemat c-dim mismatch : " +
		      to_string(trgnoffs+trgNh) + " -- " + to_string(statemat[0].size()));

	for (int srcn=0; srcn<srcN; srcn++) {

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgNh; trgn++)

		if (statestr=="Wij") {

		    connijs[srcn]->Wij[trgn] = statemat[srcn][trgnoffs + trgn];

		    connijs[srcn]->applywon();

		} else if (statestr=="Pij")

		    connijs[srcn]->Pij[trgn] = statemat[srcn][trgnoffs + trgn];

	}

    } else

	error("Prj::setstate","Cannot set this state: " + statestr);

    if (statestr=="Pij") updBW(true);

}


void Prj::setstate(string statestr,float stateval,bool pfixed) {

    if (statestr=="P") {

	P = stateval;

	updbw = true;

    } else if (statestr=="Bj") {

	for (int n=0; n<trgNh; n++) Bj[n] = stateval;

    } else if (statestr=="Wij") {

	configwon(psilent,pfixed);

	for (int srcn=0; srcn<srcN; srcn++) {

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgNh; trgn++)

		connijs[srcn]->Wij[trgn] = stateval * connijs[srcn]->won[trgn];

	}

    } else

	error("Prj::setstate","Cannot set this state: " + statestr);

    if (statestr=="P") updBW();
}


float Prj::getstate(string statestr) {

    if (statestr=="P")
	return P;

    else if (statestr=="nstoa") return nstoa;
    else if (statestr=="nntos") return nntos;

    else error("Prj::getstate","Illegal statestr: " + statestr);

    return -1;

}


vector<float> const &Prj::getstatei(string statestr) {

    if (statestr=="delact") return delact;

    if ((statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="Mic" or statestr=="Sil" or
	 statestr=="Age") and bcpvar==FIXED) {

	warning("Prj::getstatei","Illegal statestr for bcpvar==FIXED");

	return tmpstatei;

    }

    if (tmpstatei.size()!=srcN) tmpstatei = vector<float>(srcN,0);
    else fill(tmpstatei.begin(),tmpstatei.end(),0);

    switch (bcpvar) {

    case INCR_E:

	if (statestr=="Ei")

	    for (int srcn=0; srcn<srcN; srcn++) if (connijs[srcn]!=NULL) tmpstatei[srcn] = connijs[srcn]->Ei;

    case INCR:

	if (statestr=="Zi") {

	    for (int srcn=0; srcn<srcN; srcn++) if (connijs[srcn]!=NULL) tmpstatei[srcn] = connijs[srcn]->Zi;

	} else if (statestr=="Mic") {

	    if (statestr=="Mic") updMic(true);

	    for (int srcn=0; srcn<srcN; srcn++) if (connijs[srcn]!=NULL)

		    tmpstatei[srcn] = connijs[srcn]->mic;

	} else if (statestr=="Sil") {

	    for (int srcn=0; srcn<srcN; srcn++)

		if (connijs[srcn]!=NULL) tmpstatei[srcn] = connijs[srcn]->silent;

	} else if (statestr=="Age") {

	    for (int srcn=0; srcn<srcN; srcn++)

		if (connijs[srcn]!=NULL) tmpstatei[srcn] = connijs[srcn]->getinfo("age");

	}

    case COUNT:

	if (statestr=="Pi")

	    for (int srcn=0; srcn<srcN; srcn++) if (connijs[srcn]!=NULL) tmpstatei[srcn] = connijs[srcn]->Pi;

	break;

    case FIXED:

	if (statestr=="sdep") {

	    for (int srcn=0; srcn<srcN; srcn++) if (connijs[srcn]!=NULL) tmpstatei[srcn] = connijs[srcn]->sdep;

	} else if (statestr=="sfac") {

	    for (int srcn=0; srcn<srcN; srcn++) if (connijs[srcn]!=NULL) tmpstatei[srcn] = connijs[srcn]->sfac;

	} else error("Prj::getstatei","Illegal statestr (bcpvar==FIXED)");

	break;

    default: error("Prj::getstatei","Illegal statestr:" + statestr);

    }

    return tmpstatei;

}


vector<float> const &Prj::getstatej(string statestr) {

    if ((statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="kBj")	and bcpvar==FIXED) {

	fill(tmpstatej.begin(),tmpstatej.end(),0);

	warning("Prj::getstatej","Illega statestr for bcpvar==FIXED");

	return tmpstatej;

    }

    if (statestr=="bwsup") return bwsup;

    else if (statestr=="cond") return cond;

    else if (statestr=="Bj") {

	updBW();

	return Bj;

    } else if (statestr=="kBj") return kBj;

    switch (bcpvar) {

    case INCR_E:

	if (statestr=="Ej") return Ej;

    case INCR:

	if (statestr=="Zj") return Zj;

    case COUNT:

	if (statestr=="Pj") return Pj;

	break;

    case FIXED:

	error("Prj::getstatej","Illegal statestr (bcpvar==FIXED)"); return tmpstatej;

	break;

    default: error("Prj::getstatej","Illegal statestr:" + statestr);

    }

    error("Prj::getstatej","Illegal statestr:" + statestr);

    return tmpstatej;

}


vector<vector<float> > const &Prj::getstateij(string statestr) {

    if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won") {

	if (tmpstateij.size()!=srcN or tmpstateij[0].size()!=trgN)

	    tmpstateij = vector<vector<float> >(srcN,vector<float>(trgN,0));

	if ((statestr=="Eij" or statestr=="Pij") and bcpvar==FIXED) {

	    warning("Prj::getstateij","Illegal statestr for bcpvar==FIXED");

	    for (int srcn=0; srcn<srcN; srcn++)

		fill(tmpstateij[srcn].begin(),tmpstateij[srcn].end(),0);

	    return tmpstateij;

	}

	for (int srcn=0; srcn<srcN; srcn++) {

	    fill(tmpstateij[srcn].begin(),tmpstateij[srcn].end(),0);

	    if (connijs[srcn]==NULL) continue;

	    for (int trgn=0; trgn<trgN; trgn++) {

		if (statestr=="Wij") {

		    tmpstateij[srcn][trgn] = connijs[srcn]->Wij[trgn];

		    if (tausddt>0) tmpstateij[srcn][trgn] *= connijs[srcn]->sdep;

		    if (tausfdt>0) tmpstateij[srcn][trgn] *= connijs[srcn]->sfac;

		}

		else if (statestr=="Won")

		    tmpstateij[srcn][trgn] = connijs[srcn]->won[trgn];

		else if (statestr=="Pij") tmpstateij[srcn][trgn] = connijs[srcn]->Pij[trgn];

		else if (statestr=="Eij" and bcpvar==INCR_E)
		    tmpstateij[srcn][trgn] = connijs[srcn]->Eij[trgn];

		else error("Prj::getstate","Illegal statestr");

	    }
	}

	return tmpstateij;

    }

    return tmpstateij;

}


void Prj::fetchstatei(string statestr,vector<float> &statevec) {

    fill(statevec.begin(),statevec.end(),0);

    if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="sdep" or
	statestr=="sfac" or statestr=="delact" or statestr=="Mic" or statestr=="Sil" or statestr=="Age") {

	statevec = getstatei(statestr);

    } else error("Prj::fetchstatei","Illegal statestr " + statestr);

}


void Prj::fetchstatej(string statestr,vector<float> &statevec) {

    if (statestr=="Bj") updBW();

    if (statestr=="Zj")	cpyvectovec(Zj,statevec,trgnoffs);

    else if (statestr=="Ej") cpyvectovec(Ej,statevec,trgnoffs);

    else if (statestr=="Pj") cpyvectovec(Pj,statevec,trgnoffs);

    else if (statestr=="Bj") cpyvectovec(Bj,statevec,trgnoffs);

    else if (statestr=="kBj") cpyvectovec(kBj,statevec,trgnoffs);

    else if (statestr=="bwsup") cpyvectovec(bwsup,statevec,trgnoffs);

    else if (statestr=="cond") cpyvectovec(cond,statevec,trgnoffs);

    else error("Prj::fetchstatej","Illegal statestr " + statestr);

}


void Prj::fetchstateij(string statestr) {

    if (tmpstateij.size()!=srcN or tmpstateij[0].size()!=trgNh)
	tmpstateij = vector<vector<float> >(srcN,vector<float> (trgNh,0));

    else for (int r=0; r<srcN; r++) fill(tmpstateij[r].begin(),tmpstateij[r].end(),0);

    for (int srcn=0; srcn<srcN; srcn++)

	fill(tmpstateij[srcn].begin(),tmpstateij[srcn].end(),0);

    if (statestr=="Pij") {

	for (int srcn=0; srcn<srcN; srcn++)

	    if (connijs[srcn]!=NULL)

		tmpstateij[srcn] = connijs[srcn]->Pij;

    } else if (statestr=="Eij" and bcpvar==INCR_E) {

	for (int srcn=0; srcn<srcN; srcn++)

	    if (connijs[srcn]!=NULL)

		tmpstateij[srcn] = connijs[srcn]->Eij;

    } else if (statestr=="Won") {

	for (int srcn=0; srcn<srcN; srcn++)

	    if (connijs[srcn]!=NULL)

		tmpstateij[srcn] = connijs[srcn]->won;

    } else if (statestr=="Wij") {

	updBW();

	for (int srcn=0; srcn<srcN; srcn++)

	    if (connijs[srcn]!=NULL)

		tmpstateij[srcn] = connijs[srcn]->Wij;

    } else error("Prj::fetchstateij","Illegal statestr: " + statestr);

}


void Prj::fetchstateij(string statestr,vector<vector<float> > &statemat) {

    fetchstateij(statestr);

    cpyrectomat(statemat,tmpstateij,0,trgnoffs);

}


void Prj::fetchstate(string statestr) {

    if (bcpvar==FIXED and (statestr=="P" or statestr=="Pi" or statestr=="Pj" or statestr=="Pij"))
	error("prj::logstate","Illegal statestr for bcpvar==FIXED: " + statestr);
    else if ((bcpvar==FIXED or bcpvar==INCR) and (statestr=="Ei" or statestr=="Ej" or statestr=="Eij"))
	error("prj::logstate","Illegal statestr for bcpvar==FIXED/INCR: " + statestr);

    if (statestr=="P")

	tmpstate = getstate(statestr);

    else if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="delact" or statestr=="Mic"
	     or statestr=="Sil" or statestr=="Age") {

	if (tmpstatei.size()!=srcN) tmpstatei = vector<float>(srcN,0);

	fetchstatei(statestr,tmpstatei);

    } else if (statestr=="sdep" or statestr=="sfac") {

	if (tmpstatei.size()!=srcN) tmpstatei = vector<float>(srcN,0);

	fetchstatei(statestr,tmpstatei);

    } else if (statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="Bj" or statestr=="kBj" or
	       statestr=="bwsup" or statestr=="cond") {

	if (tmpstatej.size()==0) tmpstatej = vector<float>(trgNh,0);

	fetchstatej(statestr,tmpstatej);

    } else if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won") {

	if (tmpstateij.size()==0) tmpstateij = vector<vector<float> > (srcNh,vector<float>(trgN,0));

	fetchstateij(statestr,tmpstateij);

    } else error("Prj::fetchstate","Illegal statestr: " + statestr);

}


void Prj::fwritestateval(string statestr,FILE *outf) {

    fwriteval(getstate(statestr),outf);

}


void Prj::fwritestatevec(string statestr,FILE *outf) {

    fetchstate(statestr);

    if (statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="Bj" or statestr=="kBj" or
	statestr=="bwsup" or statestr=="cond") {

	fwriteval(tmpstatej,outf);

    } else error("Prj::fwritestate","Illegal statestr: " + statestr);

}


void Prj::fwritestatemat(string statestr,FILE *outf) {

    if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="delact" or statestr=="Mic" or
	statestr=="Sil" or statestr=="Age" or statestr=="sdep" or statestr=="sfac") {

	fetchstatei(statestr,tmpstatei);

	fwriteval(tmpstatei,outf);

    } else if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won") {

	fetchstateij(statestr,tmpstateij);

	fwritemat(tmpstateij,outf);

    }

    else error("Prj::fwritestate","Illegal statestr: " + statestr);

}


void Prj::fwritestate(string statestr,FILE *outf) {

    fetchstate(statestr);

    if (statestr=="P")

	fwritestateval(statestr,outf);

    else if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="Mic" or statestr=="Sil" or
	      statestr=="Sil" or statestr=="delact")

	fwritestatevec(statestr,outf);

    else if (statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="Bj" or statestr=="kBj" or
	     statestr=="bwsup" or statestr=="cond") {

	fwritestatevec(statestr,outf);

    } else if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won" or
	       statestr=="sdep" or statestr=="sfac") {

	fwritestatemat(statestr,outf);

    } else error("Prj::fwritestate","Illegal statestr: " + statestr);

}


void Prj::fwritestate(string statestr,string filename) {

    FILE *outf = fopen(filename.c_str(),"w");

    if (outf==NULL) error("Prj::fwritestate","Could not open file" + filename);

    fwritestate(statestr,outf);

    fclose(outf);

}


Logger *Prj::logstate(std::string statestr,std::string filename,int logstep) {

    return new Logger(this,statestr,filename,logstep);

}


void Prj::resetstate() {

    fill(delact.begin(),delact.end(),0);

    fill(bwsup.begin(),bwsup.end(),0);

    fill(cond.begin(),cond.end(),0);

    for (size_t srcn=0; srcn<connijs.size(); srcn++)

	if (connijs[srcn]!=NULL) connijs[srcn]->resetstate();

    switch (bcpvar) {

    case INCR_E:

	fill(Ej.begin(),Ej.end(),eps);

    case INCR:

	fill(Zj.begin(),Zj.end(),eps);

    case COUNT: break;

    case FIXED: break;

    default: error("Prj::resetstate","Illegal bcpvar");

    }

    resetbwsup();

}


void Prj::resetstateall() {

    for (size_t h=0; h<prjs.size(); h++) prjs[h]->resetstate();

}


void Prj::resetbwsup() {

    for (int n=0; n<trgNh; n++) bwsup[n] = Bj[n];

}


void Prj::resetbwsupall() {

    for (size_t c=0; c<prjs.size(); c++) prjs[c]->resetbwsup();

}


void Prj::upddelact() {

    for (int srch=0; srch<srcH; srch++) {

	for (int srcn=0; srcn<srcNh; srcn++) {

	    if (srcpoph!=NULL and idelay[srch*srcNh+srcn]>0)

		delact[srch*srcNh+srcn] = srcpoph->getpop(srch)->getdelact(srcn,idelay[srch*srcNh+srcn]);

	    else if (idelay[srcn]>0)

		delact[srcn] = srcpop->getdelact(srcn,idelay[srcn]);

	}

    }
}


void Prj::setpupdfn(Pupdfn_t pupdfn_t) { this->pupdfn_t = pupdfn_t; }


long Prj::setseed(long seed) {

    if (seed==0) seed = random_device{}();

    if (seed<0) seed = -seed;

    seed = 20000000 + (seed + 17*id)%10000000;

    generator.seed(seed);

    this->seed = seed;

    return seed;

}


long Prj::getseed() { return this->seed; }


float Prj::nextfloat() { return uniformfloatdistr(generator); }


int Prj::nextint() { return uniformintdistr(generator); }


void Prj::updsdep() {

    if (tausddt<=0) return;

    for (int srcn=0; srcn<srcN; srcn++)

	if (connijs[srcn]!=NULL) connijs[srcn]->updsdep(delact[srcn],srcspkwgain);

}


void Prj::updsfac() {

    if (tausfdt<=0) return;

    for (int srcn=0; srcn<srcN; srcn++)

	if (connijs[srcn]!=NULL) connijs[srcn]->updsfac(delact[srcn],srcspkwgain);

}

void Prj::propagate() {

    if (updbw) updBW();

    resetbwsup();

    for (int srcn=0; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL or connijs[srcn]->silent) continue;

	connijs[srcn]->updbwsup(delact[srcn],srcspkwgain,bwsup);

    }

    for (size_t n=0; n<trgNh; n++) cond[n] += (bwsup[n] - cond[n]) * tauconddt;

    trgpop->contribute(cond);

}


void Prj::updPc() {

    float ki = srcspkwgain,kj = trgspkwgain,kij = srcspkwgain * trgspkwgain;

    P += prn;

    vector<float> trgact = trgpop->getstate("act");

    for (int trgn=0; trgn<trgNh; trgn++) Pj[trgn] += trgact[trgn] * kj * prn;

    for (int srcn=0; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL) continue;

	connijs[srcn]->updPc(delact[srcn],trgact,ki * prn,kij * prn);

    }

    updbw = true;

}


void Prj::updZj() {

    for (int trgn=0; trgn<trgNh; trgn++)

      // Zj[trgn] += (((1 - eps) * trgpop->getstate("act")[trgn] + eps) * trgspkwgain - Zj[trgn]) * tauzjdt; // ala

      Zj[trgn] += (trgpop->getstate("act")[trgn] * trgspkwgain - Zj[trgn]) * tauzjdt; // ala

}


void Prj::updPz() {

  //if (prn<=0) return; // nbrav // ala reverted

    float kp = taupdt * prn;

    updZj();

    if (prn>0) {

	if (pupdfn_t==PNORMAL) P += (1 - P) * kp; else P = 1;

	for (int trgn=0; trgn<trgNh; trgn++) Pj[trgn] += (Zj[trgn] - Pj[trgn]) * kp;

    }

    for (int srcn=0; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL) continue;

	connijs[srcn]->updPz(delact[srcn],Zj,srcspkwgain,tauzidt,kp,P);

    }

    if (prn>0) updbw = true;

}


void Prj::updPze() {

    updZj();

    for (int trgn=0; trgn<trgNh; trgn++)

	Ej[trgn] += (Zj[trgn] - Ej[trgn]) * tauedt;

    if (prn>0) {

	if (pupdfn_t==PNORMAL) P += (1 - P) * taupdt * prn; else P = 1;

	for (int trgn=0; trgn<trgNh; trgn++) Pj[trgn] += (Ej[trgn] - Pj[trgn]) * taupdt * prn;

    }

    for (int srcn=0; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL) continue;

	connijs[srcn]->updPze(delact[srcn],Zj,srcspkwgain,tauzidt,tauedt,taupdt * prn);

    }

    if (prn>0) updbw = true;

}


void Prj::updP() {

    switch (bcpvar) {

    case FIXED: break;

    case COUNT: updPc(); break;

    case INCR: updPz(); break;

    case INCR_E: updPze(); break;

    default: error("Prj::updP","No such bcpvar");

    }
}


void Prj::updkbj() {

    float pme = 0.25/trgM,pj;

    for (int trgn=0; trgn<trgNh; trgn++) {

	pj = Pj[trgn]/P;

	kBj[trgn] += (1 + (kbjhalf - 1) * pme/(pj - pme) * pme/(pj - pme) - kBj[trgn]) * taubdt;

    }
}


void Prj::updBW(bool force) {

    if (not force and not updbw) return;

    if (bcpvar!=FIXED) {

	if (kbjhalf==0)

	    for (size_t trgn=0; trgn<Bj.size(); trgn++)

			if (Pj[trgn]<eps/2.) Bj[trgn] = bgain * log(eps); // nbrav
			else Bj[trgn] = bgain * log(Pj[trgn]/P);

	else {

	    updkbj();

	    for (size_t trgn=0; trgn<Bj.size(); trgn++)

			if (Pj[trgn]<eps/2.) Bj[trgn] = bgain * log(eps); // nbrav
			else Bj[trgn] = kBj[trgn] * log(Pj[trgn]/P); // bgain is ineffective

	}

	for (size_t srcn=0; srcn<connijs.size(); srcn++)

	    if (connijs[srcn]!=NULL) connijs[srcn]->updBW(Pj,P,ewgain,iwgain);

    }

    updbw = false;

}


void Prj::setstrcpl(bool on) { strcplon = on; }


void Prj::updMic(bool force) {

    if (prjh==NULL) error("Prj::updMic","Illegal prjh==NULL");

    if (flpscr==3) updBW(true);

    for (int srcn=0; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL) continue;

	if (not force and (connijs[srcn]->getinfo("age")<minage)) { connijs[srcn]->mic = -1; continue; }

	switch (flpscr) {

	case 1: connijs[srcn]->updMic1(Pj,P); break;

	case 2: connijs[srcn]->updMic2(Pj,P); break;

	case 3: connijs[srcn]->updMic3(); break;

	default: error("Prj::updMic","No such 'flpscr'");

	}

	if (prjh->getntpatch(srcn)>0) connijs[srcn]->mic /= pow(prjh->getntpatch(srcn),1.25);

    }
}


bool Prj::getsilminmaxscore(float &scmin,int &scminidx,float &scmax,int &scmaxidx) {

    // The 'max' is calculated for silent connijs and 'min' for active connijs

    float mic;

    bool found = false;

    for (int srcn=1; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL or not connijs[srcn]->silent or connijs[srcn]->mic==-1) continue;

	mic = connijs[srcn]->mic;

	scmin = mic; scminidx = srcn; scmax = mic; scmaxidx = srcn;

	break;

    }

    for (int srcn=1; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL or not connijs[srcn]->silent or connijs[srcn]->mic==-1) continue;

	found = true;

	mic = connijs[srcn]->mic;

	if (mic>scmax) { scmax = mic; scmaxidx = srcn; }

	if (mic<scmin) { scmin = mic; scminidx = srcn; }

    }

    return found;

}


bool Prj::getactminmaxscore(float &scmin,int &scminidx,float &scmax,int &scmaxidx) {

    // The 'max' is calculated for silent connijs and 'min' for active connijs

    float mic;

    bool found = false;

    for (int srcn=0; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL or connijs[srcn]->silent or connijs[srcn]->mic==-1) continue;

	mic = connijs[srcn]->mic;

	scmin = mic; scminidx = srcn; scmax = mic; scmaxidx = srcn;

	break;

    }

    /// A bit clumsy, but ...

    for (int srcn=0; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL or connijs[srcn]->silent or connijs[srcn]->mic==-1) continue;

	found = true;

	mic = connijs[srcn]->mic;

	/// if (id==4) { printf("srcn = %3d mic = %.2e ",srcn,mic); connijs[srcn]->prnp(); }

    }

    /// if (id==4) printf("\n");

    return found;

}


float Prj::getactmicscoremean() {

    float actmicsum = 0;

    bool found = false;

    int nact = 0;

    for (int srcn=0; srcn<srcN; srcn++) {

	if (connijs[srcn]==NULL or connijs[srcn]->silent or connijs[srcn]->mic==-1) continue;

	actmicsum += connijs[srcn]->mic;

	nact++;

    }

    if (nact>0) actmicsum /= nact;

    return actmicsum;

}



bool Prj::qsiltoact() {

    if (pspa<=0) return false;

    if (prjh==NULL) error("Prj::qsiltoact","Illegal prjh==NULL");

    int scminidx_sil,scmaxidx_sil,scminidx_act,scmaxidx_act;

    float scmin_sil = 0,scmax_sil = 0,scmin_act = 0,scmax_act = 0,ksc = 1;

    bool foundsilmax = false,foundactmin = false;

    if (nextfloat()<pspa) {

	updMic();

	foundsilmax = getsilminmaxscore(scmin_sil,scminidx_sil,scmax_sil,scmaxidx_sil);

	foundactmin = getactminmaxscore(scmin_act,scminidx_act,scmax_act,scmaxidx_act);

	if (foundsilmax and foundactmin and (scmax_sil>ksc * scmin_act)) {

	    nstoa++;

	    // simstep scmaxidx_sil scmax_sil scminidx_act scmin_act scratio actmicscoremean nflip

	    // printf("%8d %3d %3d %.2e %3d %.2e %.2e %.2e %3d\n",
	    // 	   simstep,id,
	    // 	   scmaxidx_sil,scmax_sil,
	    // 	   scminidx_act,scmin_act,
	    // 	   scmax_sil/scmin_act,
		//    getactmicscoremean(),
	    // 	   nstoa);

	    fflush(stdout);

	    connijs[scminidx_act]->setsilent(); connijs[scminidx_act]->mic = -1;

	    connijs[scmaxidx_sil]->setactive(); connijs[scmaxidx_sil]->mic = -1;

	    return true;

	}
    }

    return false;

}


void Prj::qsiltoactn(int n,int updint) {

    if (not strcplon) return;

    float oldpspa = pspa;

    pspa = 1;

    this->updint = updint;

    int tmp = prjh->trgH * updint;

    if (simstep%tmp!=updint*id) return;

    for (int i=0; i<n; i++) qsiltoact();

    pspa = oldpspa;

}


int Prj::getrndnew() {

    int srcn0 = abs(nextint())%srcN;

    for (int srcn = srcn0; srcn<srcn0 + srcN; srcn++)
	if (connijs[srcn%srcN]!=NULL) { srcn0 = srcn%srcN; break; }

    if (connijs[srcn0]==NULL) error("Prj::getrndnew","connijs[srcn]==NULL: " + to_string(srcn0));

    return srcn0;

}


bool Prj::qnewtosil() {



    if (pdens==1 and pspc>0) error("Prj::qnewtosil","Illegal pscp>0 when pdens==1");

    if (pspc<=0) return false;

    if (prjh==NULL) error("Prj::qnewtosil","Illegal prjh==NULL");

    int scminidx_sil,scmaxidx_sil,scminidx_new,scmaxidx_new;

    float scmin_sil = 0,scmax_sil = 0,scmin_new = 0,scmax_new = 0;

    if (nextfloat()<pspc) {

	updMic();

	getsilminmaxscore(scmin_sil,scminidx_sil,scmax_sil,scmaxidx_sil);

	scmaxidx_new = getrndnew();

	// printf("%6d %2d %f\n",simstep,id,scmin_sil); fflush(stdout);

	if (scmin_sil<1) { // <<<<<<<<<<<<<<<<< Always picking new

	    // printf("qnewtosil: %4d id = %2d flip %3d--> act %3d--> sil\n",
	    // 	   simstep,id,scmaxidx_new,scminidx_sil);

	    // printf("%d %f %d\n",scminidx_sil,scmin_sil,scmaxidx_new);

	    fflush(stdout);

	    connijs[scminidx_sil] = NULL;

	    connijs[scmaxidx_new] = new Connij(this,scmaxidx_new,trgNh,1,eps,bcpvar);

	    nntos++;

	    return true;

	}
    }

    return false;

}


void Prj::updconnijs() {

    if (not strcplon) return;

    if (pdens==0) error("Prj::mvconnij","pdens==0");

    if (updint>0) return;

    qnewtosil();

    qsiltoact();

}


void Prj:: update() {

    upddelact();

    resetbwsup();

    propagate();

    updP();

    updsdep();

    updsfac();

    if (prn>0) updconnijs();

}


 void Prj::updateall() {

     for (size_t c=0; c<prjs.size(); c++) prjs[c]->update();

 }
