#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string.h>
#include <stdio.h>

#include "bcpnnsim.h"
#include "pbcpnnsim.h"
#include "PPopR.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;


int PPopR::maxidelay = 0;

PPopR::PPopR(int npop0,int rank0,int nrank,int hoffs,int HperR,int M,int U,Actfn_t actfn_t,
	     Normfn_t normfn_t) : PPobjR(rank0,nrank) {

    this->HperR = HperR;

    this->NperH = M * U;

    NperR  = HperR * NperH;

    N = nrank * NperR;

    noffs = hoffs * M * U;

    issrcpop = istrgpop = false;

    isspiking = not (actfn_t==BCP or actfn_t==LIN or actfn_t==LOG or actfn_t==EXP);

    ninprj = 0;

    Esyn = -1;

    locppopr = this;

    poph = new PopH(HperR,NperH,actfn_t,normfn_t);

    pops = poph->pops;

    for (size_t p=0; p<pops.size(); p++) {

	pops[p]->poph = poph;

    	pops[p]->sethoffs(hoffs + p);

	pops[p]->setnoffs(noffs + p*NperH);

	pops[p]->id = npop0 + p;

	pops[p]->setseed(pops[p]->id + 1);

    }
    
    ppact = std::vector<float>(NperR,0);

    ppactnz = std::vector<float>(1 + 2*NperR,0);
    
}


void PPopR::setseed(long newseed) {

    for (size_t p=0; p<pops.size(); p++) pops[p]->setseed(pops[p]->id + 1 + newseed);

}


void PPopR::setparam(Popparam_t poppar_t,float parval) {

    if (locppopr==NULL) return;

    poph->setparam(poppar_t,parval);

}


void PPopR::reallocaxons(float maxdelay) {

    if (locppopr==NULL) return;

    PPopR::maxidelay = maxdelay/timestep + 1;

    Shmem::reqshmem(AXNMEM,NperR,maxidelay);

}


void PPopR::setmaxdelay(float maxdelay) {

    for (size_t p=0; p<pops.size(); p++) {

	pops[p]->setmaxidelay(maxdelay);

	pops[p]->setupaxons();
	
    }
}


void PPopR::resetstate() {

    for (size_t p=0; p<pops.size(); p++) pops[p]->resetstate();

}


void PPopR::setinp(float inpval) {

    for (size_t p=0; p<pops.size(); p++) pops[p]->setinp(inpval);

}


void PPopR::setinp(vector<float> inp) {

    for (size_t p=0; p<pops.size(); p++) pops[p]->setinp(inp);

}


void PPopR::gatheract() {

    for (size_t p=0; p<pops.size(); p++) {

	for (size_t n=0; n<NperH; n++)

	    ppact[p*NperH + n] = pops[p]->getstate("act")[n];

    }
}


void PPopR::doputsrcact() {

    if (Shmem::maxnvec[SRCMEM]==0 or maxidelay<0) return;

    if (locppopr!=NULL and locppopr->issrcpop) {
	
	gatheract();

	for (size_t r=0; r<fwranks.size(); r++)

	    shmem_float_put(Shmem::getshmemarr(SRCMEM) + noffs,ppact.data(),NperR,fwranks[r]);

    }
}


void PPopR::gatheractnz() {

    vector<float> popact;

    int nz = 0;

    fill(ppactnz.begin(),ppactnz.end(),0);

    for (size_t p=0; p<pops.size(); p++) {

	popact = pops[p]->getstate("act");

	for (size_t n=0; n<NperH; n++) {

	    if (popact[n]>0) {

		if (isspiking)

		    ppactnz[1 + nz] = p*NperH + n;

		else {

		    ppactnz[1 + 2*nz] = p*NperH + n;
		
		    ppactnz[1 + 2*nz + 1] = popact[n];

		}

		nz++;

	    }
	}
    }
    
    ppactnz[0] = nz;

}


void PPopR::doputsrcactnz() {

    if (Shmem::maxnvec[SRCMEM]==0 or maxidelay<0) return;

    if (locppopr!=NULL and locppopr->issrcpop) {
	
	gatheractnz();

	for (size_t r=0; r<fwranks.size(); r++)

	    if (isspiking)

		shmem_float_put(Shmem::getshmemarr(SRCMEMNZ) + noffs + locrank,ppactnz.data(),NperR + 1,
				fwranks[r]);
	    else

		shmem_float_put(Shmem::getshmemarr(SRCMEMNZ) + 2*noffs + locrank,ppactnz.data(),2*NperR + 1,
				fwranks[r]);
    }
}


void PPopR::doupdpaxons() {

    if (maxidelay<=0) return;

    gatheract();

    paxons->update(ppact);

}


void PPopR::doputtrgact() {

    if (Shmem::maxnvec[TRGMEM]==0) return;

    if (locppopr!=NULL and locppopr->istrgpop) {

	gatheract();

	for (size_t r=0; r<bwranks.size(); r++) {

	    shmem_float_put(Shmem::getshmemarr(TRGMEM),ppact.data(),NperR,bwranks[r]);

	}
    }
}


void PPopR::dosumbwcond() {

    if (Shmem::maxnvec[BWCMEM]==0) return;

    if (locppopr==NULL) return;

    for (size_t p=0,u=0; p<pops.size(); p++) {

     	fill(pops[p]->bwsup.begin(),pops[p]->bwsup.end(),0);

    	for (int ni=0; ni<locppopr->ninprj; ni++) {

    	    for (int n=0; n<NperH; n++)

    		pops[p]->bwsup[n] += Shmem::getshmemarr(BWCMEM)[ni*N + locrank*NperR + p*NperH + n];

    	}

	for (size_t n=0; n<NperH; n++) pops[p]->bwsup[n] *= pops[p]->bwgain;

    }

    if (Esyn>-1) {

    	for (size_t p=0,u=0; p<pops.size(); p++) {

     	    for (size_t n=0; n<NperH; n++,u++)

     		pops[p]->bwsup[n] *= (Esyn - pops[p]->dsup[n]);
     	}
    }
}


void PPopR::fwritestate(PIO *pio,string statestr) {

    vector<float> statevec;

    for (size_t p=0; p<pops.size(); p++) {

	if (statestr=="dsupmax" or statestr=="expdsupsum") {

	    statevec = vector<float>(1);

	    statevec[0] = ((ExpPop *)pops[p])->getstate1(statestr);

	    pio->pfwritestatevec(pops[p]->hoffs,statevec);

	} else {

	    switch (pops[p]->actfn_t) {

	    case BCP:
	    case EXP: statevec = ((ExpPop *)pops[p])->getstate(statestr); break;

	    default: statevec = pops[p]->getstate(statestr);

	    }

	    pio->pfwritestatevec(pops[p]->noffs,statevec);

	}
    }
}


int PPopulation::ppopul_n = 0,PPopulation::ppopul_npop = 0;

PPopulation::PPopulation(int nrank,int H,int M,int U,Actfn_t actfn_t,Normfn_t normfn_t,float maxfq)
    : PPobjRH(nrank) {

    if (H%nrank!=0) perror("PPopulation::PPopulation","Illegal: H%nrank!=0: H = " + to_string(H) +
			   "nrank = " + to_string(nrank));

    this->H = H; this->M = M; this->U = U;

    N = H * M * U;

    HperR = H/nrank;

    NperH = M * U;

    isspiking = not (actfn_t==BCP or actfn_t==LIN or actfn_t==LOG or actfn_t==EXP);

    this->maxfq = maxfq;

    maxdelay = 0;

    ninprj = 0;

    id = ppopul_n;

    PPopR *ppopr;

    for (int r=0; r<nrank; r++) {

    	if (shrank==rank0 + r) {

	    ppopr = new PPopR(ppopul_npop,rank0,nrank,r * HperR,HperR,M,1,actfn_t,normfn_t);

	    ppopr->poph->id = id;

	}

	ppopul_npop += HperR;

    }

    ppopul_n++;

    raisebarrier();

}


PPopulation::PPopulation(int nrank,int H,int M,Actfn_t actfn_t,Normfn_t normfn_t,float maxfq)
    : PPobjRH(nrank) {

    if (H%nrank!=0) perror("PPopulation::PPopulation","Illegal: H%nrank!=0: H = " + to_string(H) +
			   "nrank = " + to_string(nrank));

    this->H = H; this->M = M; this->U = 1;

    N = H * M * U;

    HperR = H/nrank;

    NperH = M * U;

    isspiking = not (actfn_t==BCP or actfn_t==LIN or actfn_t==LOG or actfn_t==EXP);

    this->maxfq = maxfq;

    maxdelay = 0;

    ninprj = 0;

    id = ppopul_n;

    PPopR *ppopr;

    for (int r=0; r<nrank; r++) {

    	if (shrank==rank0 + r) {

    	    ppopr = new PPopR(ppopul_npop,rank0,nrank,r * HperR,HperR,M,1,actfn_t,normfn_t);

	    ppopr->poph->id = id;

	}

	ppopul_npop += HperR;

    }
    
    ppopul_n++;

    raisebarrier();

}


void PPopulation::setparam(Popparam_t poppar_t,float parval) {

    if (not onthisrank()) return;

    if (locppopr==NULL) return;

    locppopr->setparam(poppar_t,parval);

}


void PPopulation::setseed(long newseed) {

    if (not onthisrank()) return;

    if (locppopr==NULL) return;

    locppopr->setseed(newseed);

}


void PPopulation::setmaxdelay(float maxdelay) {

    this->maxdelay = maxdelay;

    if (onthisrank()) locppopr->setmaxdelay(maxdelay);
    
}


void PPopulation::resetstate() {

    if (onthisrank()) locppopr->resetstate();

}


void PPopulation::setinp(float inpval) {

    if (onthisrank()) locppopr->setinp(inpval);

}


void PPopulation::setinp(vector<float> inpvec) {

    if (onthisrank()) locppopr->setinp(inpvec);

}


void PPopulation::prnidandseed() {

    if (not onthisrank()) return;

    for (size_t p=0; p<locppopr->pops.size(); p++) {

	printf("%4d %4d %9ld\n",shrank,locppopr->pops[p]->getinfo("id"),locppopr->pops[p]->getseed());

	fflush(stdout);

    }
}


void PPopulation::fwritestate(PIO *pio,string statestr) {

    if (pio==NULL) perror("PPopulation::fwritestate","Illegal pio==NULL");

    if (onthisrank()) locppopr->fwritestate(pio,statestr);

}


void PPopulation::fwritestate(string statestr,string filename) {

    PIO *pio;

    if (pio==NULL) pio = new PIO(N);

    pio->pfopen(filename);

    if (onthisrank()) locppopr->fwritestate(pio,statestr);

    pio->pfclose();

}
