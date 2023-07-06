
#include <cstring>
#include <assert.h>

#include "pbcpnnsim.h"

#include "PPobjR.h"
#include "PPopR.h"
#include "PPrj.h"
#include "PPrjH.h"
#include "PPrjR.h"

#include <set>

#define SHMEM_GENERIC_32

using namespace std;
using namespace Globals;
using namespace PGlobals;

int *PPrjR::gntpatch = NULL;
long PPrjR::gtpatchsync[SHMEM_REDUCE_SYNC_SIZE];
int *PPrjR::gtpatchpwrk;
int PPrjR::maxidelay = 0;

PPrjR::PPrjR(int nprj0,int rank0,int nrank,
	     int srcpopnrank,int srcH,int srcM,int srcU,float srcspkwgain,
	     int trgH,int trgM,int trgU,int trghoffs,int trgHperR,float trgspkwgain,
	     float pdens,BCPvar_t bcpvar,Prjarch_t prjarch)
    : PPobjR(rank0,nrank) {

    srcN = srcH * srcM * srcU;

    srcNperR = srcN/srcpopnrank;

    srcNperH = srcM * srcU;

    this->trgH = trgH;

    trgN = trgH * trgM * trgU;

    this->trgNperH = trgM * trgU;

    this->trgHperR = trgHperR;

    trgNperR = trgHperR * trgM * trgU;
    
    ppbwcond = vector<float>(trgNperR,0);

    this->trghoffs = trghoffs;

    locpprjr = this;

    pprjh = new PPrjH(nprj0,srcH,srcM,srcU,srcspkwgain,
    		      trgH,trgHperR,trgM,trgU,trghoffs,trgspkwgain,
    		      pdens,bcpvar,prjarch);

}


void PPrjR::setparam(Prjparam_t prjpar_t,float parval) {

    if (locpprjr==NULL) return;

    for (size_t p=0; p<pprjh->prjs.size(); p++) {

	switch (prjpar_t) {
	case PRN:
	    pprjh->prjs[p]->setparam("prn",parval);
	    break;
	case WDENS:
	    pprjh->prjs[p]->setparam("wdens",parval);
	    break;
	case TAUP:
	    pprjh->prjs[p]->setparam("taup",parval);
	    break;
	case TAUB:
	    pprjh->prjs[p]->setparam("taub",parval);
	    break;
	case PSILENT:
	    pprjh->prjs[p]->setparam("psilent",parval);
	    break;
	case FLPSCR:
	    pprjh->prjs[p]->setparam("flpscr",parval);
	    break;
	case PSPA:
	    pprjh->prjs[p]->setparam("pspa",parval);
	    break;
	case PSPC:
	    pprjh->prjs[p]->setparam("pspc",parval);
	    break;
	case MINAGE:
	    pprjh->prjs[p]->setparam("minage",parval);
	    break;
	case BGAIN:
	    pprjh->prjs[p]->setparam("bgain",parval);
	    break;
	case KBJHALF:
	    pprjh->prjs[p]->setparam("kbjhalf",parval);
	    break;
	case TAUZI:
	    pprjh->prjs[p]->setparam("tauzi",parval);
	    break;
	case TAUZJ:
	    pprjh->prjs[p]->setparam("tauzj",parval);
	    break;
	case WIJ:
	    pprjh->prjs[p]->setstate("Wij",parval);
	    break;
	case EWGAIN:
	    pprjh->prjs[p]->setparam("ewgain",parval);
	    break;
	case IWGAIN:
	    pprjh->prjs[p]->setparam("iwgain",parval);
	    break;
	case WGAIN:
	    setparam(EWGAIN,parval); setparam(IWGAIN,parval);
	    break;
	default: perror("PPrjR::setparam","No such prjpar_t " + to_string(prjpar_t));

	}
    }
}


void PPrjR::setdelays(vector<vector<float> > delaymat) {

    if (locpprjr==NULL) return;

    pprjh->setdelays(delaymat);

    idelay = vector<vector<int> >(srcN,vector<int>(trgHperR,0));

    for (int srcn=0; srcn<srcN; srcn++) {

    	for (int trgh=0; trgh<trgHperR; trgh++) {

    	    if (delaymat[srcn/srcNperH][trghoffs+trgh]<=0) continue;

    	    idelay[srcn][trgh] = (int)(delaymat[srcn/srcNperH][trghoffs+trgh]/timestep + 0.5);

    	}
    }
}


void PPrjR::setWij(vector<vector<float> > Wij) {

    pprjh->setstate("Wij",Wij);

}


void PPrjR::doscattersrcact() {

    if (locpprjr==NULL or maxidelay>0) return;

    prnval(Shmem::getshmemarr(SRCMEM),srcN);

    for (int trgh=0; trgh<trgHperR; trgh++)

    	for (int srcn=0; srcn<srcN; srcn++)

	    pprjh->prjs[trgh]->delact[srcn] = Shmem::getshmemarr(SRCMEM)[srcn];

}


void PPrjR::doscattersrcactnz() {

    if (locpprjr==NULL or maxidelay>0) return;

    int nz,srcnrank = srcN/srcNperR; float srcn,act; Prj *prj;

    if (spikinginput) {

	for (int trgh=0,srcrank; trgh<trgHperR; trgh++) {

	    prj = pprjh->prjs[trgh];

	    fill(prj->delact.begin(),prj->delact.end(),0);

	    for (int srcrank=0,noffs,srcranknoffs; srcrank<srcnrank; srcrank++) {

		srcranknoffs = srcrank * (srcNperR + 1);

		noffs = srcrank * srcNperR;

		nz = Shmem::getshmemarr(SRCMEMNZ)[srcranknoffs];

		for (int n=0; n<nz; n++) {

		    srcn = Shmem::getshmemarr(SRCMEMNZ)[srcranknoffs + 1 + n];

		    prj->delact[noffs + srcn] = 1;

		}
	    }
	}

    } else {

	for (int trgh=0,srcrank; trgh<trgHperR; trgh++) {

	    prj = pprjh->prjs[trgh];

	    fill(prj->delact.begin(),prj->delact.end(),0);

	    for (int srcrank=0,noffs,srcranknoffs; srcrank<srcnrank; srcrank++) {

		srcranknoffs = srcrank * (2*srcNperR + 1);

		noffs = srcrank * srcNperR;

		nz = Shmem::getshmemarr(SRCMEMNZ)[srcranknoffs];

		for (int n=0; n<nz; n++) {

		    srcn = Shmem::getshmemarr(SRCMEMNZ)[srcranknoffs + 1 + 2*n];

		    act = Shmem::getshmemarr(SRCMEMNZ)[srcranknoffs + 2 + 2*n];

		    prj->delact[noffs + srcn] = act;

		}
	    }
	}
    }
}


void PPrjR::doscattertrgact() {

    if (locpprjr==NULL) return;

    for (int trgh=0; trgh<trgHperR; trgh++)

    	for (int trgn=0; trgn<trgNperH; trgn++)

    	    pprjh->prjs[trgh]->trgact[trgn] = Shmem::getshmemarr(TRGMEM)[trgh*trgNperH + trgn];

}


void PPrjR::dogetdelsrcact() {
 
    if (locpprjr==NULL or maxidelay<=0) return;

    if (locpprjr!=NULL) {
	
	int delidx; float getdata;

	for (int srcrank=srcrank0,locsrcrank; srcrank<srcrankn; srcrank++) {

	    locsrcrank = srcrank - srcrank0;

	    for (int srcn=0; srcn<srcNperR; srcn++) {

		for (int trgh=0; trgh<trgHperR; trgh++) {

     		    if (pprjh->prjs[trgh]->connijs[srcn]==NULL) continue;

		    if (idelay[(locsrcrank*srcNperR+srcn)/srcNperH][trgh]<=0) continue;
		    
		    delidx = idelay[(locsrcrank*srcNperR+srcn)/srcNperH][trgh];

		    shmem_float_get(&getdata,PPobjR::shaxons[srcn] + delidx,1,srcrank);

		    pprjh->prjs[trgh]->delact[locsrcrank*srcNperR+srcn] = getdata;

		}
	    }
	}	
    }
}


void PPrjR::gatherbwcond() {

    fill(ppbwcond.begin(),ppbwcond.end(),0);

    for (int trgh=0,trghn=0; trgh<trgHperR; trgh++)

    	for (int trgn=0; trgn<trgNperH; trgn++)

    	    ppbwcond[trghn++] += pprjh->prjs[trgh]->cond[trgn];

}


void PPrjR::doputbwcond() {

    if (Shmem::maxnvec[BWCMEM]==0) return;

    if (locpprjr!=NULL) {
	
	int noffs = (locpprjr->ninprj - 1) * trgN + locrank * trgNperR;

	shmem_float_put(Shmem::getshmemarr(BWCMEM) + noffs,ppbwcond.data(),trgNperR,bwctrgrank);

    }
}


void PPrjR::updgntpatch() {

    pprjh->fetchntpatch();

    shmem_int_sum_to_all(gntpatch,pprjh->ntpatch.data(),srcN,rank0,0,nrank,
			 gtpatchpwrk,gtpatchsync);

    pprjh->gntpatch.assign(gntpatch,gntpatch + srcN);

}


void PPrjR::setstrcpl(bool on) {

    pprjh->setstrcpl(on);


}


void PPrjR::fliptpatchn(int nflp,int flpupdint) {

    pprjh->fliptpatchn(nflp,flpupdint);

}


void PPrjR::fwritestate(PIO *pio,string statestr) {

    if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="Mic" or statestr=="Sil" or
	statestr=="Age" or statestr=="delact") {

	vector<vector<float> > xij(srcN,vector<float>(trgHperR,0));

	for (size_t p=0; p<trgHperR; p++) {

	    vector<float> xi = pprjh->prjs[p]->getstatei(statestr);

	    for (int srcn=0; srcn<srcN; srcn++)

		xij[srcn][p] = xi[srcn];

	}

	for (size_t p=0; p<trgHperR; p++) {

	    for (int srcn=0; srcn<srcN; srcn++) {

		pio->pfwritestatevec(srcn * trgH + locrank * trgHperR,xij[srcn]);

	    }
	}

    } else for (size_t p=0; p<pprjh->prjs.size(); p++) {

	if (statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="Bj" or
		   statestr=="bwsup" or statestr=="cond") {

	    pio->pfwritestatevec(pprjh->prjs[p]->trgnoffs,pprjh->prjs[p]->getstatej(statestr));

	} else if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won") {

	    vector<vector<float> > xij = pprjh->prjs[p]->getstateij(statestr);

	    int trgnoffs = pprjh->gettrgnoffs(p);

	    for (int srcn=0; srcn<srcN; srcn++) {

		pio->pfwritestatevec(srcn * trgN + trgnoffs,xij[srcn]);

	    }
	}
    }
}


int PProjection::pproje_n = 0,PProjection::pproje_nprj;

PProjection::PProjection(PPopulation *srcppopulation,PPopulation *trgppopulation,float pdens,
			 BCPvar_t bcpvar,Prjarch_t prjarch)
    : PPobjRH(trgppopulation->nrank) {

    this->srcppopulation = srcppopulation; this->trgppopulation = trgppopulation;

    srcN = srcppopulation->N; srcH = srcppopulation->H; trgH = trgppopulation->H; trgN = trgppopulation->N;

    srcR = 1;

    int srcH = srcppopulation->H;

    trgR = trgppopulation->nrank;

    int trgHperR = trgppopulation->HperR;

    trgNperR = trgHperR * trgppopulation->NperH;

    int srcM = srcppopulation->M,srcU = srcppopulation->U,
    	trgM = trgppopulation->M,trgU = trgppopulation->U;

    /// *spkwgain needed? Does not update if maxfq is changed for src- or trgppopulation.
    float srcspkwgain = 1/(srcppopulation->maxfq*timestep),
    	trgspkwgain = 1/(trgppopulation->maxfq*timestep);

    srcspkwgain = trgspkwgain = 1;

    if (srcppopulation->onthisrank()) {

	locppopr = (PPopR *)locppobjr;

	locppopr->issrcpop = true;

    }

    trgppopulation->ninprj++;

    // if (srcppopulation->isspiking)  // Not yet used, using SCRMEMNZ

    //  	Shmem::reqshmem(SRCMEMSPK,srcN + srcppopulation->nrank);

    // else {

	Shmem::reqshmem(SRCMEM,srcN);
	Shmem::reqshmem(SRCMEMNZ,2*srcN + srcppopulation->nrank);

	// }

    Shmem::reqshmem(BWCMEM,trgppopulation->ninprj * trgN);
    Shmem::reqshmem(TRGMEM,trgNperR);

    if (trgppopulation->onthisrank())

    	{ locppobjr = (PPopR *)locppobjr; locppopr->istrgpop = true; locppopr->ninprj++;}

    id = pproje_n;

    mkgntpatch();

    for (int trgr=0; trgr<trgR; trgr++) {

    	if (shrank==rank0 + trgr) {

    	    locpprjr = new PPrjR(pproje_nprj,rank0,trgR,srcppopulation->nrank,srcH,srcM,srcU,srcspkwgain,
	    			 trgH,trgM,trgU,trgr * trgHperR,trgHperR,trgspkwgain,pdens,bcpvar,prjarch);

	    locpprjr->srcrank0 = srcppopulation->rank0;

	    locpprjr->srcrankn = srcppopulation->rank0 + srcppopulation->nrank;

	    locpprjr->spikinginput = srcppopulation->isspiking;

	}

	pproje_nprj += trgHperR;

    }

    if (onthisrank()) {

    	locpprjr->ninprj = trgppopulation->ninprj;

    	locpprjr->bwctrgrank = trgppopulation->rank0 + getlocrank();

    }

    mktrgranks();

    pproje_n++;

    raisebarrier();

    if (srcppopulation->onthisrank()) locppopr->reallocaxons(0);

}


void PProjection::prnidandseed() {

    if (not onthisrank()) return;

    locpprjr->pprjh->prnidandseed();

}


void PProjection::mktrgranks() {

    if (onthisrank()) {

	for (int trgr=0; trgr<trgR; trgr++)

	    locppobjr->fwranks.push_back(rank0 + trgr);

    }

    if (srcppopulation->onthisrank()) {

	for (int trgr=0; trgr<trgR; trgr++)

	    locppobjr->fwranks.push_back(rank0 + trgr);

    }

    if (trgppopulation->onthisrank()) {

	int trgpoprank = shrank - trgppopulation->rank0;

	int trgprjrank = rank0 + trgpoprank;

	locppobjr->bwranks.push_back(trgprjrank);

    }
}


void PProjection::mkgntpatch() {

    int pwrk_size = max(srcN/2 + 1,_SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    PPrjR::gtpatchpwrk = (int *)shmem_calloc(pwrk_size,sizeof(int *));
    assert(PPrjR::gtpatchpwrk != NULL);

    PPrjR::gntpatch = (int *)shmem_calloc(srcN,sizeof(int));

    for (int i=0; i<_SHMEM_REDUCE_SYNC_SIZE; i++) PPrjR::gtpatchsync[i] = SHMEM_SYNC_VALUE;

}


void PProjection::setstrcpl(bool on) {

    if (not onthisrank()) return;

    if (locpprjr==NULL) return;

    locpprjr->setstrcpl(on);

}


void PProjection::fliptpatchn(int nflp,int flpupdint) {

    if (not onthisrank()) return;

    if (locpprjr==NULL) return;

    locpprjr->fliptpatchn(nflp,flpupdint);

}

void PProjection::setparam(Prjparam_t prjpar_t,float parval) {

    if (not onthisrank()) return;

    if (locpprjr==NULL) return;

    locpprjr->setparam(prjpar_t,parval);

}


void PProjection::setdelays(vector<vector<float> > delaymat) {

    if (delaymat.size()!=srcppopulation->H) perror("PProjection::setdelays","Illegal row-dim: " +
						   to_string((int)delaymat.size()));
    else if (delaymat[0].size()!=trgppopulation->H) perror("PProjection::setdelays","Illegal col-dim: " +
							   to_string((int)delaymat[0].size()));
    float maxdelay = 0;

    for (size_t row=0; row<delaymat.size(); row++)

    	for (size_t col=0; col<delaymat[0].size(); col++)

    	    if (delaymat[row][col]>maxdelay) maxdelay = delaymat[row][col];

    if (maxdelay>srcppopulation->maxdelay) locppopr->reallocaxons(maxdelay);

    if (onthisrank()) {


     	locpprjr->setdelays(delaymat);

	locpprjr->maxidelay = maxdelay/timestep + 1;

    } else if (srcppopulation->onthisrank()) {

     	srcppopulation->setmaxdelay(maxdelay);

    }
}


void PProjection::setWij(vector<vector<float> > Wij) {

    if (Wij.size()!=srcppopulation->H) perror("PProjection::setWij","Illegal row-dim: " +
					      to_string((int)Wij.size()));
    else if (Wij[0].size()!=trgppopulation->H) perror("PProjection::setWij","Illegal col-dim: " +
						      to_string((int)Wij[0].size()));

    if (onthisrank()) locpprjr->setWij(Wij);

}


void PProjection::prntpatch(int rank) {
    
    if (rank<0) rank = rank0;

    if (shrank==rank and locpprjr!=NULL) {

	int csum = 0;

	for (size_t i=0; i<locpprjr->pprjh->ntpatch.size(); i++) csum += locpprjr->gntpatch[i];

	printf("shrank = %4d csum  = %3d ",shrank,csum); fflush(stdout);

	prnvec(locpprjr->pprjh->ntpatch);

    }
}


void PProjection::fwritestate(PIO *pio,string statestr) {

    if (pio==NULL) perror("PProjection::fwritestate","Illegal pio==NULL");

    if (onthisrank()) locpprjr->fwritestate(pio,statestr);

}


void PProjection::fwritestate(string statestr,string filename) {

    PIO *pio;

    if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="Mic" or statestr=="Sil" or
	statestr=="Age" or statestr=="delact")

	pio = new PIO(srcN * trgH);

    else if (statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="Bj" or statestr=="bwsup" or
	     statestr=="cond" )

	pio = new PIO(trgH * trgN);

    else if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won" or
	     statestr=="Wijsil" ) {

	pio = new PIO(srcN * trgN);

    }

    pio->pfopen(filename);

    if (onthisrank()) locpprjr->fwritestate(pio,statestr);

    pio->pfclose();

}


