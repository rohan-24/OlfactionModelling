#include <stdio.h>
#include <shmem.h>
#include <shmemx.h>
#include <assert.h>

#include "bcpnnsim.h"
#include "pbcpnnsim.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;

#undef isroot

#define isroot() (shrank==0)

int Rin = 2,Hin = 1,Min = 1,Rhid = 2,Hhid = 1,Mhid = 1,nstep = 1,ngap = 1,nepoc = 1,
    nutrpat = 10,nstrpat = 10,ntetrpat = nstrpat,ntepat = -1,flpscr = 1,verbosity = 3,loglevel = 1;
float inhidpdens = 1,inhidwdens = 1,hidagain = 1,inhidtaup = 1,hidcltaup = 1,hidbgain = -1,
    hidnmean = 0,hidnamp = 0,maxfq = 100,inhidpsilent = 0,pspa = 0,pspc = 0;
bool dolog = true;
string inactfn = "LIN",hidactfn = "BCP";

PPopulation *inpop,*hidpop,*clpop;
PProjection *inhidprj,*hidclprj;
Actfn_t inactfn_t = LIN,hidactfn_t = BCP;

PLogger *troutlog = NULL,*teoutlog = NULL,*trteoutlog = NULL,*trhidlog = NULL,*tehidlog = NULL;

void parseparams(std::string paramfile) {

    Parseparam *parseparam = new Parseparam(paramfile);

    parseparam->postparam("seed",&gseed,Long);

    parseparam->postparam("Rin",&Rin,Int);
    parseparam->postparam("Hin",&Hin,Int);
    parseparam->postparam("Min",&Min,Int);
    parseparam->postparam("Rhid",&Rhid,Int);
    parseparam->postparam("Hhid",&Hhid,Int);
    parseparam->postparam("Mhid",&Mhid,Int);

    parseparam->postparam("inactfn",&inactfn,String);
    parseparam->postparam("hidactfn",&hidactfn,String);

    parseparam->postparam("maxfq",&maxfq,Float);

    parseparam->postparam("hidagain",&hidagain,Float);
    parseparam->postparam("maxfq",&maxfq,Float);

    parseparam->postparam("inhidpdens",&inhidpdens,Float);
    parseparam->postparam("inhidwdens",&inhidwdens,Float);
    
    parseparam->postparam("inhidtaup",&inhidtaup,Float);
    parseparam->postparam("hidcltaup",&hidcltaup,Float);

    parseparam->postparam("hidbgain",&hidbgain,Float);
    parseparam->postparam("hidnmean",&hidnmean,Float);
    parseparam->postparam("hidnamp",&hidnamp,Float);
    
    parseparam->postparam("inhidpsilent",&inhidpsilent,Float);
    parseparam->postparam("flpscr",&flpscr,Int);
    parseparam->postparam("pspa",&pspa,Float);
    parseparam->postparam("pspc",&pspc,Float);

    parseparam->postparam("nepoc",&nepoc,Int);
    parseparam->postparam("nutrpat",&nutrpat,Int);
    parseparam->postparam("nstrpat",&nstrpat,Int);
    parseparam->postparam("ntetrpat",&ntetrpat,Int);
    parseparam->postparam("ntepat",&ntepat,Int);
    parseparam->postparam("nstep",&nstep,Int);
    parseparam->postparam("ngap",&ngap,Int);

    parseparam->postparam("verbosity",&verbosity,Int);
    parseparam->postparam("loglevel",&loglevel,Int);

    parseparam->doparse();

    if (ntepat<0) ntepat = nstrpat;

}

void loadMNISTdata() {

    int Min2 = Min;

    if (Min==1) Min2 *= 2;

    /* Reduced MNIST */

    string datadir = "../../../MNIST_Data/";

    trdata = readpats(nutrpat,Hin*Min2,datadir + "mnist_1k_xtrdata3c.bin"); // Reduced from 28x28 to 22x22

    tedata = readpats(ntepat,Hin*Min2,datadir + "mnist_1k_xtedata3c.bin");

    trlbl = readpats(nutrpat,10,datadir + "mnist_1k_trlbl.bin");

    telbl = readpats(ntepat,10,datadir + "mnist_1k_telbl.bin");

    if (verbosity>5 and isroot()) printf("trdata.size() = %zu trdata.wid = %zu\n",
					 trdata.size(),trdata[0].size());

    if (Hin!=484) perror("Shmemmnistmain1::loadMNISTdata","Not using reduced MNIST");

    /* Full MNIST */

    // string datadir = "../../../MNIST_Data/MNISTorig/";

    // trdata = readpats(nutrpat,Hin*Min2,datadir + "mnist_60k_traindata.bin");

    // tedata = readpats(ntepat,Hin*Min2,datadir + "mnist_10k_testdata.bin");

    // trlbl = readpats(nutrpat,10,datadir + "mnist_60k_trainlbl.bin");

    // telbl = readpats(ntepat,10,datadir + "mnist_10k_testlbl.bin");

    // if (verbosity>5 and isroot()) printf("trdata.size() = %u trdata.wid = %u\n",
    //                                       trdata.size(),trdata[0].size());

    // if (Hin!=784) perror("Shmemmnistmain1::loadMNISTdata","Not using full MNIST");

    if (Min==1) {

	vector<vector<float> > trdata1(nutrpat,vector<float>(Hin*Min,0));

	vector<vector<float> > tedata1(ntepat,vector<float>(Hin*Min,0));

	for (int p=0; p<nutrpat; p++)

	    for (int i=0; i<Hin; i++)

		trdata1[p][i] = trdata[p][2*i+1];

	trdata = trdata1;

	for (int p=0; p<ntepat; p++)

	    for (int i=0; i<Hin; i++)

		tedata1[p][i] = tedata[p][2*i+1];

	tedata = tedata1;

	// prnmat(tedata,1);
	
    }

}


void setuplogging() {

    if (loglevel==0) return;

    // new PLogger(inpop,"inp","inp.log");

    // new PLogger(inpop,"act","inact.log");

    // new PLogger(hidpop,"dsup","hiddsup.log");

    // new PLogger(hidpop,"act","hidact.log");

    // new PLogger(clpop,"inp","clinp.log");

    // new PLogger(clpop,"dsup","cldsup.log");

    // new PLogger(inhidprj,"Zi","inhidzi.log");

    // new PLogger(inhidprj,"Zj","inhidzj.log");

    // new PLogger(inhidprj,"bwsup","inhidbwsup.log");

    // new PLogger(inhidprj,"Wij","inhidwij.log",100);

    if (loglevel>0) {

	troutlog = new PLogger(clpop,"act","trout.log");

	trteoutlog = new PLogger(clpop,"act","trteout.log");

	teoutlog = new PLogger(clpop,"act","teout.log");

    }

    if (loglevel>1) {

	trhidlog = new PLogger(hidpop,"act","trhid.log");

	tehidlog = new PLogger(hidpop,"act","tehid.log");

    }

    if (loglevel>2) {

	new PLogger(inhidprj,"Pj","pj_ih.log",nutrpat);

	new PLogger(inhidprj,"Bj","bj_ih.log",nutrpat);

    }

    if (troutlog!=NULL) troutlog->stop(); if (teoutlog!=NULL) teoutlog->stop();
    if (trteoutlog!=NULL) trteoutlog->stop();
    if (trhidlog!=NULL) trhidlog->stop(); if (tehidlog!=NULL) tehidlog->stop();

}


int main(int argc,char **args) {

    string paramfile = "shmemmnistmain1.par";
    
    if (argc>1) paramfile = args[1];

    parseparams(paramfile);

    pginitialize(argc,args);

    int minpes = Rin + Rhid + 1 + Rhid + 1;

    if (isroot()) {

    	if (shsize<minpes) {

    	    fprintf(stderr,"shmemmnistmain1::main: minpes -- shsize mismatch (minpes = %d)\n",minpes);

    	    shmem_global_exit(9);

    	}
    }

    gsetseed(gseed);

    if (inactfn=="SPKLIN") inactfn_t = SPKLIN;

    if (hidactfn=="SPKBCP") hidactfn_t = SPKBCP;

    if (Min==1) {

    	if (inactfn_t==SPKLIN)

    	    inpop = new PPopulation(Rin,Hin,Min,SPKLIN,CAP);

    	else

    	    inpop = new PPopulation(Rin,Hin,Min,LIN,CAP);

    } else {
    
    	    inpop = new PPopulation(Rin,Hin,Min,inactfn_t);

    }

    // inpop->prnidandseed();    

    if (verbosity>3 and isroot()) printf("Hhid = %d Mhid = %d Nhid = %d\n",Hhid,Mhid,Hhid*Mhid);

    hidpop = new PPopulation(Rhid,Hhid,Mhid,hidactfn_t);
    
    // hidpop->prnidandseed();    

    clpop = new PPopulation(1,1,10);

    // clpop->prnidandseed();    

    inhidprj = new PProjection(inpop,hidpop,inhidpdens,INCR,HPATCHY);

    inhidprj->setparam(WDENS,inhidwdens);

    // inhidprj->prnidandseed();    

    hidclprj = new PProjection(hidpop,clpop,1,INCR);

    // inhidprj->setparam(TAUZI,0.003);

    // inhidprj->setparam(TAUZJ,0.003);

    pgalloc();

    loadMNISTdata();

    raisebarrier();

    if (dolog) setuplogging();

    if (inactfn_t==SPKLIN) inpop->setparam(MAXFQ,maxfq);

    if (hidactfn_t==SPKBCP) hidpop->setparam(MAXFQ,maxfq);

    hidpop->setparam(AGAIN,hidagain);

    inpop->setparam(NMEAN,0); inpop->setparam(NAMP,0);

    hidpop->setparam(NMEAN,hidnmean); hidpop->setparam(NAMP,hidnamp);

    clpop->setparam(NMEAN,0); clpop->setparam(NAMP,0);

    clpop->setparam(BWGAIN,0);

    inhidprj->setparam(TAUP,inhidtaup);

    inhidprj->setparam(BGAIN,hidbgain);

    inhidprj->setparam(PSILENT,inhidpsilent);

    inhidprj->setparam(FLPSCR,flpscr);

    inhidprj->setparam(PSPA,pspa);

    inhidprj->setparam(PSPC,pspc);

    hidclprj->setparam(TAUP,hidcltaup);

    raisebarrier();

    Timer *alltimer = new Timer("Time elapsed");

    if (verbosity>1 and isroot()) { printf("Unsupervised training w/o structural plasticity\n");
	fflush(stdout); }

    for (int epoc=0; epoc<nepoc; epoc++) {

	if (verbosity>2 and isroot()) { printf("\nepoc = %d: ",epoc); fflush(stdout); }

	for (int p=0; not pexitflg and p<nutrpat; p++ ) {

	    if (verbosity>3 and isroot()) { printf("c"); fflush(stdout); }

	    inpop->setparam(IGAIN,1); clpop->setparam(IGAIN,1);

	    inpop->setinp(trdata[p]); clpop->setinp(trlbl[p]);

	    inhidprj->setparam(PRN,1);

	    psimulate(nstep);

	    inpop->setparam(IGAIN,0); clpop->setparam(IGAIN,0);

	    inhidprj->setparam(PRN,0);

	    psimulate(ngap);

	}
    }
    
    if (isroot()) printf("\n");

    if (nstrpat>0) {

	if (verbosity>1 and isroot()) { printf("\nSupervised traning (%d)\n",simstep); fflush(stdout); }

	inhidprj->setparam(PRN,0);

	for (int p=0; not pexitflg and p<nstrpat; p++ ) {

	    if (verbosity>3 and isroot()) { printf("s"); fflush(stdout); }

	    inpop->setparam(IGAIN,1); clpop->setparam(IGAIN,1);

	    inpop->setinp(trdata[p]); clpop->setinp(trlbl[p]);

	    hidclprj->setparam(PRN,1);

	    if (troutlog!=NULL) troutlog->start();

	    if (trhidlog!=NULL) trhidlog->start();

	    psimulate(nstep);

	    if (troutlog!=NULL) troutlog->stop();

	    if (trhidlog!=NULL) trhidlog->stop();

	    inpop->setparam(IGAIN,0); clpop->setparam(IGAIN,0);

	    hidclprj->setparam(PRN,0);

	    psimulate(ngap);

	}
    }

    inhidprj->setparam(PSPA,0); inhidprj->setparam(PSPC,0);

    if (ntetrpat>0) {

	if (ntetrpat>nstrpat) perror("shmemmnistmain1::main","Illegal: ntetrpat>nstrpat");

	if (verbosity>2 and isroot()) {

	    printf("\nTesting on training patterns(%d)\n",simstep);

	    fflush(stdout);

	}

	clpop->setparam(IGAIN,0);

	clpop->setparam(BWGAIN,1);

	for (int p=0; not pexitflg and p<ntetrpat; p++ ) {

	    if (verbosity>3 and isroot()) { printf("t"); fflush(stdout); }

	    inpop->setparam(IGAIN,1);
	
	    inpop->setinp(trdata[p]);

	    if (trteoutlog!=NULL) trteoutlog->start();

	    psimulate(nstep);

	    if (trteoutlog!=NULL) trteoutlog->stop();

	    inpop->setparam(IGAIN,0);

	    psimulate(nstep);

	}
    }
    
    if (ntepat>0) {
    
	if (ntetrpat>nstrpat) perror("shmemmnistmain1::main","Illegal: ntepat>nstrpat");

	if (verbosity>2 and isroot()) { printf("\nTesting on test patterns(%d)\n",simstep); fflush(stdout); }

	clpop->setparam(IGAIN,0);

	clpop->setparam(BWGAIN,1);

	hidclprj->setparam(PRN,0);

	for (int p=0; not pexitflg and p<ntepat; p++ ) {

	    if (verbosity>3 and isroot()) { printf("t"); fflush(stdout); }

	    inpop->setparam(IGAIN,1);
	
	    inpop->setinp(tedata[p]);

	    if (teoutlog!=NULL) teoutlog->start();

	    if (tehidlog!=NULL) tehidlog->start();

	    psimulate(nstep);

	    if (teoutlog!=NULL) teoutlog->stop();

	    if (tehidlog!=NULL) tehidlog->stop();

	    inpop->setparam(IGAIN,0);

	    psimulate(nstep);

	}
    }
    
    if (pexitflg) goto finish;

    if (verbosity>1 and isroot()) alltimer->print();

    if (verbosity>2 and isroot()) printf("N:o simsteps = %d\n",simstep);

    inhidprj->fwritestate("Mic","inhidmic.bin");
    inhidprj->fwritestate("Sil","inhidsil.bin");
    inhidprj->fwritestate("Wij","inhidwij.bin");
    inhidprj->fwritestate("Won","inhidwon.bin");
    inhidprj->fwritestate("Bj","inhidbj.bin");
    hidclprj->fwritestate("Wij","hidclwij.bin");
    hidclprj->fwritestate("Bj","hidclbj.bin");

 finish:
    
    finalize(); return 0;

}
