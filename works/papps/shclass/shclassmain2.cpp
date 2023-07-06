#include <stdio.h>
#include <shmem.h>
#include <assert.h>

#include "pbcpnnsim.h"

#include "DataFactory.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;

#undef isroot

#define isroot() (shrank==0)

int Rin = 2,Hin = 1,Min = 1,Rhid = 2,Hhid = 1,Mhid = 1,Mout = 10,nstep = 1,ngap = 1,nepoc = 1,
    nutrpat = 10,nstrpat = 10,ntetrpat = nstrpat,ntepat = -1,flpscr = 1,flpupdint = 0,nflp = 0,
    verbosity = 3,logselect = 1,r0 = 0,nrow = -1,c0 = 0,ncol = -1;
long shseed = -1,hidseed = -1;
float inhidpdens = 1,inhidwdens = 1,hidtaum = timestep,hidagain = 1,inhidtaup = -1,inhidtaub = -1,
    hidcltaup = -1,hidbgain = -1,hidnmean = 0,hidnamp = 0,maxfq = 100,inhidpsilent = 0,pspa = 0,pspc = 0,
    kbjhalf = 0,kTp = 0.1,kTb = 0.1;
bool strcpl = true;
string inactfn = "LIN",hidactfn = "BCP",datasetstr = "mnist",datapath = "";

PPopulation *inpop,*hidpop,*clpop;
PProjection *inhidprj,*hidclprj;
Actfn_t inactfn_t = LIN,hidactfn_t = BCP;

PLogger *troutlog = NULL,*teoutlog = NULL,*trteoutlog = NULL,*trhidlog = NULL,*tehidlog = NULL;

void parseparams(std::string paramfile) {

    Parseparam *parseparam = new Parseparam(paramfile);

    parseparam->postparam("gseed",&gseed,Long);

    parseparam->postparam("Rin",&Rin,Int);
    parseparam->postparam("Hin",&Hin,Int);
    parseparam->postparam("Min",&Min,Int);
    parseparam->postparam("Rhid",&Rhid,Int);
    parseparam->postparam("Hhid",&Hhid,Int);
    parseparam->postparam("Mhid",&Mhid,Int);
    parseparam->postparam("Mout",&Mout,Int);

    parseparam->postparam("inactfn",&inactfn,String);
    parseparam->postparam("hidactfn",&hidactfn,String);

    parseparam->postparam("maxfq",&maxfq,Float);

    parseparam->postparam("hidtaum",&hidtaum,Float);
    parseparam->postparam("hidagain",&hidagain,Float);
    parseparam->postparam("maxfq",&maxfq,Float);

    parseparam->postparam("inhidpdens",&inhidpdens,Float);
    parseparam->postparam("inhidwdens",&inhidwdens,Float);
    
    parseparam->postparam("inhidtaup",&inhidtaup,Float);
    parseparam->postparam("hidcltaup",&hidcltaup,Float);
    parseparam->postparam("kTp",&kTp,Float);

    parseparam->postparam("kbjhalf",&kbjhalf,Float);
    parseparam->postparam("inhidtaub",&inhidtaub,Float);
    parseparam->postparam("kTb",&kTb,Float);

    parseparam->postparam("hidseed",&hidseed,Long);
    parseparam->postparam("hidbgain",&hidbgain,Float);
    parseparam->postparam("hidnmean",&hidnmean,Float);
    parseparam->postparam("hidnamp",&hidnamp,Float);
    
    parseparam->postparam("inhidpsilent",&inhidpsilent,Float);

    parseparam->postparam("strcpl",&strcpl,Boole);
    parseparam->postparam("flpscr",&flpscr,Int);
    parseparam->postparam("flpupdint",&flpupdint,Int);
    parseparam->postparam("nflp",&nflp,Int);
    parseparam->postparam("pspa",&pspa,Float);
    parseparam->postparam("pspc",&pspc,Float);

    parseparam->postparam("dataset",&datasetstr,String);
    parseparam->postparam("datapath",&datapath,String);

    parseparam->postparam("r0",&r0,Int);
    parseparam->postparam("c0",&c0,Int);
    parseparam->postparam("nrow",&nrow,Int);
    parseparam->postparam("ncol",&ncol,Int);

    parseparam->postparam("nepoc",&nepoc,Int);
    parseparam->postparam("nutrpat",&nutrpat,Int);
    parseparam->postparam("nstrpat",&nstrpat,Int);
    parseparam->postparam("ntetrpat",&ntetrpat,Int);
    parseparam->postparam("ntepat",&ntepat,Int);

    parseparam->postparam("shseed",&shseed,Long);

    parseparam->postparam("nstep",&nstep,Int);
    parseparam->postparam("ngap",&ngap,Int);

    parseparam->postparam("verbosity",&verbosity,Int);
    parseparam->postparam("logselect",&logselect,Int);

    parseparam->doparse();

    if (ntepat<0) ntepat = nstrpat;

}

void setuplogging() {

    if (logselect==0) return;

    if (1<=logselect) {

	troutlog = new PLogger(clpop,"act","trout.log");

	trteoutlog = new PLogger(clpop,"act","trteout.log");

	teoutlog = new PLogger(clpop,"act","teout.log");

    }

    if (2<=logselect) {

	trhidlog = new PLogger(hidpop,"act","trhid.log");

	tehidlog = new PLogger(hidpop,"act","tehid.log");

    }

    if (3<=logselect) {

	new PLogger(inhidprj,"Pj","pj_ih.log",nutrpat);

	new PLogger(inhidprj,"Bj","bj_ih.log",nutrpat);

    }

    if (4<=logselect) {

	new PLogger(hidpop,"dsupmax","hiddsupmax.log");
	
	new PLogger(hidpop,"expdsupsum","hidexpdsupsum.log");

	new PLogger(hidpop,"expdsup","hidexpdsup.log");

	new PLogger(hidpop,"dsup","hiddsup.log");

	new PLogger(hidpop,"act","hidact.log");

    }

    if (5<=logselect) {
	
	new PLogger(inpop,"inp","inp.log");

	new PLogger(inpop,"dsup","indsup.log");

	new PLogger(inpop,"act","inact.log");

    }

    if (6<=logselect) {

	new PLogger(inhidprj,"Zi","inhidzi.log");

	new PLogger(inhidprj,"Zj","inhidzj.log");

	new PLogger(inhidprj,"bwsup","inhidbwsup.log");

	new PLogger(inhidprj,"Wij","inhidwij.log",100);

    }
    
    if (7<=logselect) {

	new PLogger(clpop,"inp","clinp.log");

	new PLogger(clpop,"dsup","cldsup.log");

    }

    if (logselect==-5) {

	new PLogger(inhidprj,"Pi","pi_ih.log",100);

	new PLogger(inhidprj,"Pj","pj_ih.log",100);

	new PLogger(inhidprj,"Pij","pij_ih.log",100);

	new PLogger(inhidprj,"Mic","mic_ih.log",100);

	new PLogger(inhidprj,"Sil","sil_ih.log",100);

    }

    if (troutlog!=NULL) troutlog->stop(); if (teoutlog!=NULL) teoutlog->stop();
    if (trteoutlog!=NULL) trteoutlog->stop();
    if (trhidlog!=NULL) trhidlog->stop(); if (tehidlog!=NULL) tehidlog->stop();

}


int main(int argc,char **args) {

    string paramfile = "shclass_mnist.par"; // nbrav
    
    if (argc>1) paramfile = args[1];

    parseparams(paramfile);

    pginitialize(argc,args,verbosity!=0);

    int minpes = Rin + Rhid + 1 + Rhid + 1;

    if (isroot()) {

    	if (shsize<minpes) {

    	    fprintf(stderr,"shclassmain1::main: minpes -- shsize mismatch (minpes = %d)\n",minpes);

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

    if (verbosity>3 and isroot()) printf("Hhid = %d Mhid = %d Nhid = %d\n",Hhid,Mhid,Hhid*Mhid);

    hidpop = new PPopulation(Rhid,Hhid,Mhid,hidactfn_t);
    
    clpop = new PPopulation(1,1,Mout);

    float epocdur = nutrpat * timestep; // * nstep // nbrav

    inhidprj = new PProjection(inpop,hidpop,inhidpdens,INCR,HPATCHY);

    inhidprj->setparam(WDENS,inhidwdens);

    inhidprj->setparam(KBJHALF,kbjhalf);

    hidclprj = new PProjection(hidpop,clpop,1,INCR);

    pgalloc();

    DataFactory *datafactory = new DataFactory(Hin,Min);

    DataFactory *lblfactory = new DataFactory(1,Mout);

    if (datasetstr=="mnist") {

	datafactory->setdatapath(datapath);
	datafactory->loadMNIST();
	datafactory->cutout2(r0,r0+nrow,c0,c0+ncol);
	if (nrow*ncol!=Hin*Min)
	    perror("shclassmain1::main","Data image -- Input layer format mismatch");
	telbl = lblfactory->getpats(telbl,ntepat);

    } else if (datasetstr=="rnd01") {

	trdata = datafactory->mkpats(nutrpat,RND01,1,0);
	trlbl = lblfactory->mkpats(nutrpat,RND01,1,0);
	tedata = vector<vector<float> >(0);
	telbl = vector<vector<float> >(0);	    

    }

    if (0<=shseed) datafactory->setshseed(shseed);

    vector<int> idxvec(nutrpat); for (int i=0; i<nutrpat; i++) idxvec[i] = i;

    setuplogging();

    if (inactfn_t==SPKLIN) inpop->setparam(MAXFQ,maxfq);

    if (hidactfn_t==SPKBCP) hidpop->setparam(MAXFQ,maxfq);

    hidpop->setparam(TAUM,hidtaum);

    hidpop->setparam(AGAIN,hidagain);

    hidpop->setseed(hidseed);

    inpop->setparam(NMEAN,0); inpop->setparam(NAMP,0);

    hidpop->setparam(NMEAN,hidnmean); hidpop->setparam(NAMP,hidnamp);

    clpop->setparam(NMEAN,0); clpop->setparam(NAMP,0);

    clpop->setparam(BWGAIN,0);

    if (inhidtaup<0) inhidtaup = kTp*epocdur; // effective nstep = 1, since only one step has PRN>0

    if (verbosity>1 and isroot()) printf("inhidtaup = %.2f inhidtaupdt = %.2e\n",inhidtaup,timestep/inhidtaup);

    inhidprj->setparam(TAUP,inhidtaup);

    if (inhidtaub>0) inhidprj->setparam(TAUB,inhidtaub); else  inhidprj->setparam(TAUB,kTb*epocdur);

    inhidprj->setparam(BGAIN,hidbgain);

    inhidprj->setparam(PSILENT,inhidpsilent);

    inhidprj->setstrcpl(strcpl);

    inhidprj->setparam(FLPSCR,flpscr);

    inhidprj->setparam(PSPA,pspa);

    inhidprj->setparam(PSPC,pspc);

    if (hidcltaup>0) hidclprj->setparam(TAUP,hidcltaup); else  hidclprj->setparam(TAUP,kTp*epocdur);

    hidclprj->setparam(PRN,0);

    raisebarrier();

    Timer *alltimer = new Timer("Time elapsed");

    if (verbosity>1 and isroot()) {

	if (strcpl)
	    
	    printf("Unsupervised training with structural plasticity\n");

	else

	    printf("Unsupervised training without structural plasticity\n");

	fflush(stdout); }

    for (int epoc=0; epoc<nepoc; epoc++) {

	if (verbosity>2 and isroot()) { printf("\nepoc = %d: ",epoc); fflush(stdout); }

	if (epoc>0) hidpop->setparam(NAMP,0);

	for (int p=0; not pexitflg and p<nutrpat; p++ ) {

	    if (verbosity>3 and isroot()) { printf("c"); fflush(stdout); }

	    inpop->setparam(IGAIN,1); clpop->setparam(IGAIN,1);

	    inpop->setinp(trdata[idxvec[p]]); clpop->setinp(trlbl[idxvec[p]]);

	    psimulate(nstep - 1);

	    inhidprj->setparam(PRN,1);

	    psimulate(1);

	    // inpop->setparam(IGAIN,0); clpop->setparam(IGAIN,0); // nbrav

	    inhidprj->setparam(PRN,0); // nbrav

	    // psimulate(ngap); // nbrav

	    if (flpupdint>0) inhidprj->fliptpatchn(nflp,flpupdint);

	}

	if (0<=shseed) datafactory->getshuffledidx(idxvec);

    }
    
    if (isroot()) printf("\n");

    inhidprj->setparam(PSPA,0); inhidprj->setparam(PSPC,0);

    if (nstrpat>0) {

	if (verbosity>1 and isroot()) { printf("\nSupervised training (%d)\n",simstep); fflush(stdout); }

	inhidprj->setparam(PRN,0);

	for (int p=0; not pexitflg and p<nstrpat; p++ ) {

	    if (verbosity>3 and isroot()) { printf("s"); fflush(stdout); }

	    inpop->setparam(IGAIN,1); clpop->setparam(IGAIN,1);

	    inpop->setinp(trdata[p]); clpop->setinp(trlbl[p]);

	    psimulate(nstep - 1);

	    hidclprj->setparam(PRN,1);

	    if (p<ntetrpat) {

			if (troutlog!=NULL) troutlog->start();

			if (trhidlog!=NULL) trhidlog->start();

	    }

	    psimulate(1);

	    if (p<ntetrpat) {

			if (troutlog!=NULL) troutlog->stop();

			if (trhidlog!=NULL) trhidlog->stop();

	    }

	    inpop->setparam(IGAIN,0); clpop->setparam(IGAIN,0);

	    hidclprj->setparam(PRN,0);

	    psimulate(ngap);

	}
    }

    if (ntetrpat>0) {

	if (ntetrpat>nstrpat) perror("shclassmain1::main","Illegal: ntetrpat>nstrpat");

	if (verbosity>1 and isroot())

	    { printf("\nTesting on train patterns(%d)\n",simstep); fflush(stdout); }

	clpop->setparam(IGAIN,0);

	clpop->setparam(BWGAIN,1);

	for (int p=0; not pexitflg and p<ntetrpat; p++ ) {

	    if (verbosity>3 and isroot()) { printf("t"); fflush(stdout); }

	    inpop->setparam(IGAIN,1);
	
	    inpop->setinp(trdata[p]);

	    psimulate(nstep - 1);

	    if (trteoutlog!=NULL) trteoutlog->start();

	    psimulate(1);

	    if (trteoutlog!=NULL) trteoutlog->stop();

	    inpop->setparam(IGAIN,0);

	    psimulate(nstep);

	}
    }
    
    if (tedata.size()>0 and ntepat>0) {
    
	if (ntetrpat>nstrpat) perror("shclassmain1::main","Illegal: ntepat>nstrpat");

	if (verbosity>1 and isroot()) { printf("\nTesting on test patterns(%d)\n",simstep); fflush(stdout); }

	clpop->setparam(IGAIN,0);

	clpop->setparam(BWGAIN,1);

	hidclprj->setparam(PRN,0);

	for (int p=0; not pexitflg and p<ntepat; p++ ) {

	    if (verbosity>3 and isroot()) { printf("t"); fflush(stdout); }

	    inpop->setparam(IGAIN,1);
	
	    inpop->setinp(tedata[p]);

	    psimulate(nstep - 1);

	    if (teoutlog!=NULL) teoutlog->start();

	    if (tehidlog!=NULL) tehidlog->start();

	    psimulate(1);

	    if (teoutlog!=NULL) teoutlog->stop();

	    if (tehidlog!=NULL) tehidlog->stop();

	    inpop->setparam(IGAIN,0);

	    psimulate(nstep);

	}
    }
    
    if (pexitflg) goto finish;

    if (verbosity>1 and isroot()) alltimer->print();

    if (verbosity>2 and isroot()) printf("N:o simsteps = %d (%.3f sec)\n",simstep,simstep*timestep);

    inhidprj->fwritestate("Pi","inhidpi.bin");
    inhidprj->fwritestate("Pj","inhidpj.bin");
    inhidprj->fwritestate("Pij","inhidpij.bin");
    inhidprj->fwritestate("Bj","inhidbj.bin");
    inhidprj->fwritestate("Wij","inhidwij.bin");
    inhidprj->fwritestate("Mic","inhidmic.bin");
    inhidprj->fwritestate("Age","inhidage.bin");
    inhidprj->fwritestate("Sil","inhidsil.bin");
    inhidprj->fwritestate("Won","inhidwon.bin");
    hidclprj->fwritestate("Wij","hidclwij.bin");
    hidclprj->fwritestate("Bj","hidclbj.bin");
    
    if (logselect>0) {

      fwritemat(trlbl,"trlbl.bin"); // ala

      fwritemat(telbl,"telbl.bin");

    }

 finish:
    
    finalize(); return 0;

}
