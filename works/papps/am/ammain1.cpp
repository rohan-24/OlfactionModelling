#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ctype.h>
#include <shmem.h>
#include <shmemx.h>
#include <assert.h>

#include "bcpnnsim.h"
#include "pbcpnnsim.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;

int R1 = 2,H1 = 1,M1 = 1,nstep = 1,ngap = 1,ntrpat = 10,ntepat = 10,nrep = 1,
    verbosity = 2,logselect = 0,flpscr = 1;
float pdens = 1,wdens = 1,again = 1,taup = 1,psilent = 0,hspc = 0.5e-3,cspeed = 0,pspa = 0,pspc = 0,
    maxfq = 100;
string patform = "ortho",actfn = "BCP";

void parseparams(std::string paramfile) {

    Parseparam *parseparam = new Parseparam(paramfile);

    parseparam->postparam("gseed",&gseed,Long);

    parseparam->postparam("R1",&R1,Int);
    parseparam->postparam("H1",&H1,Int);
    parseparam->postparam("M1",&M1,Int);
    
    parseparam->postparam("actfn",&actfn,String);
    parseparam->postparam("maxfq",&maxfq,Float);

    parseparam->postparam("patform",&patform,String);
    
    parseparam->postparam("pdens",&pdens,Float);
    parseparam->postparam("wdens",&wdens,Float);
    parseparam->postparam("again",&again,Float);
    parseparam->postparam("taup",&taup,Float);
    parseparam->postparam("psilent",&psilent,Float);

    parseparam->postparam("flpscr",&flpscr,Int);
    parseparam->postparam("pspa",&pspa,Float);
    parseparam->postparam("pspc",&pspc,Float);

    parseparam->postparam("hspc",&hspc,Float);
    parseparam->postparam("cspeed",&cspeed,Float);

    parseparam->postparam("ntrpat",&ntrpat,Int);
    parseparam->postparam("nrep",&nrep,Int);
    parseparam->postparam("ntepat",&ntepat,Int);
    parseparam->postparam("nstep",&nstep,Int);
    parseparam->postparam("ngap",&ngap,Int);
    
    parseparam->postparam("verbosity",&verbosity,Int);
    parseparam->postparam("logselect",&logselect,Int);
    
    parseparam->doparse();

}

#undef isroot

#define isroot() (shrank==0)

PPopulation *pop1 = NULL;
Actfn_t actfn_t = BCP;
PProjection *prj11 = NULL,*prj11_2 = NULL;
PLogger *acttr = NULL,*actte = NULL,*popbwsup = NULL,*prjbwsup = NULL;


vector<vector<float> > mkdelmat(int H,float hspc,float cspeed) {

    if (isroot()) printf("H = %d hspc = %f cspeed = %f\n",H,hspc,cspeed);

    vector<vector<float> > delmat(H,vector<float>(H,1));
    
    int nrow = ceil(sqrt(H)),ncol = ceil(float(H)/nrow);

    if (isroot()) printf("nrow = %d ncol = %d nrow*ncol = %d\n",nrow,ncol,nrow*ncol);

    int srcrow,srccol,trgrow,trgcol;

    float dist;

    for (int srch=0; srch<H; srch++) {

	srcrow = srch/ncol; srccol = srch%ncol;

	for (int trgh=0; trgh<H; trgh++) {

	    trgrow = trgh/ncol; trgcol = trgh%ncol;

	    dist = hspc * sqrt((trgrow-srcrow)*(trgrow-srcrow) + (trgcol-srccol)*(trgcol-srccol));

	    delmat[srch][trgh] = dist/cspeed + timestep;

	}
    }

    fflush(stdout);

    if (verbosity>3 and isroot()) prnmat(delmat,3,6);

    return delmat;

}


int procargs(int argc, char **argv) {
    char *Rvalue = NULL;
  int index;
  int c;

  opterr = 0;

  while ((c = getopt (argc, argv, "R:")) != -1)
    switch (c)
      {
      case 'R':
        Rvalue = optarg;
	R1 = atoi(Rvalue);
        break;
      case '?':
        if (optopt == 'R')
	    if (isroot()) fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          if (isroot()) fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          if (isroot()) fprintf (stderr,
				 "Unknown option character `\\x%x'.\n",
				 optopt);
        return 1;

      default:
        abort ();
      }

  if (isroot()) printf ("R1 = %d\n",R1);

  for (index = optind; index < argc; index++)
    if (isroot()) printf ("Non-option argument %s\n", argv[index]);

  return 0;

}


void setuplogging() {

    if (logselect<=0) return;

    new PLogger(pop1,"inp","inp.log");

    new PLogger(pop1,"dsup","dsup.log");

    new PLogger(pop1,"act","act.log");

    acttr = new PLogger(pop1,"act","acttr.log"); acttr->stop();
    
    actte = new PLogger(pop1,"act","actte.log"); actte->stop();
    
    popbwsup = new PLogger(pop1,"bwsup","popbwsup.log"); popbwsup->stop();
    
    prjbwsup = new PLogger(prj11,"bwsup","prjbwsup.log"); prjbwsup->stop();

    new PLogger(prj11,"delact","delact.log");

    // new PLogger(prj11,"Zi","Zi.log");

    // new PLogger(prj11,"Pi","Pi.log");

    new PLogger(prj11,"Zj","Zj.log");

    // new PLogger(prj11_2,"Zi","Zi_2.log");

    // new PLogger(prj11_2,"bwsup","prjbwsup_2.log");

    // new PLogger(prj11,"Pj","Pj.log");

    // new PLogger(prj11,"Mic","Mic.log",10);
    
    // new PLogger(prj11,"Sil","Sil.log",10);
    
}    


int main(int argc,char **args) {

    string paramfile = "ammain1.par";
    
    if (argc>1) paramfile = args[1];

    parseparams(paramfile);

    pginitialize(argc,args);

    procargs(argc,args);

    int minpes = R1 + R1;

    if (isroot()) {

	if (shsize<minpes) {

	    fprintf(stderr,"ammain1::main: minpes -- shsize mismatch (minpes = %d -- shsize = %d)\n",
		    minpes,shsize);

	    shmem_global_exit(9);

	}
    }

    gsetseed(gseed);

    if (isroot()) printf("gseed = %ld\n",gseed);

    if (actfn=="SPKBCP") actfn_t = SPKBCP;
    
    pop1 = new PPopulation(R1,H1,M1,actfn_t);
    
    prj11 = new PProjection(pop1,pop1,pdens,INCR);

    // prj11_2 = new PProjection(pop1,pop1,pdens,INCR);

    if (cspeed>0) {

	vector<vector<float> > delmat = mkdelmat(H1,hspc,cspeed);

	prj11->setdelays(delmat);

    }

    pgalloc();

    Pat_t pat_t = ORTHO; if (patform=="rnd01") pat_t = RND01;

    DataFactory *datafactory = new DataFactory(H1,M1);

    vector<vector<float> > trpats = datafactory->mkpats(ntrpat,pat_t);

    if (verbosity>3 and isroot()) prnmat(trpats,0);
    
    vector<vector<float> > tepats = trpats;

    for (int p=0; p<ntepat; p++)

    	for (int m=0; m<M1; m++) tepats[p][(H1-1)*M1+m] = 1./M1;

    if (verbosity>3 and isroot()) prnmat(tepats,0);

    setuplogging();

    pop1->setparam(TAUM,2*timestep);

    if (actfn_t==SPKBCP) pop1->setparam(MAXFQ,maxfq);

    pop1->setparam(AGAIN,again);

    pop1->setparam(BWGAIN,0);

    prj11->setparam(WDENS,wdens);

    prj11->setparam(TAUP,taup);

    prj11->setparam(PRN,1);

    // prj11_2->setparam(WDENS,wdens);

    // prj11_2->setparam(PRN,1);

    prj11->setparam(PSILENT,psilent);

    prj11->setparam(FLPSCR,flpscr);

    prj11->setparam(PSPA,pspa);

    prj11->setparam(PSPC,pspc);

    // prj11->prntpatchn();

    raisebarrier();

    if (isroot()) printf("Training\n");

    Timer *alltimer = new Timer("Time elapsed");

    for (int rep=0; rep<nrep; rep++) {

	if (verbosity>1 and isroot()) { printf("rep = %d\n",rep); fflush(stdout); }

	for (int p=0; not pexitflg and p<ntrpat; p++ ) {

	    if (verbosity>2 and isroot()) { printf("e"); fflush(stdout); }

 	    pop1->setinp(trpats[p]);

	    if (rep==nrep-1 and acttr!=NULL) acttr->start();

 	    psimulate(nstep);

	    if (acttr!=NULL) acttr->stop();

	    pop1->setinp(0);

	    psimulate(ngap);

	}
    }

    prj11->setparam(PRN,0);

    // prj11_2->setparam(PRN,0);

    prj11->setparam(BGAIN,1);

    prj11->setparam(WGAIN,1);

    prj11->setparam(PSPA,0); prj11->setparam(PSPC,0);

    raisebarrier();

    // prj11->prntpatchn();

    if (isroot()) printf("Testing (%d)\n",simstep);

    for (int p=0; not pexitflg and p<ntepat; p++ ) {

	if (verbosity>2 and isroot()) { printf("r"); fflush(stdout); }

    	pop1->setinp(tepats[p]);

	pop1->setparam(IGAIN,1); pop1->setparam(BWGAIN,1);

	if (popbwsup!=NULL) popbwsup->start(); if (prjbwsup!=NULL) prjbwsup->start();

	if (actte!=NULL) actte->start();

    	psimulate(nstep);

	if (actte!=NULL) actte->stop();

	if (popbwsup!=NULL) popbwsup->stop(); if (prjbwsup!=NULL) prjbwsup->stop();

    	pop1->setparam(IGAIN,0);

    	psimulate(nstep);

	pop1->setparam(BWGAIN,0);

    	psimulate(ngap);

    }

    if (pexitflg) goto finish;

    if (verbosity>0 and isroot()) { printf("N:o steps = %d\n",simstep); alltimer->print(); }


    prj11->fwritestate("Mic","mic.bin");
    prj11->fwritestate("Sil","sil.bin");
    prj11->fwritestate("Wij","wij.bin");
    prj11->fwritestate("Won","won.bin");
    prj11->fwritestate("Bj","bj.bin");
    prj11->fwritestate("Bj","Bj.bin");
    prj11->fwritestate("Wij","Wij.bin");
    prj11->fwritestate("Pij","Pij.bin");

 finish:
    
    finalize(); return 0;

}
