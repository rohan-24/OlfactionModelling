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

int R1 = 2,H1 = 1,nstep = 1,ngap = 1,verbosity = 2;
float pdens = 1,wdens = 1,again = 1,hspc = 0.5e-3,cspeed = 0,wij = 0,wii = 0,stimval = 1,
    taum = 2*timestep,adgain = 0,taua = 0.1,nmean = 0,namp = 1;

void parseparams(std::string paramfile) {

    Parseparam *parseparam = new Parseparam(paramfile);

    parseparam->postparam("gseed",&gseed,Long);

    parseparam->postparam("R1",&R1,Int);
    parseparam->postparam("H1",&H1,Int);
    
    parseparam->postparam("taum",&taum,Float);
    parseparam->postparam("again",&again,Float);
    parseparam->postparam("adgain",&adgain,Float);
    parseparam->postparam("taua",&taua,Float);
    parseparam->postparam("nmean",&nmean,Float);
    parseparam->postparam("namp",&namp,Float);

    parseparam->postparam("pdens",&pdens,Float);
    parseparam->postparam("wdens",&wdens,Float);
    parseparam->postparam("wij",&wij,Float);
    parseparam->postparam("wii",&wii,Float);

    parseparam->postparam("hspc",&hspc,Float);
    parseparam->postparam("cspeed",&cspeed,Float);

    parseparam->postparam("stimval",&stimval,Float);
    parseparam->postparam("nstep",&nstep,Int);
    parseparam->postparam("ngap",&ngap,Int);
    
    parseparam->postparam("verbosity",&verbosity,Int);
    
    parseparam->doparse();

}

#undef isroot

#define isroot() (shrank==0)

PPopulation *pop1 = NULL;
PProjection *prj11 = NULL;

int nrow,ncol;

vector<vector<float> > mkdelmat(int H,float hspc,float cspeed,int dim = 2) {

    if (isroot()) printf("H = %d hspc = %f cspeed = %f\n",H,hspc,cspeed);

    vector<vector<float> > delmat(H,vector<float>(H,1));
    
    nrow = ceil(sqrt(H)); ncol = ceil(float(H)/nrow);

    // if (isroot()) printf("nrow = %d ncol = %d nrow*ncol = %d\n",nrow,ncol,nrow*ncol);

    int srcrow,srccol,trgrow,trgcol;

    float dist;

    for (int srch=0; srch<H; srch++) {

	switch (dim) {

	case 1: srcrow = 0; srccol = srch; break;
	    
	case 2: srcrow = srch/ncol; srccol = srch%ncol; break;

	default: perror("delmain2::mkdelmat","Illegal dim not in {1,2}");

	}

	for (int trgh=0; trgh<H; trgh++) {

	    switch (dim) {

	    case 1: trgrow = 0; trgcol = trgh; break;
	    
	    case 2: trgrow = trgh/ncol; trgcol = trgh%ncol; break;

	    default: perror("delmain2::mkdelmat","Illegal dim not in {1,2}");

	    }

	    dist = hspc * sqrt((trgrow-srcrow)*(trgrow-srcrow) + (trgcol-srccol)*(trgcol-srccol));

	    delmat[srch][trgh] = dist/cspeed + timestep;

	}
    }

    fflush(stdout);

    // if (verbosity>2 and isroot()) prnmat(delmat,4,7);

    if (isroot()) fwritemat(delmat,"delmat.bin");

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
	  if (optopt == 'R') {
	    if (isroot()) fprintf (stderr, "Option -%c requires an argument.\n", optopt);
	  } else if (isprint (optopt)) {
	      if (isroot()) fprintf (stderr, "Unknown option `-%c'.\n", optopt);
	  } else {
	      if (isroot()) fprintf (stderr,
				     "Unknown option character `\\x%x'.\n",
				     optopt);
	  }
        return 1;

      default:
        abort ();
      }

  if (isroot()) printf ("R1 = %d\n",R1);

  for (index = optind; index < argc; index++)
    if (isroot()) printf ("Non-option argument %s\n", argv[index]);

  return 0;

}


int main(int argc,char **args) {

    string paramfile = "delmain2.par";
    
    // if (argc>1) paramfile = args[1];

    parseparams(paramfile);

    pginitialize(argc,args);

    procargs(argc,args);

    if (isroot()) printf("H1 = %d R1 = %d H1%R1 = %d\n",H1,R1,H1%R1);
    
    if (H1%R1!=0 and isroot()) perror("delmain2::main","Illegal: H1%R1!=0");

    int minpes = R1 + R1;

    if (isroot()) {

	if (shsize<minpes) {

	    fprintf(stderr,"delmain2::main: minpes -- shsize mismatch (minpes = %d -- shsize = %d)\n",
		    minpes,shsize);

	    shmem_global_exit(9);

	}
    }

    gsetseed(gseed + shrank);

    pop1 = new PPopulation(R1,H1,1,LIN,CAP);
    
    prj11 = new PProjection(pop1,pop1,pdens,FIXED);

    prj11->setparam(WDENS,wdens);

    vector<vector<float> > Wij(H1,vector<float>(H1,wij));

    for (int h=0; h<H1; h++) Wij[h][h] = wii;	    

    prj11->setWij(Wij);

    if (cspeed>0) {

	vector<vector<float> > delmat = mkdelmat(H1,hspc,cspeed,2);

	prj11->setdelays(delmat);

    }

    pgalloc();

    new PLogger(pop1,"inp","inp.log");

    new PLogger(pop1,"dsup","dsup.log");

    new PLogger(pop1,"act","act.log");

    pop1->setparam(TAUM,taum);

    pop1->setparam(AGAIN,again);

    pop1->setparam(NMEAN,nmean);

    pop1->setparam(NAMP,namp);

    pop1->setparam(ADGAIN,adgain);

    pop1->setparam(TAUA,taua);

    raisebarrier();

    vector<float> stim(H1,0.1);

    stim[nrow/2 * ncol + ncol/2] = stimval;

    pop1->setinp(stim);

    Timer *alltimer = new Timer("Time elapsed");

    psimulate(nstep);

    pop1->setparam(IGAIN,0);

    psimulate(ngap);

    raisebarrier();

    if (pexitflg) goto finish;

    if (verbosity>0 and isroot()) { printf("N:o steps = %d\n",simstep); alltimer->print(); }

    prj11->fwritestate("Wij","Wij.bin");

 finish:
    
    finalize(); return 0;

}
