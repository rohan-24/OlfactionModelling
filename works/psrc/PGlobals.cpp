#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <assert.h>

#include "pbcpnnsim.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;

string PGlobals::PBCPNNSim_version = "PBCPNNSim Version 0.9.5";
int PGlobals::mpisize,PGlobals::mpirank,PGlobals::pexitflg = 0;
int PGlobals::shsize,PGlobals::shrank;
PPobjR *PGlobals::locppobjr = NULL;
PPopR *PGlobals::locppopr = NULL;
PPrjR *PGlobals::locpprjr = NULL;

void PGlobals::pginitialize(int argc,char **args,bool printversion) {

    ginitialize(false);

    MPI_Init(&argc,&args);

    MPI_Comm_size(MPI_COMM_WORLD,&mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);

    shmem_init();
    shsize = num_pes();
    shrank = my_pe();

    if (printversion and shrank==0) printf("Starting %s ...\n",PBCPNNSim_version.c_str()); fflush(stdout);

}


void PGlobals::perror(string errloc,string errstr) {

    fprintf(stderr,"ERROR in %s %s\n",errloc.c_str(),errstr.c_str()); fflush(stderr);

    raisebarrier();

    shmem_global_exit(9);

}


void PGlobals::pgalloc() { PPobjR::palloc(); }


void PGlobals::raisebarrier() {

    shmem_barrier_all();

}

#define DOCOMM
#define DOCOMP
#define DOSYNC
#define RMSYNC
#define DOLOGG

void PGlobals::psimulate(int nstep) {

#ifdef DOSYNC
    raisebarrier();
#endif

    setupdone = true;

    for (int step=0; step<nstep; step++) {

#ifdef DOCOMM
#ifndef	SRCACTNZ
 	if (locppopr!=NULL) locppopr->doputsrcact();
#endif
#endif

#ifdef DOCOMM
#ifdef	SRCACTNZ
 	if (locppopr!=NULL) locppopr->doputsrcactnz();
#endif
#endif

#ifdef DOSYNC
#ifndef RMSYNC
	raisebarrier();
#endif
#endif

#ifdef DOCOMM
	if (locppopr!=NULL) locppopr->doupdpaxons();
#endif

#ifdef DOSYNC
	raisebarrier();
#endif

#ifdef DOCOMM
	if (locpprjr!=NULL) locpprjr->dogetdelsrcact();
#endif

#ifdef DOSYNC
#ifndef RMSYNC
	raisebarrier();
#endif
#endif

#ifdef DOCOMM
#ifndef	SRCACTNZ
 	if (locpprjr!=NULL) locpprjr->doscattersrcact();
#endif
#endif

#ifdef DOCOMM
#ifdef	SRCACTNZ
 	if (locpprjr!=NULL) locpprjr->doscattersrcactnz();
#endif
#endif

#ifdef DOSYNC
#ifndef RMSYNC
	raisebarrier();
#endif
#endif

#ifdef DOCOMP
	Pop::resetbwsupall();

	Prj::updateall();
#endif

#ifdef DOSYNC
#ifndef RMSYNC
	raisebarrier();
#endif
#endif

#ifdef DOCOMM
	if (locpprjr!=NULL) locpprjr->gatherbwcond();
#endif

#ifdef DOSYNC
#ifndef RMSYNC
	raisebarrier();
#endif
#endif

#ifdef DOCOMM
	if (locpprjr!=NULL) locpprjr->doputbwcond();
#endif

#ifdef DOSYNC
	raisebarrier();
#endif

#ifdef DOCOMM
	if (locppopr!=NULL) locppopr->dosumbwcond();
#endif

#ifdef DOSYNC
#ifndef RMSYNC
	raisebarrier();
#endif
#endif

#ifdef DOCOMP
	Pop::updateall();
#endif

#ifdef DOSYNC
#ifndef RMSYNC
	raisebarrier();
#endif
#endif

#ifdef DOCOMM
	if (locppopr!=NULL) locppopr->doputtrgact();
#endif

#ifdef DOSYNC
	raisebarrier();
#endif

#ifdef DOCOMM
 	if (locpprjr!=NULL) locpprjr->doscattertrgact();
#endif

#ifdef DOSYNC
#ifndef RMSYNC
	raisebarrier();
#endif
#endif

#ifdef DOCOMM
	if (locpprjr!=NULL) locpprjr->updgntpatch();
#endif

#ifdef DOSYNC
	raisebarrier();
#endif

#ifdef DOLOGG
	PLogger::dologall();
#endif

	simstep++;

	raisebarrier();

    }
}


void PGlobals::finalize() {

    shmem_barrier_all();

    PIO::pfcloseall();

    //locppobjr->finalize();

    shmem_finalize();

    MPI_Finalize();

}
