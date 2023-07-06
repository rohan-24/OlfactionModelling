#ifndef __PGlobals_included
#define __PGlobals_included

#include <vector>
#include <string>
#include <mpi.h>
#include <shmem.h>
#include <shmemx.h>

#include "Globals.h"
#include "Pop.h"

#define isroot() (mpirank==0)

#define SRCACTNZ

class PPobjR;

class PPopR;

class PPrjR;

namespace PGlobals {

    enum PPobjR_t { EMPTYOBJ = 0,PPOP, PPRJ, PPOPULATION, PPROJECTION } ;

    enum Shfield_t { SRCMEM = 0, SRCMEMNZ, SRCMEMSPK, TRGMEM, BWCMEM, AXNMEM, NSHFIELD } ;

    extern std::string PBCPNNSim_version;

    extern int mpisize,mpirank;

    extern int shsize,shrank;

    extern PPobjR *locppobjr;

    extern PPopR *locppopr;

    extern PPrjR *locpprjr;

    extern int pexitflg;

    void pginitialize(int argc,char **args,bool printversion = true) ;

    void perror(std::string errloc,std::string errstr) ;
    
    void raisebarrier() ;

    void pgalloc() ;

    void psimulate(int nstep = 1) ;

    void finalize() ;

}

#endif // __PGlobals_included
