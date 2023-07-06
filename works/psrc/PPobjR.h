#ifndef __PPobjR_included
#define __PPobjR_included

#include <vector>
#include <string>
#include <mpi.h>
#include <shmem.h>
#include <shmemx.h>

#include "PGlobals.h"
#include "Globals.h"

class Shmem {

protected:

    static bool alloced;

    static long psync[];
    static int maxnvec[],maxmvec[];

    static std::vector<float *> shmemmat;

    friend class PPobjR;

    friend class PPopR;

    friend class PPrjR;

public:

    static void reqshmem(PGlobals::Shfield_t shfield_t,int n,int N = 0) ;

    static void mkshalloc() ;

    static float *getshmemarr(PGlobals::Shfield_t shmemfield_t) ;

} ;


class PAxons;

class PPobjR {

protected:

    int rank0,nrank,locrank;

    static std::vector<float *> shaxons;

    static PAxons *paxons;

    std::vector<float> ppbwcond;

    std::vector<int> fwranks,bwranks;

    int ninprj;
    // For PPopR object: the number of projections that
    // target it.
    // for PPrjR object: the allocation order number of the projection
    // targeting the PPopR.

    friend class Shmem;

    friend class PPopR;

    friend class PPrjR;

    friend class PIO;

    friend class PPopulation;

    friend class PProjection;

public:

    static void palloc() ;

    PPobjR(int rank0,int nrank) ;

    void dosumbwcond() ;

    void finalize() ;

} ;

class PPobjRH {

 protected:

    static int lastrank,nobj;

    int id;

    int rank0,nrank,rankn;

 public:

    PPobjRH(int nrank) ;

    bool onthisrank() ;

    bool onrank0() ;

    int getrank0() ;

    int getlocrank() ;

} ;

#endif // __PPobjR_included
