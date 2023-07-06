#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <cstring>
#include <assert.h>

#include "pbcpnnsim.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;

vector<float *> Shmem::shmemmat(NSHFIELD); bool Shmem::alloced = false;
long Shmem::psync[SHMEM_REDUCE_SYNC_SIZE];
int Shmem::maxnvec[NSHFIELD],Shmem::maxmvec[NSHFIELD];

void Shmem::reqshmem(Shfield_t shfield_t,int n,int m) {

    if (alloced) perror("Shmem::reqshmem","Illegal if 'alloced'");

    if (n>maxnvec[shfield_t]) maxnvec[shfield_t] = n;

    if (m>maxmvec[shfield_t]) maxmvec[shfield_t] = m;

}


void Shmem::mkshalloc() {

    int pwrk_size = max(NSHFIELD/2 + 1, _SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    int *pwrk = (int *)shmem_malloc(pwrk_size * sizeof(*pwrk));
    assert(pwrk != NULL);

    for (int i=0; i<SHMEM_REDUCE_SYNC_SIZE; i++) psync[i] = SHMEM_SYNC_VALUE;

    raisebarrier();

    shmem_int_max_to_all(maxnvec,maxnvec,NSHFIELD,0,0,shsize,pwrk,psync);

    for (size_t shf=0; shf<NSHFIELD; shf++) {

	// if (isroot()) printf("shf = %d n = %d m = %d\n",shf,maxnvec[shf],maxmvec[shf]);

	if (shf==AXNMEM) {

	    PPobjR::paxons = new PAxons(maxnvec[shf],maxmvec[shf]+1);

	    PPobjR::shaxons = PPobjR::paxons->shaxons;

	} else

	    Shmem::shmemmat[shf] = (float *)shmem_calloc(maxnvec[shf],sizeof(float));

    }

    alloced = true;

}


float *Shmem::getshmemarr(Shfield_t shfield_t) {

    if (not alloced) perror("Shmem::getshmem","Illegal when not 'shalloced'");

    return shmemmat[shfield_t];

}


PAxons *PPobjR::paxons;
vector<float *> PPobjR::shaxons;

void PPobjR::palloc() {

    Shmem::mkshalloc();

}


PPobjR::PPobjR(int rank0,int nrank) {

    locppobjr = this;

    this->rank0 = rank0;

    this->nrank = nrank;

    locrank = shrank - rank0;

}


void PPobjR::finalize() {

}


int PPobjRH::lastrank = 0,PPobjRH::nobj = 0; 

PPobjRH::PPobjRH(int nrank) {

    if (nrank<=0) perror("PPobjRH::PPobjRH","Illegal nrank<=0");

    if (lastrank + nrank>mpisize) perror("PPobjRH::PPobjRH","Cannot create 'nrank' nrank = " +
					 to_string(nrank) + " mpisize = " + to_string(shsize) +
					 " lastrank = " + to_string(lastrank));

    rank0 = lastrank;
    this->nrank = nrank;

    rankn = rank0 + nrank - 1;

    lastrank = rank0 + nrank;

}


bool PPobjRH::onthisrank() { return rank0<=shrank and shrank<rank0 + nrank; }


bool PPobjRH::onrank0() { return mpirank==rank0; }


int PPobjRH::getrank0() { return rank0; }


int PPobjRH::getlocrank() {

    if (not onthisrank()) perror("PPobjRH::getlocrank","Illegal if onthisrank()==false");

    return shrank - rank0;

}
