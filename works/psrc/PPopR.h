#ifndef __PPopR_included
#define __PPopR_included

#include <vector>
#include <string>
#include <mpi.h>

#include "bcpnnsim.h"
#include "PAxons.h"
#include "PPobjR.h"


class PIO;


class PPopR : public PPobjR {

public:

    static int maxidelay;

protected:

    int N,HperR,NperH,NperR,hoffs,noffs,ninprj;

    PopH *poph;

    std::vector<Pop *> pops;

    std::vector<float> ppact,ppactnz;

    bool isspiking,issrcpop,istrgpop;

    float Esyn;

    friend class PPobjR;

    friend class PPrjR;

    friend class PPopulation;

    friend class PProjection;

public:

    PPopR(int npop0,int rank0,int nrank,int Hoffs,int HperR,int M,int U = 1,
	  Globals::Actfn_t actfn_t = Globals::BCP,Globals::Normfn_t normfn_t = Globals::FULL) ;

    void setseed(long newseed) ;

    void setparam(Globals::Popparam_t poppar_t,float parval) ;

    void setmaxdelay(float maxdelay) ;

    void reallocaxons(float maxdelay) ;

    void resetstate() ;

    void setinp(float inpval) ;

    void setinp(std::vector<float> inpvec) ;

    void gatheract() ;

    void doputsrcact() ;

    void gatheractnz() ;

    void doputsrcactnz() ;

    void doupdpaxons() ;

    void doputtrgact() ;

    void dosumbwcond() ;

    void fwritestate(PIO *pio,std::string statestr) ;

} ;


class PPopulation : public PPobjRH {

protected:

    static int ppopul_n,ppopul_npop;

    int N,H,M,U,HperR,NperH,ninprj;

    bool isspiking;

    float maxfq,maxdelay;

    friend class PProjection;

    friend class PLogger;

public:

    PPopulation(int nrank,int H,int M,int U,Globals::Actfn_t actfn_t = Globals::BCP,
    		Globals::Normfn_t normfn_t = Globals::FULL,float maxfq = 1/Globals::timestep) ;

    PPopulation(int nrank,int H,int M,Globals::Actfn_t actfn_t = Globals::BCP,
		Globals::Normfn_t normfn_t = Globals::FULL,float maxfq = 1/Globals::timestep) ;

    void setseed(long newseed) ;

    void setparam(Globals::Popparam_t poppar_t,float parval) ;

    void setmaxdelay(float maxdelay) ;

    void resetstate() ;

    void setinp(float inpval) ;

    void setinp(std::vector<float> inpvec) ;

    void prnidandseed() ;

    void fwritestate(PIO *pio,std::string statestr) ;

    void fwritestate(std::string statestr,std::string filename) ;

} ;

#endif // __PPopR_included
