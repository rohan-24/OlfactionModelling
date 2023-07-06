#ifndef __PPrjR_included
#define __PPrjR_included

#include "PPobjR.h"

class PPrj;

class PPrjH;

class PIO;

class PProjection;

class PPrjR : public PPobjR {

public:

    static std::vector<PPrj *> pprjs;

    static int *gntpatch;

    static long gtpatchsync[];

    static int *gtpatchpwrk;

    static int mvmisrc[],mvmitrg[];

    static int maxidelay;

protected:

    int srcN,srcNperR,srcNperH,trgN,trgH,trgNperH,trgHperR,trgNperR,trghoffs;

    std::vector<float> ppbwcond;
    int srcrank0,srcrankn,bwctrgrank;
    bool spikinginput;

    PPrjH *pprjh;

    std::vector<std::vector<int> > idelay;

    friend class PProjection;

public:

    PPrjR(int nprj0,int rank0,int nrank,int srcpopnrank,int srcH,int srcM,int srcU,float srcspkwgain,
	  int trgN,int trghoffs,int trgHperR,int trgM,int trgU,float trgspkwgain,float pdens,
	  Globals::BCPvar_t bcpvar = Globals::INCR,
	  Globals::Prjarch_t bccarch = Globals::HPATCHY) ;

    void setparam(Globals::Prjparam_t param_t,float parval) ;

    void setdelays(std::vector<std::vector<float> > delaymat) ;

    void setWij(std::vector<std::vector<float> > Wij) ;

    void doscattersrcact() ;

    void doscattersrcactnz() ;

    void doscattertrgact() ;

    void dogetdelsrcact() ;

    void gatherbwcond() ;

    void doputbwcond() ;

    void updgntpatch() ;

    void setstrcpl(bool on) ;

    void fliptpatchn(int nflp,int flpupdint);

    void fwritestate(PIO *pio,std::string statestr) ;

} ;


class PProjection : public PPobjRH {

protected:

    static int pproje_n,pproje_nprj;

    PPopulation *srcppopulation,*trgppopulation;

    int srcR,trgR,nrank;

    int srcN,srcH,trgH,trgN,trgNperH,trgNperR;

    friend class PPopulation;

    friend class PPrjR;

    friend class PLogger;

public:

    PProjection(PPopulation *srcppopulation,PPopulation *trgppopulation,float pdens,
		Globals::BCPvar_t bcpvar = Globals::INCR,
		Globals::Prjarch_t bccarch = Globals::HPATCHY) ;

    void prnidandseed() ;

    void setparam(Globals::Prjparam_t param_t,float parval) ;

    void setdelays(std::vector<std::vector<float> > delaymat) ;

    void setWij(std::vector<std::vector<float> > Wij) ;

    void mktrgranks() ;

    void mkgntpatch() ;

    void setstrcpl(bool on) ;

    void fliptpatchn(int nflp,int flpupdint);

    void prntpatch(int shrank = -1) ;

    void fwritestate(PIO *pio,std::string statestr) ;

    void fwritestate(std::string statestr,std::string filename) ;

} ;


#endif // __PPrjR_included
