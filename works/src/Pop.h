/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __POP_included
#define __POP_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"

class Logger;

class Prj;

class PopH;

class Pop {

 protected:

    static std::vector<Pop *> pops;

    Globals::Actfn_t actfn_t;

    Globals::Normfn_t normfn_t;

    int id,M,U,N,hoffs,noffs;

    PopH *poph;

    float igain,again,bwgain,taumdt,adgain,tauadt,namp,nmean,maxfq,thres,spkwgain,Esyn;

    std::vector<float> inp,bwsup,dsup,pact,act,ada,H_en;
    std::vector<int> spkstep;
    int spkiper,spkipha;

    std::vector<float> xyz;

    std::vector<Prj *> inprjs,utprjs;

    int now,maxidelay;
    float *axons;  // Needs to be 'float *' because of psrc, not garbage collected

    long seed;
    std::mt19937_64 generator;
    std::poisson_distribution<int> poissondistr;
    std::uniform_real_distribution<float> uniformdistr;

    friend class Prj;

    friend class PPop; /// Needed?

    friend class PPopH; /// Needed?

    friend class PopH;

    friend class SNNPopH;

    friend class BCU;

    friend class PPopR;

 public:

    static Pop *getpop(int id = -1) ;

    static int getnpop() ;

    Pop(int M,Globals::Actfn_t actfn_t = Globals::BCP,Globals::Normfn_t normfn_t = Globals::FULL) ;

    Pop(int M,int U = 1,Globals::Actfn_t actfn_t = Globals::BCP,Globals::Normfn_t normfn_t = Globals::FULL) ;

    Pop(int M,int U = 1,bool noinit = true) ;

    void init() ;

    virtual void initparams() ;

    virtual void allocatestate() ;

    int getinfo(std::string infostr) ;

    void setxyz(std::vector<float> xyz) ;

    std::vector<float> const &getxyz() ;

    void setupaxons() ;

    static void setupaxonsall() ;

    void updaxons() ;

    void setnmean(float nmean) ;

    void fixupspkwgain() ;

    virtual bool checkparam(std::string paramstr) ;

    virtual void setparam(std::string parstr,float parval) ;

    virtual void setparam(Globals::Popparam_t param_t,float parval) ;

    virtual void setparamsfromfile(std::string paramfilename) ;

    virtual float getparam(std::string paramstr) ;

    void sethoffs(int hoffs) ;

    void setnoffs(int noffs) ;

    void sethnoffs(int hoffs) ;

    virtual std::vector<float> const &getstate(std::string statestr) ;

    void cpyvectostate(std::string statestr,std::vector<float> &vec) ;

    void cpystatetovec(std::string statestr,std::vector<float> &vec) ;

    long setseed(long seed = 0) ;

    long getseed() ;

    int nextpoisson() ;

    float nextfloat() ;

    void prnparam(std::string paramstr) ;

    virtual void prnstate(std::string statestr) ;

    virtual void fwritestate(std::string statestr,FILE *outf) ;

    virtual void fwritestate(std::string statestr,std::string filename) ;

    virtual Logger *logstate(std::string statestr,std::string filename,int logstep = 1) ;

    virtual void setinp(float inpval,int n = -1) ;

    virtual void setinp(std::vector<float> inpvec) ;

    virtual void resetstate() ;

    static void resetstateall() ;

    void setmaxidelay(float maxdelay) ;

    int getdslot(int idelay) ;

    float getdelact(int m,int idelay) ;

    virtual void updada() ;

    void upddsup() ;
    
    void compute_energy();

    virtual void normalize() ;

    void updLIN() ;

    void updLOG() ;

    void updSIG() ;

    void updEXP() ;

    virtual void updspk() ;

    virtual void update() ;

    static void updateall() ;

    void resetbwsup() ;

    static void resetbwsupall() ;

    void contribute( std::vector<float> &cond) ;

} ;


class ExpPop : public Pop {

 protected:

    std::vector<float> expdsup;
    float expdsupsum,dsupmax;

 public:

    ExpPop(int M,int U,Globals::Actfn_t actfn_t,Globals::Normfn_t normfn_t) ;

    ExpPop(int M,Globals::Actfn_t actfn_t,Globals::Normfn_t normfn_t) ;

    void expallocatestate();

    float getstate1(std::string statestr) ;

    virtual std::vector<float> const &getstate(std::string statestr) ;

    virtual void prnstate(std::string statestr) ;

    virtual void fwritestate(std::string statestr,FILE *outf) ;

    virtual void resetstate() ;

    void upddsupmax() ;

    void updexpdsup() ;

    void updexpdsupsum() ;

    void updEXP() ;

    void normalize() ;

    void update() ;

} ;


class BCPPop : public ExpPop {

 public:

    BCPPop(int M,int U,Globals::Actfn_t actfn_t,Globals::Normfn_t normfn_t) ;

    BCPPop(int M,Globals::Actfn_t actfn_t,Globals::Normfn_t normfn_t) ;

    virtual void setinp(float inpval,int n = -1) ;

    virtual void setinp(std::vector<float> inpvec) ;

    virtual void fwritestate(std::string statestr,FILE *outf) ;

} ;


class SNNPop : public Pop {

 protected:

    float C,gL,EL,DT,VR,VT,spkreft,taua;
    int spkireft;

 public:

    SNNPop(int M,Globals::Actfn_t actfn_t = Globals::AdExS) ;

    SNNPop(int M,int U,Globals::Actfn_t actfn_t = Globals::AdExS) ;

    virtual void initparams() ;

    virtual bool checkparam(std::string paramstr) ;

    virtual void setparamsfromfile(std::string paramfilename) ;

    virtual void setparam(std::string paramstr,float paramval) ;

    virtual float getparam(std::string paramstr) ;

    virtual void allocatestate() ;

    virtual void resetstate() ;

    virtual void updada() ;

    virtual void updspk() ;

    virtual void update() ;

} ;

#endif // __POP_included
