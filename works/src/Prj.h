/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __Prj_included
#define __Prj_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"


class Prj;

class Connij {

public:

 protected:

    Prj *prj;

    int srcn,trgM,bstep;

    float Zi,Ei,Pi,mic,eps,sdep,tausddt,sfac,tausfdt,sp0;

    bool silent;

    std::vector<float> Wij,Pij,Eij,won;

    friend class Prj;

    friend class PPrj;

 public:

    Connij(Prj *prj,int srcn,int trgM,float psilent,float eps,Globals::BCPvar_t bcpvar) ;

    int getinfo(std::string infostr) ;

    int getwon(int trgm) ;

    void initstate() ;

    void resetstate() ;

    void updsdep(float srcact,float spkwgain) ;

    void updsfac(float srcact,float spkwgain) ;

    void updbwsup(float srcact,float spkwgain,std::vector<float> &bwsup) ;

    void applywon() ;

    void setactive() ;  // i.e. non-silent

    void setsilent() ;

    void setsilent(float psilent) ;

    void updPc(float srcact,std::vector<float> &trgact,float ki,float kij) ;

    void updPz(float srcact,std::vector<float> &Zj,float ki,float tauzidt,float kp,float P = 0) ;

    void updPze(float srcact,std::vector<float> &Zj,float ki,float tauzidt,float ke,float kp) ;
    
    void updBW(std::vector<float> &Pj,float P,float ewgain,float iwgain) ;

    void updMic1(std::vector<float> &Pj,float P) ;

    void updMic2(std::vector<float> &Pj,float P) ;

    void updMic3() ;

    void prnp(FILE *outf = stdout) ;

} ;


class Pop;

class PopH;

class PrjH;

class Logger;

class Prj {

protected:

    static std::vector<Prj *> prjs;

    Pop *srcpop,*trgpop;
    PopH *srcpoph,*trgpoph;

    PrjH *prjh;

    int id,srcH,srcNh,srcN,srcM,srcU,srchoffs,srcnoffs,trgH,trgNh,trgN,trgM,trgU,trghoffs,trgnoffs;

    float pdens,wdens,tauconddt,cspeed,srcspkwgain,trgspkwgain,sp0,tausddt,tausfdt,bgain,ewgain,iwgain;

    float tmpstate;
    std::vector<std::vector<float> > tmpstateij; 
    std::vector<float> tmpstatei,tmpstatej;

    std::vector<int> idelay;

    Globals::BCPvar_t bcpvar;
    Globals::Prjarch_t prjarch;
    std::vector<float> Zj,Ej,Pj,Bj,delact,bwsup,cond;
    std::vector<std::vector<float> > locbwsup; // For OPENMP
    std::vector<float> trgact; // For MPI

    Globals::Pupdfn_t pupdfn_t;

    float tauzidt,tauzjdt,taupdt,tauedt,prn,psilent;
    float P,eps;

    std::vector<Connij *> connijs;
    bool updbw;

    Prj *cprj;

    long seed;
    std::mt19937_64 generator;
    std::uniform_real_distribution<float> uniformfloatdistr;
    std::uniform_int_distribution<int> uniformintdistr;

    float kbjhalf,taubdt;
    std::vector<float> kBj;

    int flpscr,nstoa,nntos;

    bool strcplon;
    int kspa,kspc,minage,updint;
    float pspa,pspc;

    friend class Connij;

    friend class Pop;

    friend class PPrj;

    friend class PrjH;

    friend class Logger;

    friend class PPrjH; // in ../psrc

    friend class PPrjR; // in ../psrc

 public:

    Prj(Pop *srcpop,Pop *trgpop,float pdens = 1,Globals::BCPvar_t bcpvar = Globals::FIXED,
	Globals::Prjarch_t bccarch = Globals::HPATCHY) ;

    Prj(PopH *srcpoph,Pop *trgpop,float pdens = 1,Globals::BCPvar_t bcpvar = Globals::FIXED,
	Globals::Prjarch_t bccarch = Globals::HPATCHY) ;

    Prj(PopH *srcpoph,Pop *trgpop,Prj *cprj,Globals::BCPvar_t bcpvar = Globals::FIXED,
	Globals::Prjarch_t bccarch = Globals::HPATCHY) ;

    // This is for use in psrc
    Prj(int srcH,int srcM,int srcU,int srcspkwgain,int trgH,int trgM,int trgU,int trgspkwgain,
	float pdens,Globals::BCPvar_t bcpvar = Globals::FIXED,
    	Globals::Prjarch_t bccarch = Globals::HPATCHY) ;

    static Prj *getprj(int id) ;

    void initparam() ;

    virtual void updidelay() ;

    void setdelays(float delaymat) ;

    void setdelays(std::vector<std::vector<float> > delaymat) ;

    void allocatestate() ;

    void configwon(float psilent = 0,bool pfixed = false) ;

    bool isconn(int srcn) ;

    int getwon(int srcn,int trgm = -1) ;

    void initstate() ;

    /// bool getinfo(std::string infostr,int &infoval) ;

    int getinfo(std::string infostr) ;

    void setsrchnoffs(int srchoffs,int srcnoffs) ;

    void settrghnoffs(int trghoffs,int trgnoffs) ;

    void setparam(std::string paramstr,float paramval) ;

    void setparam(Globals::Prjparam_t param_t,float parval) ;

    void setparams(std::string paramfilename) ;

    float getparam(Globals::Prjparam_t param_t) ;

    void setstate(std::string statestr,std::vector<float> statevec) ;

    void setstate(std::string statestr,std::vector<std::vector<float> > statemat,bool pfixed = true) ;

    void setstate(std::string statestr,float stateval,bool pfixed = true) ;

    float getstate(std::string statestr) ;

    std::vector<float> const &getstatei(std::string statestr) ;

    std::vector<float> const &getstatej(std::string statestr) ;

    std::vector<std::vector<float> > const &getstateij(std::string statestr) ;

    void fetchstatei(std::string statestr,std::vector<float> &statevec) ;

    void fetchstatej(std::string statestr,std::vector<float> &statevec) ;

    void fetchstate1(std::string statestr,std::vector<float> &statevec) ;

    void fetchstateij(std::string statestr,std::vector<std::vector<float> > &statevec) ;

    void fetchstateij(std::string statestr) ;

    void fetchstate(std::string statestr) ;

    void fwritestateval(std::string statestr,FILE *outf) ;

    void fwritestatevec(std::string statestr,FILE *outf) ;

    void fwritestatemat(std::string statestr,FILE *outf) ;

    void fwritestate(std::string statestr,FILE *outf) ;

    void fwritestate(std::string statestr,std::string filename) ;

    Logger *logstate(std::string statestr,std::string filename,int logstep = 1) ;

    void setpupdfn(Globals::Pupdfn_t pupdfn_t) ;

    long setseed(long seed = 0) ;

    long getseed() ;

    float nextfloat() ;

    int nextint() ;

    void resetstate() ;

    static void resetstateall() ;

    void resetbwsup() ;

    static void resetbwsupall() ;

    virtual void upddelact() ;

    void updsdep() ;

    void updsfac() ;

    virtual void propagate() ;

    void updPc() ;

    virtual void updZj() ;

    void updPz() ;

    void updPze() ;

    void updP() ;

    void updkbj() ;

    void updBW(bool force = false) ;

    void updMic(bool force = false) ;

    void updconnijs() ;

    void setstrcpl(bool on) ;

    bool getsilminmaxscore(float &scmin,int &scminidx,float &scmax,int &scmaxidx) ;

    bool getactminmaxscore(float &scmin,int &scminidx,float &scmax,int &scmaxidx) ;

    float getactmicscoremean() ;

    int getrndnew() ;

    bool qsiltoact() ;

    void qsiltoactn(int n,int updint) ;

    bool qnewtosil() ;

    virtual void update() ;

    static void updateall() ;
    
} ;

#endif // __Prj_included


