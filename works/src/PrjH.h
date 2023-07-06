/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __PrjH_included
#define __PrjH_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"

class PopH;

class Prj;

class PrjH {

 protected:

    static std::vector<PrjH *> prjhs;

    PopH *srcpoph,*trgpoph;

    Globals::Prjarch_t prjarch;

    PrjH *cprjh;

    float pdens;

    int id,srcM,trgM,srcH,trgH,srcNh,trgNh,srcN,trgN;

    std::vector<Prj *> prjs;

    std::vector<float> tmpstate,tmpstatej;

    std::vector<std::vector<float> > tmpstatei,tmpstateij;

    friend class Prj;

 public:

    PrjH(PopH *srcpoph,PopH *trgpoph,float pdens = 1,Globals::BCPvar_t bcpvar = Globals::FIXED,
	Globals::Prjarch_t prjharch = Globals::HPATCHY) ;

    PrjH(PopH *srcpoph,PopH *trgpoph,PrjH *cprjh,Globals::BCPvar_t bcpvar = Globals::FIXED,
	Globals::Prjarch_t prjharch = Globals::HPATCHY) ;

    // This is for use in psrc
    PrjH(int srcH,int srcM,int srcU,int srcspkwgain,int trgH,int trgM,int trgU,int trgspkwgain,
	 float pdens,Globals::BCPvar_t bcpvar = Globals::FIXED,
	 Globals::Prjarch_t prjarch = Globals::HPATCHY) ;

    // void configwon() ;

    Prj *getprj(int prjid) ;

    int getinfo(std::string infostr) ;

    virtual int getntpatch(int srcn) ;

    float getparam(Globals::Prjparam_t param_t) ;

    void setdelays(float delaymat) ;

    void setdelays(std::vector<std::vector<float> > delaymat) ;

    void setparam(std::string paramstr,float paramval) ;

    void setparam(Globals::Prjparam_t param_t,float paramval) ;

    void setstate(std::string statestr,std::vector<float> statevec) ;

    void setstate(std::string statestr,std::vector<std::vector<float> > statemat,bool pfixed = true) ;

    void setstate(std::string statestr,float stateval,bool pfixed = true) ;

    void setpupdfn(Globals::Pupdfn_t pupdfn_t) ;

    void setseed(long seed = 0) ;

    std::vector<long> getseedvec(std::vector<long> &seeds) ;

    void prnparam(std::string paramstr) ;

    void fetchstate(std::string statestr) ;

    void prnstate(std::string statestr) ;

    void fwritestate(std::string statestr,FILE *outf) ;

    void fwritestate(std::string statestr,std::string filename) ;

    std::vector<std::vector<float> > getstateij(std::string statestr) ;
    
    std::vector<float> getstatej(std::string statestr) ;

    void logstate(std::string statestr,std::string filename,int logstep = 1) ;

} ;

#endif // __PrjH_included

