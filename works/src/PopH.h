/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __PopH_included
#define __PopH_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"

class Pop;

class Prj;

class SNNPop;

class Logger;

class PopH {

 protected:

    static std::vector<PopH *> pophs;

    int id,H,M,U,Nh,N;

    std::vector<float> tmpstatevec,tmphstatevec;

    std::vector<Pop *> pops;

    friend class Pop;

    friend class Prj;

    friend class PPopR; // For psrc

    friend class PPopulation; // For psrc

 public:

    PopH(int H,int M,int U,Globals::Actfn_t actfn_t = Globals::BCP,Globals::Normfn_t normfn_t = Globals::FULL) ;

    PopH(int H,int M,Globals::Actfn_t actfn_t = Globals::BCP,Globals::Normfn_t normfn_t = Globals::FULL) ;

    PopH(bool nopopulate,int H,int M,int U) ;

    virtual void populate(int H,int M,int U,Globals::Actfn_t actfn_t,Globals::Normfn_t normfn_t) ;

    Pop *getpop(int h) ;

    void updinprjs(Prj *inprj) ;

    void updutprjs(Prj *utprj) ;

    bool getinfo(std::string infostr,int &infoval) ;

    int getinfo(std::string infostr) ;

    void setgeom(Globals::Geom_t geom,std::vector<float> cxyz = {0,0,0},float wid = 0.000500,
		 float spc = 0.000100) ;

    int getH(int i) ;

    int getU(int i) ;

    void setparam(std::string paramstr,float paramval) ;

    void setparam(std::string paramstr,std::vector<float> paramvec) ;

    virtual void setparam(Globals::Popparam_t param_t,float paramval) ;

    void setparam(Globals::Popparam_t param_t,std::vector<float> paramvec) ;

    virtual void setparamsfromfile(std::string paramfilename) ;

    float getparam(std::string paramstr) ;

    std::vector<float> getparamvec(std::string paramstr,std::vector<float> &paramvec) ;

    std::vector<float> const &getstate(std::string statestr) ;

    void setseed(long seed = 0) ;

    std::vector<long> getseedvec(std::vector<long> &seeds) ;

    void prnparam(std::string paramstr) ;

    void prngeom() ;

    void prnidandseed() ;

    void fwritegeom(std::string filename) ;

    void prnstate(std::string statestr) ;

    virtual void fwritestate(std::string statestr,FILE *outf) ;

    void fwritestate(std::string statestr,std::string filename) ;

    virtual Logger *logstate(std::string statestr,std::string filename,int logstep = 1) ;

    virtual void setinp(float inpval,int n = -1) ;

    virtual void setinp(std::vector<float> inpvec) ;

    void resetstate() ;

} ;

class SNNPopH : public PopH {

 public:

    SNNPopH(int H,int M,int U = 1,Globals::Actfn_t actfn_t = Globals::BCP) ;
    
    virtual void populate(int H,int M,int U,Globals::Actfn_t actfn_t);

    virtual void setparamsfromfile(std::string paramfilename) ;

} ;

#endif // __PopH_included

