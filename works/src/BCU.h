/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __BCU_included
#define __BCU_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"
#include "PopH.h"

class BCU : public PopH {

 protected:

    friend class BCC;

    friend class Prj;

 public:

    BCU(int H,int M,Globals::Actfn_t actfn_t = Globals::BCP,Globals::Normfn_t normfn = Globals::FULL) ;

    void populate(int H,int M,Globals::Actfn_t actfn_t,Globals::Normfn_t normfn) ;

    virtual void fwritestate(std::string statestr,FILE *outf) ;

    virtual Logger *logstate(std::string statestr,std::string filename,int logstep = 1) ;

} ;

#endif // __BCU_included

