/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __PPopH_included
#define __PPopH_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"

class Pop;

class PopH;

class PPopH : public PopH {

protected:

    friend class PPopR;

    friend class PPopulation;

 public:

    PPopH(int H,int M,int U = 1,Globals::Actfn_t actfn_t = Globals::BCP,
	  Globals::Normfn_t normfn_t = Globals::FULL) ;

    void populate(int H,int M,int U,Globals::Actfn_t actfn_t,Globals::Normfn_t normfn_t) ;

    void sethnoffs(int hoffs,int noffs) ;

    void setmaxdelay(float maxdelay) ;

    void gatheract(std::vector<float> &ppact) ;

    void bwsuptocurr(float Esyn) ;

    void fwritestate(PIO *pio,std::string statestr) ;

} ;

#endif // __PPopH_included

