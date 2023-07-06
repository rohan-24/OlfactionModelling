/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#include <stdlib.h>
#include <vector>
#include <string>
#include <random>
#include <limits>

#include "Globals.h"
#include "Parseparam.h"
#include "HCU.h"
#include "PopH.h"
#include "Logger.h"
#include "BCU.h"

using namespace std;
using namespace Globals;


BCU::BCU(int H,int M,Actfn_t actfn_t,Normfn_t normfn) : PopH(true,H,M,1) {

    populate(H,M,actfn_t,normfn);

    setseed(id + 1);

}


void BCU::populate(int H,int M,Actfn_t actfn_t,Normfn_t normfn) {

    for (int h=0; h<H; h++) {

	pops.push_back(new HCU(M,actfn_t,normfn));

	pops.back()->sethnoffs(h);

	pops.back()->poph = this;
	    
    }
}


void BCU::fwritestate(std::string statestr,FILE *outf) {

    for (int h=0; h<H; h++) ((HCU *)pops[h])->fwritestate(statestr,outf);

}


Logger *BCU::logstate(std::string statestr,std::string filename,int logstep) {

    return new Logger(this,statestr,filename,logstep);

}
