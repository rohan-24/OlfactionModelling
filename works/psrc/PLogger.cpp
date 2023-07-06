/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#include "stdio.h"

#include "pbcpnnsim.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;

vector<PLogger *> PLogger::ploggers;

PLogger::PLogger(PPopulation *ppopulation,string statestr,string logfile,int logstep) {

    this->ppopulation = ppopulation;

    this->pprojection = NULL;

    if (statestr=="dsupmax" or statestr=="expdsupsum")

	pio = new PIO(ppopulation->H);

    else

	pio = new PIO(ppopulation->N);

    pio->pfopen(logfile);

    this->statestr = statestr;
    on = true;
    this->logstep = logstep;

    ploggers.push_back(this);

}


PLogger::PLogger(PProjection *pprojection,string statestr,string logfile,int logstep) {

    this->ppopulation = NULL;

    this->pprojection = pprojection;

    if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="Mic" or statestr=="Sil" or
	statestr=="Age" or statestr=="delact")

    	pio = new PIO(pprojection->srcN * pprojection->trgH);

    else if (statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="Bj" or statestr=="bwsup" or
	     statestr=="cond")

    	pio = new PIO(pprojection->trgN);

    else if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won")

    	pio = new PIO(pprojection->srcN * pprojection->trgN);

    pio->pfopen(logfile);

    this->statestr = statestr;

    on = true;

    this->logstep = logstep;

    ploggers.push_back(this);

}


void PLogger::dolog() {

    if (not on or simstep%logstep!=0) return;

    if (ppopulation!=NULL)

    	ppopulation->fwritestate(pio,statestr);

    else if (pprojection!=NULL)

    	pprojection->fwritestate(pio,statestr);

    pio->wstep++;

}


void PLogger::start() { on = true; }


void PLogger::stop() { on = false; }


void PLogger::dologall() {

    for (size_t g=0; g<ploggers.size(); g++) ploggers[g]->dolog();

}


void PLogger::closeall() {

    PIO::pfcloseall();

}
