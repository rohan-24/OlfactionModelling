/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#include "stdio.h"

#include "Globals.h"
#include "Pop.h"
#include "HCU.h"
#include "PopH.h"
#include "BCU.h"
#include "Prj.h"
#include "PrjH.h"
#include "Logger.h"

using namespace std;
using namespace Globals;


vector<Logger *> Logger::loggers;

Logger::Logger(Pop *pop,string statestr,string logfile,int logstep) {

    this->pop = pop;
    this->hcu = NULL;
    this->poph = NULL;
    this->bcu = NULL;
    this->prj = NULL;
    this->prjh = NULL;

    this->statestr = statestr;
    outf = fopen(logfile.c_str(),"w");
    on = true;
    this->logstep = logstep;
    c0 = 0;
    cn = pop->getinfo("N");

    loggers.push_back(this);

}


Logger::Logger(HCU *hcu,string statestr,string logfile,int logstep) {

    this->pop = NULL;
    this->hcu = hcu;
    this->poph = NULL;
    this->bcu = NULL;
    this->prj = NULL;
    this->prjh = NULL;

    this->statestr = statestr;
    outf = fopen(logfile.c_str(),"w");
    on = true;
    this->logstep = logstep;
    c0 = 0;
    cn = hcu->getinfo("N");

    loggers.push_back(this);

}


Logger::Logger(PopH *poph,string statestr,string logfile,int logstep) {

    this->pop = NULL;
    this->hcu = NULL;
    this->poph = poph;
    this->bcu = NULL;
    this->prj = NULL;
    this->prjh = NULL;

    this->statestr = statestr;
    outf = fopen(logfile.c_str(),"w");
    on = true;
    this->logstep = logstep;

    loggers.push_back(this);

}


Logger::Logger(BCU *bcu,string statestr,string logfile,int logstep) {

    this->pop = NULL;
    this->hcu = NULL;
    this->poph = NULL;
    this->bcu = bcu;
    this->prj = NULL;
    this->prjh = NULL;
    this->prjh = NULL;

    this->statestr = statestr;
    outf = fopen(logfile.c_str(),"w");
    on = true;
    this->logstep = logstep;

    loggers.push_back(this);

}


Logger::Logger(Prj *prj,string statestr,string logfile,int logstep) {

    this->pop = NULL;
    this->hcu = NULL;
    this->poph = NULL;
    this->bcu = NULL;
    this->prj = prj;
    this->prjh = NULL;

    rn = prj->srcN;
    cn = prj->trgN;

    this->statestr = statestr;
    outf = fopen(logfile.c_str(),"w");
    on = true;
    this->logstep = logstep;

    loggers.push_back(this);

}


Logger::Logger(PrjH *prjh,string statestr,string logfile,int logstep) {

    this->pop = NULL;
    this->hcu = NULL;
    this->poph = NULL;
    this->bcu = NULL;
    this->prj = NULL;
    this->prjh = prjh;

    this->statestr = statestr;
    outf = fopen(logfile.c_str(),"w");
    on = true;
    this->logstep = logstep;

    loggers.push_back(this);

}


void Logger::selectc(int c0,int cn) {

    if (pop==NULL) error("Logger::selectc","Only valid for 'pop'");

    if (c0<0 or cn<1 or this->cn<c0+cn) error("Logger::selectc","Illegal c0,cn");

    this->c0 = c0; this->cn = cn;

}


void Logger::select(vector<int> limvec) {

    if (limvec.size()==2) 

	selectc(limvec[0],limvec[1]);

    else if (limvec.size()==4)

	selectrc(limvec[0],limvec[1],limvec[2],limvec[3]);

    else error("Logger::selectc","Illegal 'i0n' size");

}


void Logger::selectrc(int r0,int c0,int rn,int cn) {

    if (prj==NULL) error("Logger::select","Only valid for 'prj'");

    if (r0<0 or rn<1 or this->rn<r0 + rn) error("Logger::select","Illegal r0,rn");

    if (c0<0 or cn<1 or this->cn<c0 + cn) error("Logger::select","Illegal c0,cn");

    this->r0 = r0; this->rn = rn;

    this->c0 = c0; this->cn = cn;

}


void Logger::dolog() {

    if (not on or simstep%logstep!=0) return;

    if (pop!=NULL)
	   pop->fwritestate(statestr,outf);
    else if (hcu!=NULL)
    	hcu->fwritestate(statestr,outf);
    else if (poph!=NULL)
    	poph->fwritestate(statestr,outf);
    else if (bcu!=NULL)
    	bcu->fwritestate(statestr,outf);
    else if (prj!=NULL)
    	prj->fwritestate(statestr,outf);
    else if (prjh!=NULL)
    	prjh->fwritestate(statestr,outf);
}


void Logger::start() { on = true; }


void Logger::stop() { on = false; }


void Logger::close() { fclose(outf); }


void Logger::dologall() {

    for (size_t g=0; g<loggers.size(); g++) loggers[g]->dolog();  

}

void Logger::closeall() {

    for (size_t g=0; g<loggers.size(); g++) loggers[g]->close();

    loggers.clear();

}
