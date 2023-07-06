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
#include "Pop.h"
#include "PopH.h"
#include "Prj.h"
#include "Logger.h"
#include "PrjH.h"

using namespace std;
using namespace Globals;


vector<PrjH *> PrjH::prjhs;

PrjH::PrjH(PopH *srcpoph,PopH *trgpoph,float pdens,BCPvar_t bcpvar,Prjarch_t prjarch) {

    if (setupdone) error("PrjH::PrjH","Not allowed when setupdone");

    this->srcpoph = srcpoph;

    this->trgpoph = trgpoph;

    this->prjarch = prjarch;

    this->cprjh = NULL;

    this->pdens = pdens;

    srcpoph->getinfo("M",srcM);

    trgpoph->getinfo("M",trgM);

    srcpoph->getinfo("H",srcH);

    trgpoph->getinfo("H",trgH);

    srcpoph->getinfo("Nh",srcNh);

    trgpoph->getinfo("Nh",trgNh);

    srcpoph->getinfo("N",srcN);

    trgpoph->getinfo("N",trgN);

    Pop *trgpop;

    for (int trgh=0; trgh<trgH; trgh++) {

     	trgpop = trgpoph->getpop(trgh);

    	prjs.push_back(new Prj(srcpoph,trgpop,pdens,bcpvar,prjarch));

     	prjs.back()->prjh = this;

    	prjs.back()->settrghnoffs(trgh,trgh * trgNh);

    	prjs.back()->configwon(0);

    }

    id = prjhs.size();

    prjhs.push_back(this);

}


PrjH::PrjH(PopH *srcpoph,PopH *trgpoph,PrjH *cprjh,BCPvar_t bcpvar,Prjarch_t prjarch) {

    if (setupdone) error("PrjH::PrjH","Not allowed when setupdone");

    id = prjhs.size();

    setseed(id + 1);

    this->srcpoph = srcpoph;

    this->trgpoph = trgpoph;

    this->prjarch = prjarch;

    this->cprjh = cprjh;

    this->pdens = pdens;

    srcpoph->getinfo("M",srcM);

    trgpoph->getinfo("M",trgM);

    srcpoph->getinfo("H",srcH);

    trgpoph->getinfo("H",trgH);

    srcpoph->getinfo("Nh",srcNh);

    trgpoph->getinfo("Nh",trgNh);

    srcpoph->getinfo("N",srcN);

    trgpoph->getinfo("N",trgN);

    Pop *trgpop;

    for (int trgh=0; trgh<trgH; trgh++) {

    	trgpop = trgpoph->getpop(trgh);

    	prjs.push_back(new Prj(srcpoph,trgpop,cprjh->prjs[trgh],bcpvar,prjarch));

     	prjs.back()->prjh = this;

    	prjs.back()->settrghnoffs(trgh,trgh * trgNh);

    	prjs.back()->configwon(0);

    }

    prjhs.push_back(this);

}


// This is for use in psrc
PrjH::PrjH(int srcH,int srcM,int srcU,int srcspkwgain,int trgH,int trgM,int trgU,int trgspkwgain,
	   float pdens,BCPvar_t bcpvar,Prjarch_t prjarch) {

    if (setupdone) error("PrjH::PrjH","Not allowed when setupdone");

    id = prjhs.size();

    setseed(id + 1);

    this->srcpoph = NULL;

    this->trgpoph = NULL;

    this->prjarch = prjarch;

    this->pdens = pdens;

    this->srcH = srcH;

    this->srcM = srcM;

    this->trgH = trgH;

    this->trgM = trgM;

    srcNh = srcM * srcU;

    trgNh = trgM * trgU;

    srcN = srcH * srcM * srcU;

    trgN = trgH * trgM * trgU;

    prjhs.push_back(this);

}


Prj *PrjH::getprj(int prjid) {

    if (prjid<0 or prjs.size()<=prjid) error("PrjH::getprj","Illegal prjid: " + to_string(prjid));

    return prjs[prjid];

}


int PrjH::getinfo(string infostr) {

    if (infostr=="id") return id;
    else if (infostr=="srcid")
	return srcpoph->getinfo("id");
    else if (infostr=="trgid") return	trgpoph->getinfo("id");
    else if (infostr=="srcN") return srcpoph->getinfo("N");
    else if (infostr=="trgN") return trgpoph->getinfo("N");
    else if (infostr=="nprj") return prjs.size();
    else if (infostr=="Patch_n" or infostr=="npatch") {
	int P_n = 0;
	for (size_t p=0; p<prjs.size(); p++) P_n += prjs[p]->getinfo("Patch_n");
	return P_n;
    } else if (infostr=="W_n" or infostr=="nw") {
	int W_n = 0;
	for (size_t p=0; p<prjs.size(); p++) W_n += prjs[p]->getinfo("W_n");
	return W_n;
    } else if (infostr=="bcpvar")
	return prjs[0]->getinfo("bcpvar");
    else

	error("PrjH::getinfo","No such 'infostr': " + infostr);
    
    return 0;

}


int PrjH::getntpatch(int srcn) {

    int npa = 0;

    if (srcn<0 or srcN<=srcn) error("PrjH::getntpatch","Illegal 'srcn'");

       for (size_t p=0; p<prjs.size(); p++) npa += prjs[p]->connijs[srcn] != NULL;

    return npa;

}


void PrjH::setdelays(float delay) {

    for (size_t c=0; c<prjs.size(); c++) prjs[c]->setdelays(delay);

}


void PrjH::setdelays(vector<vector<float> > delaymat) {

    for (size_t c=0; c<prjs.size(); c++) prjs[c]->setdelays(delaymat);

}


void PrjH::setparam(std::string paramstr,float paramval){

    for (size_t c=0; c<prjs.size(); c++) prjs[c]->setparam(paramstr,paramval);

}


void PrjH::setparam(Prjparam_t param_t,float paramval) {

    for (int c=0; c<prjs.size(); c++) prjs[c]->setparam(param_t,paramval);

}


float PrjH::getparam(Prjparam_t param_t) { return prjs[0]->getparam(param_t); }


void PrjH::setstate(string statestr,vector<float> statevec) {

    // if (statevec.size()!=srcN) { std::cout<<"statevec Size: "<<statevec.size()<<" srcN: "<<srcN<<std::endl; 
    // error("PrjH::setstate","statevec size mismatch"); }

    for (size_t c=0; c<prjs.size(); c++) prjs[c]->setstate(statestr,statevec);

}


void PrjH::setstate(string statestr,vector<vector<float> > statemat,bool pfixed) {

    for (size_t c=0; c<prjs.size(); c++) prjs[c]->setstate(statestr,statemat,pfixed);

}


void PrjH::setstate(string statestr,float stateval,bool pfixed) {

    if (statestr=="Bj")
        std::cout<<"Statestr: "<<statestr<<" Stateval: "<<stateval<<" pfixed:"<<pfixed<<std::endl;
    for (size_t c=0; c<prjs.size(); c++) prjs[c]->setstate(statestr,stateval,pfixed);

}


void PrjH::setpupdfn(Pupdfn_t pupdfn_t) {

    for (size_t c=0; c<prjs.size(); c++) prjs[c]->setpupdfn(pupdfn_t);

}


void PrjH::setseed(long seed) {

    long newseed;

    if (seed==0) newseed = random_device{}();

    for (size_t p=0; p<prjs.size(); p++) {

	if (seed==0)

	    prjs[p]->setseed(newseed);

	else

	    prjs[p]->setseed(seed + 23 * p);


    }
}


vector<long>  PrjH::getseedvec(vector<long> &seedvec) {

    if (seedvec.size()<prjs.size())

	seedvec = vector<long>(prjs.size());

    for (int c=0; c<prjs.size(); c++) seedvec[c] = prjs[c]->getseed();

    return seedvec;

}


void PrjH::fetchstate(string statestr) {

    if (statestr=="P") {

	if (tmpstate.size()==0) tmpstate = vector<float>(prjs.size());

	for (size_t c=0; c<prjs.size(); c++) tmpstate[c] = prjs[c]->getstate(statestr);

    } else if (statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="Bj" or statestr=="kBj" or
	       statestr=="bwsup" or statestr=="cond") {

	if (tmpstatej.size()==0) tmpstatej = vector<float>(trgN,0);
	
	for (size_t c=0; c<prjs.size(); c++) prjs[c]->fetchstatej(statestr,tmpstatej);

    } else if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="delact" or statestr=="Mic" or
	        statestr=="Sil" or statestr=="sdep" or statestr=="sfac") {

	if (tmpstatei.size()!=prjs.size() or tmpstatei[0].size()!=srcN)

	    tmpstatei = vector<vector<float> > (prjs.size(),vector<float>(srcN,0));

	else for (size_t r=0; r<prjs.size(); r++) fill(tmpstatei[r].begin(),tmpstatei[r].end(),0);

	for (size_t c=0; c<prjs.size(); c++) prjs[c]->fetchstatei(statestr,tmpstatei[c]);

    } else if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won") {

	if (tmpstateij.size()!=srcN or tmpstateij[0].size()!=trgN)

	    tmpstateij = vector<vector<float> > (srcN,vector<float>(trgN,0));

	else for (int r=0; r<srcN; r++) fill(tmpstateij[r].begin(),tmpstateij[r].end(),0);
	    
	for (size_t c=0; c<prjs.size(); c++) prjs[c]->fetchstateij(statestr,tmpstateij);

    } else error("PrjH::fetchstate","Illegal statestr: " + statestr);

}


void PrjH::prnstate(string statestr) {

    fetchstate(statestr);

    if (statestr=="P")

	prnvec(tmpstate);

    else if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="delact" or statestr=="Mic" or
	     statestr=="Sil")

	prnmat(tmpstatei);

    else if (statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="Bj" or statestr=="kBj" or
	     statestr=="bwsup" or statestr=="cond" or statestr=="sdep" or statestr=="sfac")

	prnvec(tmpstatej);

    else if (statestr=="sdep" or statestr=="sfac")

	prnmat(tmpstatei);

    else if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won")

	prnmat(tmpstateij);

    else error("PrjH::prnstate","Illegal statestr: " + statestr);

}


void PrjH::fwritestate(string statestr,FILE *outf) {

    fetchstate(statestr);

    if (statestr=="P")

	fwriteval(tmpstate,outf);

    else if (statestr=="Zi" or statestr=="Ei" or statestr=="Pi" or statestr=="delact" or statestr=="Mic" or
	     statestr=="Sil" or statestr=="sdep" or statestr=="sfac") {

	fwritemat(tmpstatei,outf);

    } else if (statestr=="Zj" or statestr=="Ej" or statestr=="Pj" or statestr=="Bj" or statestr=="kBj" or
	     statestr=="bwsup" or statestr=="cond")

	fwriteval(tmpstatej,outf);

    else if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won") {

	fwritemat(tmpstateij,outf);

    } else error("PrjH::fwritestate","Illegal statestr: " + statestr);

}

vector<vector<float> > PrjH::getstateij(string statestr) {

    fetchstate(statestr);

    if (statestr=="Eij" or statestr=="Pij" or statestr=="Wij" or statestr=="Won") {

    return tmpstateij;

    } else error("PrjH::fwritestate","Illegal statestr: " + statestr);

}

vector<float> PrjH::getstatej(string statestr) {

    fetchstate(statestr);

    if (statestr=="Bj" or statestr=="Pj") {

    return tmpstatej;

    } else error("PrjH::fwritestate","Illegal statestr: " + statestr);

}

void PrjH::fwritestate(string statestr,string filename) {

    FILE *outf = fopen(filename.c_str(),"w");

    fwritestate(statestr,outf);

    fclose(outf);

}


void PrjH::logstate(std::string statestr,std::string filename,int logstep) {

    new Logger(this,statestr,filename,logstep);

}
