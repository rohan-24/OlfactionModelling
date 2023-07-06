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
#include "Logger.h"
#include "PopH.h"

using namespace std;
using namespace Globals;


vector<PopH *> PopH::pophs;


PopH::PopH(int H,int M,int U,Actfn_t actfn_t,Normfn_t normfn_t) {

    if (setupdone) error("PopH::PopH","Not allowed when setupdone");

    this->H = H;

    this->M = M;

    id = pophs.size();

    this->U = U;

    Nh = M * U;

    N = H * Nh;

    populate(H,M,U,actfn_t,normfn_t);

    pophs.push_back(this);

}


PopH::PopH(int H,int M,Actfn_t actfn_t,Normfn_t normfn_t) {

    if (setupdone) error("PopH::PopH","Not allowed when setupdone");

    this->H = H;

    this->M = M;

    id = pophs.size();

    this->U = 1;

    Nh = M * U;

    N = H * Nh;

    populate(H,M,U,actfn_t,normfn_t);

    pophs.push_back(this);

}


PopH::PopH(bool nopopulate,int H,int M,int U) {

    if (setupdone) error("PopH::PopH","Not allowed when setupdone");

    this->H = H;

    this->M = M;

    id = pophs.size();

    this->U = U;

    Nh = M * U;

    N = H * Nh;

    pophs.push_back(this);

}


void PopH::populate(int H,int M,int U,Actfn_t actfn_t,Normfn_t normfn_t) {

    for (int h=0; h<H; h++) {
	
	switch (actfn_t) {

	case LIN:
	case LOG:
	case SIG:
	case SPKLIN:
	case SPKLOG:
	case SPKSIG:
	case AREG:
    	    pops.push_back(new Pop(M,U,actfn_t,normfn_t));
	    break;

	case EXP:
	case SPKEXP:
    	    pops.push_back(new ExpPop(M,U,actfn_t,normfn_t));
	    break;

	case BCP:
	case SPKBCP:
    	    pops.push_back(new BCPPop(M,U,actfn_t,normfn_t));
	    break;

	case ALIF:
	case AdEx:
	case AdExS:
    	    pops.push_back(new SNNPop(M,U,actfn_t));
	    break;

	}
	
	pops.back()->poph = this;

	pops.back()->sethnoffs(h);

    }
}


Pop *PopH::getpop(int h) {

    if (h<0 or H<=h) error("PopH::getpop","Illegal h<0 or H<=h: " + to_string(h));
	
    return pops[h];

}


void PopH::updinprjs(Prj *inprj) {

    for (size_t p=0; p<pops.size(); p++) pops[p]->inprjs.push_back(inprj);

}


void PopH::updutprjs(Prj *utprj) {

    for (size_t p=0; p<pops.size(); p++) pops[p]->utprjs.push_back(utprj);

}


bool PopH::getinfo(string infostr,int &infoval) {

    if (not (infostr=="H" or infostr=="M" or infostr=="N" or infostr=="id" or infostr=="U" or infostr=="Nh" or
	     infostr=="spkwgain"))

	error("PopH::getinfo","No such 'infostr': " + infostr);

    if (infostr=="H") {
	infoval = H;
	return true;
    } else if (infostr=="M") {
	infoval = M;
	return true;
    } else if (infostr=="N") {
	infoval = N;
	return true;
    } else if (infostr=="id") {
	infoval = id;
	return true;
    } else if (infostr=="U") {
	infoval = U;
	return true;
    } else if (infostr=="Nh") {
	infoval = Nh;
	return true;
    } else if (infostr=="spkwgain") {
	infoval = pops[0]->getparam("spkwgain");
	return true;
    }
    
    return false;

}


int PopH::getinfo(string infostr) {

    int infoval;

    getinfo(infostr,infoval);

    return infoval;

}


void PopH::setgeom(Geom_t geom,vector<float> cxyz,float wid,float spc) {

    // Every Pop/HCU has an xyz position

    if (cxyz.size()!=3) error("PopH::setgeom","Illegal cxyz dimension: " + to_string(cxyz.size()));

    int C,R;
    vector<vector<Point_2D> > coord2Ds_h;
    vector<float> xyz(3);

    switch (geom) {

    case RND2D:

	for (size_t h=0; h<pops.size(); h++) {

	    for (int d=0; d<3; d++) xyz[d] = wid * (pops[h]->nextfloat() - 0.5) + cxyz[d];

	    pops[h]->setxyz(xyz);

	}

	break;

    case REC2D:

	C = sqrt(pops.size());

	R = ceil((float)pops.size()/C);

	for (size_t h=0,x=0,y=0; h<pops.size(); h++) {

	    y = h/C;

	    x = h%C ;

	    xyz[0] = (x - C/2.) * spc + cxyz[0];

	    xyz[1] = (y - C/2.) * spc + cxyz[1];

	    xyz[2] = cxyz[2];

	    pops[h]->setxyz(xyz);

	}

	break;

    case HEX2D:

	coord2Ds_h = hexgridhm(H,1,spc);

	for (size_t h=0; h<pops.size(); h++) {

	    xyz[0] = cxyz[0] + coord2Ds_h[h][0].x;

	    xyz[1] = cxyz[1] + coord2Ds_h[h][0].y;

	    xyz[2] = cxyz[2];

	    pops[h]->setxyz(xyz);

	}

	break;

    default: error("PopH::setgeom","No such geom");

    }
}


int PopH::getH(int i) {

    return i/(M*U);

}
		     


int PopH::getU(int i) {

    return i/U;

}


void PopH::setparam(string paramstr,float paramval) {

    for (int h=0; h<H; h++) pops[h]->setparam(paramstr,paramval);

}


void PopH::setparam(string paramstr,vector<float> paramvec) {

    for (int h=0; h<H; h++) pops[h]->setparam(paramstr,paramvec[h]);

}


void PopH::setparam(Popparam_t param_t,float paramval) {

    for (int h=0; h<H; h++) pops[h]->setparam(param_t,paramval);

}


void PopH::setparam(Popparam_t param_t,vector<float> paramvec) {

    for (int h=0; h<H; h++) pops[h]->setparam(param_t,paramvec[h]);

}


void PopH::setparamsfromfile(string paramfilename) {}


float PopH::getparam(string paramstr) {

    return pops[0]->getparam(paramstr);

}


vector<float> PopH::getparamvec(string paramstr,vector<float> &paramvec) {

    if (paramvec.size()<H) error("PopH::getparamvec","'paramvec' length mismatch: " + to_string(paramvec.size()));
    
    for (int h=0; h<H; h++) paramvec[h] = pops[h]->getparam(paramstr);
    
    return paramvec;

}


vector<float> const &PopH::getstate(string statestr) {

    if (tmpstatevec.size()!=N) tmpstatevec = vector<float>(N,0);

    for (int h=0; h<H; h++) pops[h]->cpystatetovec(statestr,tmpstatevec);

    return tmpstatevec;

}


void PopH::setinp(float inpval,int n) {

    if (n<-1 or N<=n) error("PopH::setinp","Illegal n: " + to_string(n));

    for (int h=0; h<H; h++) pops[h]->setinp(inpval,n);
 
}


void PopH::setinp(std::vector<float> inpvec) {

    for (int h=0; h<H; h++) pops[h]->setinp(inpvec);

}


void PopH::setseed(long seed) {

    long newseed;

    if (seed==0) newseed = random_device{}();

    for (size_t p=0; p<pops.size(); p++)

	if (seed==0)

	    pops[p]->setseed(newseed);

	else

	    pops[p]->setseed(seed + 17 * p);

}


vector<long>  PopH::getseedvec(vector<long> &seedvec) {

    if (seedvec.size()<H) seedvec = vector<long>(H);	    

    for (int h=0; h<H; h++) seedvec[h] = pops[h]->getseed();

    return seedvec;

}


void PopH::prnidandseed() {

    for (int h=0; h<H; h++) printf("%d %ld\n",pops[h]->id,pops[h]->getseed());

}


void PopH::prnparam(string paramstr) {

    for (int h=0; h<H; h++) pops[h]->prnparam(paramstr);

}



void PopH::prngeom() {

    for (size_t h=0; h<pops.size(); h++)

	prnvec(pops[h]->getxyz(),3,6);

}


void PopH::fwritegeom(string filename) {

    FILE *outfp = fopen(filename.c_str(),"w");

    for (size_t h=0; h<pops.size(); h++)

	fwriteval(pops[h]->getxyz(),outfp);

    fclose(outfp);
}


void PopH::prnstate(string statestr) {

    prnvec(getstate(statestr));

}


void PopH::fwritestate(std::string statestr,FILE *outf) {

    for (int h=0; h<H; h++) pops[h]->fwritestate(statestr,outf);

}


void PopH::fwritestate(string statestr,string filename) {

    FILE *outf = fopen(filename.c_str(),"w");

    fwritestate(statestr,outf);

    fclose(outf);

}


Logger *PopH::logstate(std::string statestr,std::string filename,int logstep) {

    return new Logger(this,statestr,filename,logstep);

}


void PopH::resetstate() {

    for (int h=0; h<H; h++) pops[h]->resetstate();

}


SNNPopH::SNNPopH(int H,int M,int U,Actfn_t actfn_t) : PopH(H,M,U,true) {

    if (actfn_t==BCP or actfn_t==LIN or actfn_t==LOG or actfn_t==EXP or actfn_t==SPKBCP)
	error("SNNPopH;;SNNPopH","Illegal actfn_t");

    populate(H,M,U,actfn_t);

}


void SNNPopH::setparamsfromfile(string paramfilename) {

    for (size_t p=0; p<pops.size(); p++) pops[p]->setparamsfromfile(paramfilename);
	
}


void SNNPopH::populate(int H,int M,int U,Actfn_t actfn_t) {

    for (int h=0; h<H; h++) {
	
	pops.push_back(new SNNPop(M,U,actfn_t));
	
	pops.back()->poph = this;

	pops.back()->sethnoffs(h);

    }
}


