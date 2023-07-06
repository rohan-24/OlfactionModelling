/*

  Author: Anders Lansner

  Copyright (c) 2020 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#include <stdlib.h>
#include <vector>
#include <string>
#include <random>
#include <algorithm>    // std::random_shuffle

#include "Globals.h"
#include "DataFactory.h"

using namespace std;
using namespace Globals;


vector<vector<float> > DataFactory::getpats(vector<vector<float> > pats,int npat,int r0) {

    if (r0<0) error("DataFactory::DataFactory","r0<0");

    if (pats.size()<r0 + npat) error("DataFactory::DataFactory","pats.size()<r0 + npat");

    vector<vector<float> > tmpats;

    for (int p=r0; p<r0 + npat; p++) tmpats.push_back(pats[p]);

    return tmpats;

}


DataFactory::DataFactory(int H,int M,int U) {

    if (U!=1) error("DataFactory::DataFactory","U!=1 not yet implemented");

    this->H = H; this->M = M; this->U = U;

    N = H * M * U;

    nrow = ncol = 0;

    datapath = "";

    setshseed(0);

}


vector<vector<float> > DataFactory::mkpats(int npat,Pat_t pat_t,float p1,float p0,int verbosity) {

    vector<float> pat;
    vector<vector<float> > pats;

    if (p0<0) p0 = (1-p1)/(M-1);

    for (int p=0; p<npat; p++) {

	pat = vector<float>(H*M,0);

	for (int h=0; h<H; h++) {
	    
	    switch (pat_t) {

	    case ORTHO:

		for (int m=h*M; m<(h+1)*M; m++) pat[m] = p0;

		pat[h*M + p%M] = p1;

		break;

	    case RND01:

		for (int m=h*M; m<(h+1)*M; m++) pat[m] = p0;

		pat[h*M + gnextint()%M] = p1;

		break;

	    case NOHRND01:

		for (int n=0; n<H*M; n++) pat[n] = p0;

		for (int h=0,i; h<H; h++) {

		    i = gnextint()%(H*M);

		    while (pat[i]==p1) i = gnextint()%(H*M);

		    pat[i] = p1;

		}

		break;

	    case RND:

		for (int m=h*M; m<(h+1)*M; m++) pat[m] = gnextfloat();

		break;

	    default: error("Globals::mkpat","No such pat_t");

	    }

	}

	pats.push_back(pat);

    }

    nrow = 1;

    ncol = N;

    if (verbosity>2) {

	for (size_t r=0; r<pats.size(); r++) prnvec(pats[r],1);

	printf("\n");

    }

    return pats;

}


void DataFactory::setdatapath(string datapath) { this->datapath = datapath; }


void DataFactory::selectdata2() {

    if (trdata[0].size()%2!=0) error("DataFactory::selectdata2","Illegal image dimension");

    vector<vector<float> > trdata1(ntrpat,vector<float>(trdata[0].size()/2,0));

    vector<vector<float> > tedata1(ntepat,vector<float>(tedata[0].size()/2,0));

    for (size_t p=0; p<trdata.size(); p++)

	for (size_t i=0; i<trdata1[p].size(); i++)

	    trdata1[p][i] = trdata[p][2*i+1];

    trdata = trdata1;

    for (size_t p=0; p<tedata.size(); p++)

	for (size_t i=0; i<tedata1[p].size(); i++)

	    tedata1[p][i] = tedata[p][2*i+1];

    tedata = tedata1;

    ncol = ncol/2;

}


void DataFactory::expanddata1() {

    vector<vector<float> > trdata2(ntrpat,vector<float>(trdata[0].size()*2,0));

    vector<vector<float> > tedata2(ntepat,vector<float>(tedata[0].size()*2,0));

    for (size_t p=0; p<trdata.size(); p++)

	for (size_t i=0; i<trdata[p].size(); i++)

	    { trdata2[p][2*i] = 1 - trdata[p][i]; trdata2[p][2*i+1] = trdata[p][i]; }

    trdata = trdata2;

    for (size_t p=0; p<tedata.size(); p++)

	for (size_t i=0; i<tedata[p].size(); i++)

	    { trdata2[p][2*i] = 1 - trdata[p][i]; trdata2[p][2*i+1] = trdata[p][i]; }

    tedata = tedata2;

    ncol = ncol*2;

}


long DataFactory::setshseed(long newshseed) {

    if (newshseed==0) newshseed = random_device{}();

    shseed = newshseed%10000000;

    if (shseed<0) shseed = -shseed;

    shgenerator.seed(shseed);

    return shseed;

}


void DataFactory::getshuffledidx(vector<int> &idxvec) {

    std::shuffle(idxvec.begin(),idxvec.end(),shgenerator);

}

void DataFactory::cutout2(int r0,int rn,int c0,int cn) {

    if (rn<0) rn = nrow; if (cn<0) cn = ncol;

    if (r0<0) error("DataFactory::cutout2","Illegal r0<0");
    else if (rn>nrow) error("DataFactory::cutout2","Illegal rn>nrow");
    if (rn<=r0) error("DataFactory::cutout2","Illegal rn<=r0");
    if (c0<0) error("DataFactory::cutout2","Illegal c0<0");
    else if (cn>ncol) error("DataFactory::cutout2","Illegal cn>ncol");
    if (cn<=c0) error("DataFactory::cutout2","Illegal cn<=c0");

    nrow = rn - r0;

    ncol = cn - c0;

    vector<vector<float> >
	newtrdata(ntrpat,vector<float>(nrow*ncol,0)),
	newtedata(ntepat,vector<float>(nrow*ncol,0));

    for (int p=0; p<ntrpat; p++)

	for (int r=r0; r<rn; r++)

	    for (int c=c0; c<cn; c++)

		newtrdata[p][(r - r0) * ncol + c - c0] = trdata[p][r*ncol + c];

    for (int p=0; p<ntepat; p++)

	for (int r=r0; r<rn; r++)

	    for (int c=c0; c<cn; c++)

		newtedata[p][(r - r0) * ncol + c - c0] = tedata[p][r*ncol + c];
    
    trdata = newtrdata;

    tedata = newtedata;



}


void DataFactory::loadMNIST(int ntrpat,int ntepat) {

    // This is not loading originally formatted MNIST, but a preformatted version

    if (ntrpat<0) { ntrpat = 60000; this->ntrpat = ntrpat; }
    if (ntepat<0) { ntepat = 10000; this->ntepat = ntepat; }

    nrow = 28; ncol = 28;

    trdata = readpats(ntrpat,nrow*ncol,datapath + "mnist_60k_trainimg.bin");

    tedata = readpats(ntepat,nrow*ncol,datapath + "mnist_10k_testimg.bin");

    trlbl = readpats(ntrpat,10,datapath + "mnist_60k_trainlbl.bin");

    telbl = readpats(ntepat,10,datapath + "mnist_10k_testlbl.bin");

}
