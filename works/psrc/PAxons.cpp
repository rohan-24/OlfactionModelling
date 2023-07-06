/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#include <stdlib.h>
#include <string>
#include <vector>
#include <shmem.h>

#include "Globals.h"
#include "Timer.h"
#include "PGlobals.h"
#include "PAxons.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;

PAxons::PAxons(int N,int maxidelay) {

    this->N = N;

    this->maxidelay = maxidelay;

    shaxons = vector<float *>(N,NULL);

    for (size_t a=0; a<N; a++)

	PAxons::shaxons[a] = (float *)shmem_calloc(maxidelay+1,sizeof(float));

}


void PAxons::update(vector<float> act) {
    
    if (act.size()>N) perror("PAxons::update","Illegal act.size>N: act.size() = " +
			     to_string((int)act.size()) + " N = " + to_string(N));

    for (size_t n=0; n<act.size(); n++) {

	for (int d=maxidelay; d>0; d--) shaxons[n][d] = shaxons[n][d-1];

	shaxons[n][0] = act[n];

    }

    /// if (shrank==0) prnval(shaxons[0],maxidelay,2);

}


void PAxons::tap(vector<vector<int> > &taps,vector<float> &y) {

    for (size_t t=0; t<taps.size(); t++) {

	if (y.size()<taps[t][0]) perror("PAxons::tap","Illegal N<=tap[][0]");

	if (taps[t][1]<1) perror("PAxons::tap","Illegal tap[][1]<1");

	if (maxidelay<taps[t][1]) perror("PAxons::tap","Illegal maxidelay<tap[][1]");

	y[taps[t][0]] = shaxons[taps[t][0]][taps[t][1]];

    }
}


void PAxons::settaps(std::vector<std::vector<int> > taps) {

    for (size_t t=0; t<taps.size(); t++) {

	if (taps[t][1]<1) perror("PAxons::settap","Illegal tap[][1]<1");

	if (maxidelay<taps[t][1]) perror("PAxons::settap","Illegal maxidelay<tap[][1]");

    }

    this->taps = taps;

}


void PAxons::tap(std::vector<float> &y)  {

    if (taps.size()==0) perror("PAxons::tap","Illegal: No taps set");

    for (size_t t=0; t<taps.size(); t++) {

	if (y.size()<taps[t][0]) perror("PAxons::tap","Illegal N<=tap[][0]");

	y[taps[t][0]] = shaxons[taps[t][0]][taps[t][1]];

    }

}


// int main(int argc,char **args) {

//     int N = 200,D = 2000,T = 400,nstep = 1000;

//     pginitialize(argc,args);

//     PAxons *paxons = new PAxons(N,D);

//     vector<float> act(N,0),delact(N,0);

//     vector<vector<int> > taps(T,vector<int>(2,0));
//     for (int t=0; t<T; t++) { taps[t][0] = t%N; taps[t][1] = gnextint()%(D-1) + 1;  }

//     paxons->settaps(taps);

//     prnmat(taps);

//     Timer *alltimer = new Timer("alltime");

//     printf("Running ...\n"); fflush(stdout);

//     for (int step=0; step<nstep; step++) {

// 	for (int n=0; n<N; n++) act[n] = gnextfloat();

// 	paxons->tap(delact);

// 	// printf("%3d ",step); fflush(stdout); prnvec(act,2,-1,false); fflush(stdout); prnvec(delact,2);

// 	paxons->update(act);

//     }

//     alltimer->print();

//     return 0;

// }
