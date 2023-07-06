/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#include <stdlib.h>
#include <string>
#include <random>
#include <omp.h>

#include "Globals.h"

#include "Timer.h"

using namespace std;

using namespace Globals;


Timer::Timer(string label,bool dostart) {

    this->label = label;

    start();
	
}


void Timer::start() {

    totime = 0;

    t0 = omp_get_wtime();
}


void Timer::stop() {

    if (t0>0) totime += omp_get_wtime() - t0;

    t0 = -1;
    
}


void Timer::resume() {

    t0 = omp_get_wtime();

}


double Timer::elapsed() {

    if (t0<-1.5) error("Timer::elapsed","Timer not started");

    stop();

    return totime;

}

void Timer::print(FILE *outf,string format) {

    format = "%s = " + format + " sec\n";

    fprintf(outf,format.c_str(),label.c_str(),elapsed());

}
	       

