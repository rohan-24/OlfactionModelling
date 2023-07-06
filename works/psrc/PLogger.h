/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __PLogger_included
#define __PLogger_included

#include <vector>
#include <string>

#include "Globals.h"
#include "PPobjR.h"

class PPopulation;

class PProjection;

class PIO;

class PLogger {

protected:

    PPopulation *ppopulation;

    PProjection *pprojection;

    PIO *pio;

    std::string statestr;
    bool on;
    int N,logstep;

    static std::vector<PLogger *> ploggers;


public:

    PLogger(PPopulation *ppopulation,std::string statestr,std::string filename,int logstep = 1) ;

    PLogger(PProjection *pprojection,std::string statestr,std::string filename,int logstep = 1) ;

    void dolog() ;

    void start() ;

    void stop() ;

    static void dologall() ;

    static void closeall() ;

} ;


#endif // __PLogger_included
