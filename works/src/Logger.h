/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __Logger_included
#define __Logger_included

#include <vector>
#include <string>

#include "Globals.h"

class Pop;
class HCU;
class PopH;
class BCU;
class Prj;
class PrjH;


class Logger {

protected:

    Pop *pop;
    HCU *hcu;
    PopH *poph;
    BCU *bcu;
    Prj *prj;
    PrjH *prjh;

    int r0,rn,c0,cn;

    std::string statestr,compstr;
    FILE *outf;
    bool on;
    int logstep;

    static std::vector<Logger *> loggers;


public:

    Logger(Pop *pop,std::string statestr,std::string filename,int logstep = 1) ;

    Logger(HCU *hcu,std::string statestr,std::string filename,int logstep = 1) ;

    Logger(PopH *poph,std::string statestr,std::string filename,int logstep = 1) ;

    Logger(BCU *bcu,std::string statestr,std::string filename,int logstep = 1) ;

    Logger(Prj *prj,std::string statestr,std::string filename,int logstep = 1) ;

    Logger(PrjH *prj,std::string statestr,std::string filename,int logstep = 1) ;

    void selectc(int i0,int n = 1) ;

    void selectrc(int r0,int c0 = 0,int rn = 1,int cn = 1) ;

    void select(std::vector<int> limvec) ;

    void dolog() ;

    void start() ;

    void stop() ;

    void close() ;

    static void dologall() ;

    static void closeall() ;

} ;


#endif // __Logger_included
