/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __Timer_included
#define __Timer_included

class Timer {

 protected:

    std::string label;

    double t0 = -2,totime;

 public:

    Timer(std::string label = "",bool dostart = true) ;

    void start() ;

    void stop() ;

    void resume() ;

    double elapsed() ;

    void print(FILE *outf = stdout,std::string = "%.3f") ;

} ;


#endif // __Timer_included
