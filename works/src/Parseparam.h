/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __Parseparam_INCLUDED__
#define __Parseparam_INCLUDED__

#include <vector>
#include <string>

enum Value_t { Int = 0, Long, Float, Boole, String } ;

class Parseparam {

 public:

    Parseparam(std::string paramfile) ;

    void error(std::string errloc,std::string errstr) ;

    void postparam(std::string paramstring,void *paramvalue,Value_t paramtype) ;

    int findparam(std::string paramstring) ;

    void doparse(std::string paramlogfile = "") ;

    bool haschanged() ;

    void padwith0(std::string &str,int len) ;

    std::string timestamp() ;

    std::string dolog(bool usetimestamp) ;

 protected:

    std::string _paramlogfile,_paramfile,_timestamp;
    std::vector<std::string> _paramstring;
    std::vector<void *> _paramvalue;
    std::vector<Value_t> _paramtype;
    time_t _oldmtime;

} ;

#endif // __Parseparam_INCLUDED__
