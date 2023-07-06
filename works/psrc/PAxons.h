/*

  Author: Anders Lansner

  Copyright (c) 2019 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __PAXONS_included
#define __PAXONS_included

#include <string>
#include <vector>


class PAxons {

 protected:

    std::vector<float *> shaxons;

    std::vector<std::vector<int> > taps;

    int N,maxidelay;
    
    friend class PPobjR;

    friend class Shmem;

    friend class PPopR;

 public:

    PAxons(int N,int maxidelay) ;

    void update(std::vector<float> act) ;

    int getdslot(int idelay) ;

    void tap(std::vector<std::vector<int> > &taps,std::vector<float> &y) ;

    void settaps(std::vector<std::vector<int> > taps);

    void tap(std::vector<float> &y) ;

} ;

#endif // __PAXONS_included

