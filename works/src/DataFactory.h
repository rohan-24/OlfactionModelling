/*

  Author: Anders Lansner

  Copyright (c) 2020 Anders Lansner

  All rights reserved. May not be derived from or modified without
  written consent of the copyright owner.

*/

#ifndef __DATAFACTORY_included
#define __DATAFACTORY_included

#include <vector>
#include <string>
#include <random>

#include "Globals.h"

class DataFactory {

 protected:

    int H,M,U,N;

    int nrow,ncol;

    int ntrpat,ntepat;

    std::string datapath;

    long shseed;

    std::mt19937_64 shgenerator;

 public:

    static std::vector<std::vector<float> > getpats(std::vector<std::vector<float> > pats,
						    int npat,int r0 = 0) ;

    DataFactory(int H,int M,int U = 1) ;

    std::vector<std::vector<float> > mkpats(int npat,Globals::Pat_t pat_t = Globals::RND01,
					    float p1 = 1,float p0 = 0,int verbosity = 0) ;

    void setdatapath(std::string datapath) ;

    //Transforms data image from [1-v,v] to [v] format
    void selectdata2() ;
    
    //Transforms data image from [v] to [1-v,v] to format
    void expanddata1() ;
    
    // Sets shgenerator seed
    long setshseed(long newshseed) ;

    // Returns shuffled index vector, size npat
    void getshuffledidx(std::vector<int> &idxvec) ;

    // Cutting out a rectangle [r0:rn,c0:cn] in each image
    void cutout2(int r0,int rn,int c0,int cn) ;

    // Load full MNIST
    void loadMNIST(int ntrpat = -1,int ntepat = -1);

} ;

#endif // __DATAFACTORY_included
