#ifndef __PIO_included
#define __PIO_included

#include <vector>
#include <string>
#include <mpi.h>

#include "Globals.h"

class PPobjR;

class PIO {

 public:

 protected:

    static std::vector<PIO *> pios;

    MPI_Comm comm;

    MPI_File fh;

    int soffs,N,wstep;

    friend class PLogger;

 public:

    static void pfcloseall() ;

    PIO(int N) ;

    void pfopen(std::string filename,std::string mode = "w") ;

    void pfreadstatevec(int offs,std::vector<float> &statevec) ;

    void pfreadstatevec(std::string filename,int offs,std::vector<float> &statevec) ;

    void pfwritestatevec(int offs,std::vector<float> statevec) ;

    void pfwritestatevec(std::string filename,int offs,std::vector<float> statevec) ;

    void pfclose() ;

} ;

#endif // __PIO_included

