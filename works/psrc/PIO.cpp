#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <stdio.h>

#include "Globals.h"
#include "pbcpnnsim.h"
#include "PIO.h"

using namespace std;
using namespace Globals;
using namespace PGlobals;


vector<PIO *> PIO::pios;

void PIO::pfcloseall() {

    for (size_t p=0; p<pios.size(); p++) pios[p]->pfclose();

}


PIO::PIO(int N) {

    this->N = N;

    wstep = 0;

    this->comm = MPI_COMM_WORLD;

    pios.push_back(this);

}


void PIO::pfopen(string filename,string mode) {

    if (mode=="r")

	MPI_File_open(comm,filename.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&fh);

    else if (mode=="w") {

	MPI_File_delete(filename.c_str(),MPI_INFO_NULL);

	MPI_File_open(comm,filename.c_str(),MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);

    }
}


void PIO::pfclose() {

    MPI_File_close(&fh);

}


void PIO::pfreadstatevec(int offs,vector<float> &statevec) {

    offs += wstep * N;

    offs *= sizeof(float);

    MPI_File_read_at(fh,offs,statevec.data(),statevec.size(),MPI_FLOAT,MPI_STATUS_IGNORE); 

    wstep++;
    
}


void PIO::pfreadstatevec(string filename,int offs,vector<float> &statevec) {

    MPI_File_open(comm,filename.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&fh);

    pfreadstatevec(offs,statevec);

    MPI_File_close(&fh);

}


void PIO::pfwritestatevec(int offs,vector<float> statevec) {

    soffs = wstep * N + offs;

    soffs *= sizeof(float);

    MPI_File_write_at(fh,soffs,statevec.data(),statevec.size(),MPI_FLOAT,MPI_STATUS_IGNORE); 

}


void PIO::pfwritestatevec(string filename,int offs,vector<float> statevec) {

    offs = 0;

    MPI_File_open(MPI_COMM_WORLD,filename.c_str(),MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);

    pfwritestatevec(offs,statevec);

    MPI_File_close(&fh);

}


