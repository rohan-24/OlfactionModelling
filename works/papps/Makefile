HOST=$(shell hostname)
USER=$(shell whoami)

ifeq ($(findstring beskow,$(HOST)),beskow)

CXX=CC
BASEDIR = /cfs/klemming/nobackup/a/$(USER)

else

CXX=mpicxx
BASEDIR = /home/$(USER)

endif

USESHMEM ?= false

CXXFLAGS ?= -fopenmp -Ofast -funroll-loops -ffast-math -std=c++11 -pedantic-errors ## gnu compile
#CXXFLAGS ?= -g -p -fopenmp -Ofast -funroll-loops -ffast-math -std=c++11 -pedantic-errors ## gnu compile
#CXXFLAGS ?= -g -p -Ofast -funroll-loops -ffast-math -std=c++11 -pedantic-errors ## gnu compile
#CXXFLAGS ?= -g -fopenmp -std=c++11 -pedantic-errors ## gnu compiler
#CXXFLAGS ?= -std=c++11 -pedantic-errors ## gnu compiler

BCPNNBIN = $(BASEDIR)/MyPrograms/BCPNNSim/works/bin
BCPNNLIB = $(BASEDIR)/MyPrograms/BCPNNSim/works/lib
BCPNNSRC = $(BASEDIR)/MyPrograms/BCPNNSim/works/src
PBCPNNSRC = $(BASEDIR)/MyPrograms/BCPNNSim/works/psrc

all: testmain1 mnistmain1 mpip2pmain1 mpiosmain1 mpicomain1 ppophmain1 pprjhmain1

TESTDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/test1
testmain1: $(BCPNNLIB)/libpbcpnnsim.a $(TESTDIR)/testmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(TESTDIR)/testmain1.cpp  -L $(BCPNNLIB) -lpbcpnnsim \
	-o $(TESTDIR)/testmain1

AMDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/am
ammain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(BCPNNSRC)/DataFactory.o $(AMDIR)/ammain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o \
	$(BCPNNSRC)/DataFactory.o $(AMDIR)/ammain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(AMDIR)/main1

DELDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/delay
delmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(DELDIR)/delmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(DELDIR)/delmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(DELDIR)/main1

delmain2: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(DELDIR)/delmain2.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(DELDIR)/delmain2.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(DELDIR)/main1

MPIP2PDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/mpip2ptest
mpip2pmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(MPIP2PDIR)/mpip2pmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(MPIP2PDIR)/mpip2pmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(MPIP2PDIR)/main1

MPIOSDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/mpiostest
mpiosmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(MPIOSDIR)/mpiosmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(MPIOSDIR)/mpiosmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(MPIOSDIR)/main1

MPICODIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/mpicotest
mpicomain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(MPICODIR)/mpicomain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(MPICODIR)/mpicomain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(MPICODIR)/main1

MPIRCDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/mpirctest
mpircmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(MPIRCDIR)/mpircmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(MPIRCDIR)/mpircmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(MPIRCDIR)/main1

SHMEMOSDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/shmemostest
shmemosmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(SHMEMOSDIR)/shmemosmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(SHMEMOSDIR)/shmemosmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(SHMEMOSDIR)/main1

SHMEMRCDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/shmemrctest
shmemrcmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemrcmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemrcmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(SHMEMRCDIR)/main1

shmemrcmain2: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemrcmain2.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemrcmain2.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(SHMEMRCDIR)/main1

shmemrcmain3: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemrcmain3.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemrcmain3.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(SHMEMRCDIR)/main1

shmemtestx: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemtestx.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemtestx.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(SHMEMRCDIR)/main1

shmemtestmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemtestmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(SHMEMRCDIR)/shmemtestmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(SHMEMRCDIR)/main1

SHMEMMNISTDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/shmemmnist
shmemmnistmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(SHMEMMNISTDIR)/shmemmnistmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o \
	$(SHMEMMNISTDIR)/shmemmnistmain1.cpp -L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim \
	-o $(SHMEMMNISTDIR)/main1

SHCLASSDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/shclass
shclassmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(BCPNNSRC)/DataFactory.o \
	$(SHCLASSDIR)/shclassmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(BCPNNSRC)/DataFactory.o \
	$(SHCLASSDIR)/shclassmain1.cpp -L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim \
	-o $(SHCLASSDIR)/main1

shclassmain2: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(BCPNNSRC)/DataFactory.o \
	$(SHCLASSDIR)/shclassmain2.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(BCPNNSRC)/DataFactory.o \
	$(SHCLASSDIR)/shclassmain2.cpp -L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim \
	-o $(SHCLASSDIR)/main2

PPOPRJDIR2 = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/ppoprj2
ppoprjmain2: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(PPOPRJDIR2)/ppoprjmain2.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o \
	-D USESHMEM=$(USESHMEM) $(PPOPRJDIR2)/ppoprjmain2.cpp -L $(BCPNNLIB) -lbcpnnsim \
	-L $(BCPNNLIB) -lpbcpnnsim -o $(PPOPRJDIR2)/main1

shmemxtest1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(PPOPRJDIR2)/shmemxtest1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o \
	-D USESHMEM=$(USESHMEM) $(PPOPRJDIR2)/shmemxtest1.cpp -L $(BCPNNLIB) -lbcpnnsim \
	-L $(BCPNNLIB) -lpbcpnnsim -o $(PPOPRJDIR2)/shmemxmain1

PPOPDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/ppop
ppopmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(PPOPDIR)/ppopmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(PPOPDIR)/ppopmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(PPOPDIR)/main1

PPOPRJDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/ppoprj
ppoprjmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(PPOPRJDIR)/ppoprjmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(PPOPRJDIR)/ppoprjmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(PPOPRJDIR)/main1

PPOPHDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/ppoph
ppophmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(PPOPHDIR)/ppophmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(PPOPHDIR)/ppophmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(PPOPHDIR)/main1

PPRJHDIR = $(BASEDIR)/MyPrograms/BCPNNSim/works/papps/pprjh
pprjhmain1: $(BCPNNLIB)/libpbcpnnsim.a $(BCPNNSRC)/Parseparam.o $(PPRJHDIR)/pprjhmain1.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) $(BCPNNSRC)/Parseparam.o $(PPRJHDIR)/pprjhmain1.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o $(PPRJHDIR)/main1

PAPPSDIR = ${BASEDIR}/MyPrograms/BCPNNSim/works/papps

clean:
	@for f in $(shell ls -d */ ${PAPPSDIR}); do cd $${f}; make lclean; cd ${PAPPSDIR}; done

lclean:
	@rm -f *.log *.bin runlog* makerr* *~ core *.pyc
#	@rm -f $(filter %main1,$(shell ls *))
