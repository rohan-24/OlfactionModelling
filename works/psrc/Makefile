HOST=$(shell hostname)
USER=$(shell whoami)

ifeq ($(findstring beskow,$(HOST)),beskow)

CXX=CC
BASEDIR = /cfs/klemming/nobackup/a/$(USER)

else

CXX=mpicxx
BASEDIR = /home/$(USER)

endif

CXXFLAGS ?= -fopenmp -Ofast -funroll-loops -ffast-math -std=c++11 -pedantic-errors ## gnu compile
#CXXFLAGS ?= -g -fopenmp -Ofast -funroll-loops -ffast-math -std=c++11 -pedantic-errors ## gnu compile
#CXXFLAGS ?= -g -p -fopenmp -Ofast -funroll-loops -ffast-math -std=c++11 -pedantic-errors ## gnu compile
#CXXFLAGS ?= -g -p -Ofast -funroll-loops -ffast-math -std=c++11 -pedantic-errors ## gnu compile
#CXXFLAGS ?= -g -fopenmp -std=c++11 -pedantic-errors ## gnu compiler
#CXXFLAGS ?= -std=c++11 -pedantic-errors ## gnu compiler

all: libpbcpnnsim.a
	cp *.o ../pbin/
	mv libpbcpnnsim.a ../lib/

BCPNNBIN = $(BASEDIR)/MyPrograms/BCPNNSim/works/bin
BCPNNLIB = $(BASEDIR)/MyPrograms/BCPNNSim/works/lib
BCPNNSRC = $(BASEDIR)/MyPrograms/BCPNNSim/works/src
PBCPNNSRC = $(BASEDIR)/MyPrograms/BCPNNSim/works/psrc

PGlobals.o: $(BCPNNSRC)/Globals.h $(BCPNNSRC)/HCU.h $(PBCPNNSRC)/PGlobals.h ./PGlobals.cpp $(BCPNNSRC)/HCU.o
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PGlobals.cpp -c

PPrj.o: $(BCPNNSRC)/Globals.h $(BCPNNSRC)/HCU.h $(BCPNNSRC)/Prj.h $(PBCPNNSRC)/PPrj.h ./PPrj.cpp \
	$(BCPNNSRC)/HCU.o
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PPrj.cpp -c

PAxons.o: $(BCPNNSRC)/Globals.h ./PAxons.h ./PAxons.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PAxons.cpp -c

paxons: $(BCPNNLIB)/libpbcpnnsim.a ./PAxons.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PAxons.cpp \
	-L $(BCPNNLIB) -lbcpnnsim -L $(BCPNNLIB) -lpbcpnnsim -o paxons

PPrjH.o: $(BCPNNSRC)/Globals.h $(BCPNNSRC)/HCU.h $(PBCPNNSRC)/PPrjH.h ./PPrjH.cpp $(BCPNNSRC)/HCU.o
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PPrjH.cpp -c

PPobjR.o: ./PGlobals.h ./PPobjR.h ./PPobjR.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PPobjR.cpp -c

PPopR.o: ./PGlobals.h ./PPobjR.h ./PPopR.h ./PPopR.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PPopR.cpp -c

PPrjR.o: ./PGlobals.h ./PPobjR.h ./PPrj.h ./PPobjR.h ./PPrjR.h ./PPrjR.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PPrjR.cpp -c

PIO.o:  ./PPobjR.h ./PIO.h ./PIO.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PIO.cpp -c

PLogger.o:  ./PPobjR.h ./PPopR.h ./PPrjR.h ./PLogger.h ./PLogger.cpp
	$(CXX) $(CXXFLAGS) -I $(BCPNNSRC) -I $(PBCPNNSRC) ./PLogger.cpp -c

libpbcpnnsim.a: $(BCPNNBIN)/Globals.o $(BCPNNBIN)/Parseparam.o $(BCPNNBIN)/Pop.o $(BCPNNBIN)/HCU.o \
	$(BCPNNBIN)/PopH.o $(BCPNNBIN)/BCU.o ./PPrj.o ./PPrjH.o $(BCPNNBIN)/PrjH.o $(BCPNNBIN)/BCC.o \
	$(BCPNNBIN)/Logger.o $(BCPNNBIN)/Area.o $(BCPNNBIN)/Timer.o ./PGlobals.o ./PPobjR.o ./PPopR.o \
	./PPrjR.o ./PAxons.o ./PIO.o ./PLogger.o
	rm -f libpbcpnnsim.a
	@ar -cq libpbcpnnsim.a $(BCPNNBIN)/Globals.o $(BCPNNBIN)/Parseparam.o $(BCPNNBIN)/Pop.o \
	$(BCPNNBIN)/HCU.o $(BCPNNBIN)/PopH.o $(BCPNNBIN)/BCU.o ./PPrj.o ./PPrjH.o $(BCPNNBIN)/PrjH.o \
	$(BCPNNBIN)/BCC.o $(BCPNNBIN)/Logger.o $(BCPNNBIN)/Area.o $(BCPNNBIN)/Timer.o ./PGlobals.o \
	./PPobjR.o ./PPopR.o ./PPrjR.o ./PAxons.o ./PIO.o ./PLogger.o

clean:
	rm -f *~  *.o *.bin *.log *.pyc makerr* runlog* core
	rm -f libpbcpnnsim.a

recompile:
	make clean
	make
