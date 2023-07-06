cmake_minimum_required(VERSION 3.10)

# set the project name
project(pbcpnnlib VERSION  0.9.5 DESCRIPTION "The PBCPNN library")

INCLUDE_DIRECTORIES(include)

# add the executable
add_library(pbcpnnlib STATIC
  ../src/Globals.cpp
  ../src/Parseparam.cpp
  ../src/Timer.cpp
  ../src/Pop.cpp
  ../src/HCU.cpp
  ../src/PopH.cpp
  ../src/BCU.cpp
  ../src/Prj.cpp
  ../src/PrjH.cpp
  ../src/BCC.cpp
  ../src/Logger.cpp
  ../src/DataFactory.cpp
  PAxons.cpp
  PGlobals.cpp
  PIO.cpp
  PLogger.cpp
  PPobjR.cpp
  PPopH.cpp
  PPopR.cpp
  PPrj.cpp
  PPrjH.cpp
  PPrjR.cpp)

target_compile_options(pbcpnnlib PRIVATE -fopenmp)
install(TARGETS pbcpnnlib DESTINATION bcpnn/lib)
