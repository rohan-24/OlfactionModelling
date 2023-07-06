cmake_minimum_required(VERSION 3.10)

# set the project name
project(bcpnnlib VERSION  0.9.5 DESCRIPTION "The BCPNN library")

INCLUDE_DIRECTORIES(include)

# add the executable
add_library(bcpnnlib STATIC
  Globals.cpp
  Parseparam.cpp
  Timer.cpp
  Pop.cpp
  HCU.cpp
  PopH.cpp
  BCU.cpp
  Prj.cpp
  PrjH.cpp
  BCC.cpp
  Logger.cpp
  DataFactory.cpp)

target_compile_options(bcpnnlib PRIVATE -fopenmp)
install(TARGETS bcpnnlib DESTINATION bcpnn/lib)
