
cmake_minimum_required(VERSION 3.10)

project(BCPNN)

add_compile_options(-O3 -fopenmp -g -DUSE_OPENMP -std=c++11)
INCLUDE_DIRECTORIES(src)
INCLUDE_DIRECTORIES(psrc)

add_subdirectory(src)
add_subdirectory(apps)

### Only on Beskow
# add_subdirectory(psrc)
# add_subdirectory(papps)

