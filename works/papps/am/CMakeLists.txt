cmake_minimum_required(VERSION 3.10)

# set the project name
project(ammain1 DESCRIPTION "Application AM")

add_executable(ammain1 ammain1.cpp)
target_link_libraries(ammain1 pbcpnnlib -fopenmp)
install(TARGETS ammain1 DESTINATION bcpnn/bin)
# install(FILES ammain1.par DESTINATION bcpnn/bin)
