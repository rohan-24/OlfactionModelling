cmake_minimum_required(VERSION 3.10)

# set the project name
project(delay1 DESCRIPTION "Application DELAY")

add_executable(delay1 delmain1.cpp)
target_link_libraries(delay1 pbcpnnlib -fopenmp)
install(TARGETS delay1 DESTINATION bcpnn/bin)
# install(FILES delmain1.par DESTINATION bcpnn/bin)
