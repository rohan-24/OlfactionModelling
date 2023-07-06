cmake_minimum_required(VERSION 3.10)

# set the project name
project(shmemmnistmain1 DESCRIPTION "Application SHMEMMNIST")

add_executable(shmemmnistmain1 shmemmnistmain1.cpp)
target_link_libraries(shmemmnistmain1 pbcpnnlib -fopenmp)
install(TARGETS shmemmnistmain1 DESTINATION bcpnn/bin)
# install(FILES shmemmnistmain1.par DESTINATION bcpnn/bin)
