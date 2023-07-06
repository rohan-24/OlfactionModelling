cmake_minimum_required(VERSION 3.10)

# set the project name
project(shclassmain1 DESCRIPTION "Application SHCLASS")

add_executable(shclassmain1 shclassmain1.cpp)
target_link_libraries(shclassmain1 pbcpnnlib -fopenmp)
install(TARGETS shclassmain1 DESTINATION bcpnn/bin)
# install(FILES shclassmain1.par DESTINATION bcpnn/bin)

add_executable(shclassmain2 shclassmain2.cpp)
target_link_libraries(shclassmain2 pbcpnnlib -fopenmp)
install(TARGETS shclassmain2 DESTINATION bcpnn/bin)
# install(FILES shclassmain2.par DESTINATION bcpnn/bin)
