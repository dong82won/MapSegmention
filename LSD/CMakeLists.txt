cmake_minimum_required(VERSION 3.0.0)
project(LSD) 

add_library(LSD src/lsd.cpp)
target_include_directories(LSD PUBLIC include)


