cmake_minimum_required(VERSION 3.5.0)
project(regionsegmentation VERSION 0.1.0 LANGUAGES C CXX)

# include(CTest)
# enable_testing()

#header include 경로
include_directories(${PROJECT_INCLUDE_DIR})
# include_directories(
#     ./include
# )

MESSAGE(STATUS "PROJECT NAME: ${CMAKE_PROJECT_NAME}")

find_package(OpenCV REQUIRED )  
include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_subdirectory(LSD)
add_subdirectory(RegionRoomSegmentaion)

#add_executable(${CMAKE_PROJECT_NAME} test.cpp ${SOURCES})
add_executable(${CMAKE_PROJECT_NAME} run_test.cpp ${SOURCES})

target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC RegionRoomSegmentaion LSD ${OpenCV_LIBS})  

# file(GLOB all_srcs "./src/*.cpp")
# add_executable( ${CMAKE_PROJECT_NAME} ${all_srcs})

# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
