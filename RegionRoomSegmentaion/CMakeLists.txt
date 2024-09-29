cmake_minimum_required(VERSION 3.0.0)
project(RegionRoomSegmentaion) 

find_package(OpenCV REQUIRED )  
link_libraries(${OpenCV_LIBS})

# add_library(RegionRoomSegmentaion src/featuredetection.cpp 
#                                   src/trajectioryPoint.cpp 
#                                   src/simplifiedMap.cpp
#                                   src/MSED.cpp
#                                   )

add_library(RegionRoomSegmentaion src/utility.cpp
                                  src/roomSeg.cpp
                                  )

target_include_directories(RegionRoomSegmentaion PUBLIC include ${OpenCV_LIBS} )

target_link_libraries(RegionRoomSegmentaion PRIVATE LSD)

