project(mesh)

cmake_minimum_required(VERSION 2.8)

find_package(DOLFIN)

include_directories(${DOLFIN_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")

find_package(CGAL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")

# message("${CGAL_LIBRARIES}")

link_directories("/usr/lib/")
include_directories("${DOLFIN_INCLUDE_DIRS}")

add_executable(mesh mesh.cpp)
target_link_libraries(mesh ${CGAL_LIBRARIES} ${DOLFIN_LIBRARIES})
