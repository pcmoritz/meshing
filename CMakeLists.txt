project(mesh)

cmake_minimum_required(VERSION 2.8)

find_package(DOLFIN)

include_directories(${DOLFIN_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/modules/")

find_package(CGAL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")

find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
include_directories("/usr/include/eigen3/")

# message("${CGAL_LIBRARIES}")

message("Generating the model...")
execute_process(COMMAND source ~/Work/FEniCS/share/dolfin/dolfin.conf)
execute_process(COMMAND ffc -l dolfin model.ufl
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

link_directories("/usr/lib/")
include_directories("${DOLFIN_INCLUDE_DIRS}")

add_executable(mesh mesh.cpp)
target_link_libraries(mesh ${CGAL_LIBRARIES} ${DOLFIN_LIBRARIES})
