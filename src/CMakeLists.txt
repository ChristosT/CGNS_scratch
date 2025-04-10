cmake_minimum_required(VERSION 3.20.3)

project(Helloworld_PETSc_MPI)

find_package(MPI)

find_package(VTK
  COMPONENTS
    CommonDataModel
    IOXML
)

include(FindPkgConfig)
pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET petsc)

if (NOT VTK_FOUND)
  message("Skipping example: ${VTK_NOT_FOUND_MESSAGE}")
  return ()
endif ()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

MESSAGE(STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER})
MESSAGE(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
MESSAGE(STATUS "PETSC LIB : " ${PETSCLIB})
MESSAGE(STATUS "PETSC INCLUDE : " ${PETSCINC})

add_executable(HelloWorld helloworld.cpp)

target_link_libraries(HelloWorld PRIVATE PkgConfig::PETSc ${MPI_CXX_LIBRARIES} ${VTK_LIBRARIES})
