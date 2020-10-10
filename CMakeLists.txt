cmake_minimum_required(VERSION 3.15)
project( SkinSegmentation )

option(USE_CUDA "Use CUDA" ON)

file(GLOB HEADERS "${CMAKE_SOURCE_DIR}/src/*.h")
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
if(USE_CUDA)
  file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
endif()

# OpenMP
find_package( OpenMP )
if(OpenMP_CXX_FOUND)
    add_compile_definitions(WITH_OPENMP)
endif()

# OpenCV
find_package( OpenCV REQUIRED COMPONENTS core imgproc videoio highgui )

# CUDA
include(CheckLanguage)
check_language(CUDA)
if (USE_CUDA)
        set(CUDA_ARCHITECTURES "Auto" CACHE STRING "\"Auto\" detects local machine GPU compute arch at runtime, \"Common\" and \"All\" cover common and entire subsets of architectures, \"Names\" is a list of architectures to enable by name, \"Numbers\" is a list of compute capabilities (version number) to enable")
        set_property(CACHE CUDA_ARCHITECTURES PROPERTY STRINGS "Auto" "Common" "All" "Kepler Maxwell Kepler+Tegra Maxwell+Tegra Pascal" "5.0 7.5")
        #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)
        enable_language( CUDA )
        if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "10.0")
            message(STATUS "Unsupported CUDA version, please upgrade to CUDA 10+. Disabling CUDA support")
            set(USE_CUDA "OFF" CACHE BOOL "Enable CUDA support" FORCE)
        else()  
            find_package( CUDA )
            cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCHITECTURES})
            message(STATUS "USE_CUDA: ${USE_CUDA}")
            message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
            message(STATUS "Building with CUDA flags: " "${CUDA_ARCH_FLAGS}")
        endif()
        set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
endif()

add_executable( SkinSegmentation main.cpp ${SOURCES} ${HEADERS} ${CUDA_SOURCES} )

if (USE_CUDA AND CUDA_FOUND)
    set_target_properties(SkinSegmentation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_link_libraries( SkinSegmentation PRIVATE ${OpenCV_LIBS} )
if(OpenMP_CXX_FOUND)
    target_link_libraries(SkinSegmentation PRIVATE OpenMP::OpenMP_CXX)
endif()
if (USE_CUDA AND CUDA_FOUND)
    target_include_directories(SkinSegmentation PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries( SkinSegmentation PRIVATE cuda )
    target_compile_definitions(SkinSegmentation PRIVATE -DWITH_CUDA)
endif()


set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Install prefix" FORCE)
set(INSTALL_BIN_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Path where exe and dll will be installed")
install(
    TARGETS SkinSegmentation
    CONFIGURATIONS Release
    RUNTIME DESTINATION bin
)