cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0054 NEW)

project(py_aff3ct LANGUAGES CXX)

# Enable C++17 (required by pybind11 3.x)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Threads library (required to link with AFF3CT)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Find pybind11 (from the active Python environment where pystreampu is installed)
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Find cpptrace, AFF3CT, and StreamPU libraries
find_package(cpptrace CONFIG 0.4.1)
find_package(AFF3CT CONFIG 4.7.2 REQUIRED)

# Extract libjulia path from AFF3CT and provide Julia::Julia imported target if needed by streampu-config.cmake
set(JULIA_LIB_PATH "")
get_target_property(AFF3CT_LINK_LIBS aff3ct::aff3ct-static-lib INTERFACE_LINK_LIBRARIES)
if (AFF3CT_LINK_LIBS)
    foreach(LIB_ITEM IN LISTS AFF3CT_LINK_LIBS)
        if (LIB_ITEM MATCHES "libjulia")
            set(JULIA_LIB_PATH "${LIB_ITEM}")
        endif()
    endforeach()
endif()
if (JULIA_LIB_PATH AND NOT TARGET Julia::Julia)
    add_library(Julia::Julia UNKNOWN IMPORTED)
    set_target_properties(Julia::Julia PROPERTIES IMPORTED_LOCATION "${JULIA_LIB_PATH}")
endif()

find_package(StreamPU CONFIG REQUIRED)

# Ensure AFF3CT / StreamPU was compiled with Julia support (-DSPU_LINK_JULIA=ON -> exports SPU_JULIA)
get_target_property(AFF3CT_COMPILE_DEFS aff3ct::aff3ct-static-lib INTERFACE_COMPILE_DEFINITIONS)
list(FIND AFF3CT_COMPILE_DEFS "SPU_JULIA" SPU_JULIA_IDX)
if (SPU_JULIA_IDX EQUAL -1)
    message(FATAL_ERROR "AFF3CT/StreamPU was not compiled with Julia support ('SPU_JULIA' definition is missing). "
                        "Please recompile AFF3CT with '-DSPU_LINK_JULIA=\"ON\"'.")
endif()

# Create the pybind11 extension module
pybind11_add_module(py_aff3ct MODULE ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper.cpp)

# Keep default symbol visibility so C++ RTTI (type_info) is shared across pyStreamPU and py_aff3ct
set_target_properties(py_aff3ct PROPERTIES
                      OUTPUT_NAME py_aff3ct
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                      POSITION_INDEPENDENT_CODE ON
                      CXX_VISIBILITY_PRESET default
                      VISIBILITY_INLINES_HIDDEN OFF)

# Detect free-threaded Python (e.g. Python 3.13t / 3.14t)
if (DEFINED Python_EXECUTABLE)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import sysconfig; print(sysconfig.get_config_var('Py_GIL_DISABLED') or 0)"
        OUTPUT_VARIABLE PY_GIL_DISABLED
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (PY_GIL_DISABLED EQUAL 1)
        message(STATUS "Free-threaded Python detected — enabling Py_GIL_DISABLED")
        target_compile_definitions(py_aff3ct PRIVATE Py_GIL_DISABLED)
    endif()
endif()

# Link against the shared StreamPU library first so all spu::* symbols resolve to libstreampu.dylib
target_link_libraries(py_aff3ct PRIVATE spu::spu-shared-lib aff3ct::aff3ct-static-lib)

# Add StreamPU and libjulia folders to RPATH so py_aff3ct finds libstreampu and libjulia at import time
get_target_property(SPU_SO_LOC spu::spu-shared-lib IMPORTED_LOCATION_RELEASE)
if (NOT SPU_SO_LOC)
    get_target_property(SPU_SO_LOC spu::spu-shared-lib IMPORTED_LOCATION)
endif()
if (SPU_SO_LOC)
    get_filename_component(SPU_LIB_DIR "${SPU_SO_LOC}" DIRECTORY)
    if (SPU_LIB_DIR)
        message(STATUS "Adding StreamPU library directory to RPATH of py_aff3ct: ${SPU_LIB_DIR}")
        set_property(TARGET py_aff3ct APPEND PROPERTY BUILD_RPATH "${SPU_LIB_DIR}")
        set_property(TARGET py_aff3ct APPEND PROPERTY INSTALL_RPATH "${SPU_LIB_DIR}")
    endif()
endif()

get_target_property(AFF3CT_LINK_LIBS aff3ct::aff3ct-static-lib INTERFACE_LINK_LIBRARIES)
if (AFF3CT_LINK_LIBS)
    foreach(LIB_ITEM IN LISTS AFF3CT_LINK_LIBS)
        if (LIB_ITEM MATCHES "libjulia")
            get_filename_component(JULIA_LIB_DIR "${LIB_ITEM}" DIRECTORY)
            if (JULIA_LIB_DIR)
                message(STATUS "Adding Julia library directory to RPATH of py_aff3ct: ${JULIA_LIB_DIR}")
                set_property(TARGET py_aff3ct APPEND PROPERTY BUILD_RPATH "${JULIA_LIB_DIR}")
                set_property(TARGET py_aff3ct APPEND PROPERTY INSTALL_RPATH "${JULIA_LIB_DIR}")
            endif()
        endif()
    endforeach()
endif()
