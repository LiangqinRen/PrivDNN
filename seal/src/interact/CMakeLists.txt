# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.13)

project(INTERACT VERSION 4.1.1 LANGUAGES CXX)

add_compile_options(-g -O0 -Wall -fstack-protector-all -fnon-call-exceptions)

# define the project path 
set(PROJECT_PATH ${PROJECT_SOURCE_DIR}/../../../)
configure_file(path.h.in ../path.h)

# If not called from root CMakeLists.txt
if(NOT DEFINED SEAL_BUILD_EXAMPLES)
    set(SEAL_BUILD_EXAMPLES ON)

    # Import Microsoft SEAL
    find_package(SEAL 4.1.1 EXACT REQUIRED)
endif()

include_directories(${PROJECT_SOURCE_DIR}/../../include)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}../../../output/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}../../../output/bin)

if(SEAL_BUILD_EXAMPLES)
    SET(server_srcs server.cpp timer.cpp utils.cpp)
    SET(client_srcs client.cpp timer.cpp utils.cpp)

    ADD_LIBRARY(server SHARED ${server_srcs})
    ADD_LIBRARY(client SHARED ${client_srcs})
    ADD_EXECUTABLE(client_bin ${client_srcs})

    if(TARGET SEAL::seal OR TARGET SEAL::seal_shared)
        target_link_libraries(server PRIVATE SEAL::seal)
        target_link_libraries(client PRIVATE SEAL::seal)
        target_link_libraries(client_bin PRIVATE SEAL::seal)
    else()
        message(FATAL_ERROR "Cannot find target SEAL::seal or SEAL::seal_shared")
    endif()
endif()
