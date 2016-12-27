# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

INCLUDE(ExternalProject)

SET(WARPCTC_SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/warpctc)
SET(WARPCTC_INSTALL_DIR ${PROJECT_BINARY_DIR}/warpctc)

IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" )
    SET(USE_OMP ON)
ELSE()
    SET(USE_OMP OFF)
ENDIF()

ExternalProject_Add(
    warpctc
    GIT_REPOSITORY  "https://github.com/gangliao/warp-ctc.git"
    PREFIX          ${WARPCTC_SOURCES_DIR}
    CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${WARPCTC_INSTALL_DIR}
    CMAKE_ARGS      -DWITH_GPU=${WITH_GPU}
    CMAKE_ARGS      -DWITH_OMP=${USE_OMP}
    UPDATE_COMMAND ""
)

SET(WARPCTC_INCLUDE_DIR "${WARP_INSTALL_DIR}/include" CACHE PATH "Warp-ctc Directory" FORCE)
INCLUDE_DIRECTORIES(${WARPCTC_INCLUDE_DIR})

IF(WIN32)
    SET(WARPCTC_LIBRARIES
        "${WARPCTC_INSTALL_DIR}/lib/warpctc.dll" CACHE FILEPATH "Warp-ctc Library" FORCE)
ELSE(WIN32)
    SET(WARPCTC_LIBRARIES
        "${WARPCTC_INSTALL_DIR}/lib/libwarpctc.dylib" CACHE FILEPATH "Warp-ctc Library" FORCE)
ENDIF(WIN32)

LIST(APPEND external_project_dependencies warpctc)
