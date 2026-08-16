cmake_minimum_required(VERSION 3.20)

project(llama_atlas_capture LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "llama-atlas-capture currently requires Linux renameat2 semantics")
endif()

set(LLAMA_CPP_ROOT "" CACHE PATH "Pinned llama.cpp source root")
set(LLAMA_CPP_BUILD "" CACHE PATH "Build directory for the same pinned llama.cpp checkout")
set(ATLAS_LLAMA_CPP_COMMIT "4df29be4f4c3673f428170fda944a5b19f743bb8")

if(NOT IS_DIRECTORY "${LLAMA_CPP_ROOT}")
    message(FATAL_ERROR "LLAMA_CPP_ROOT must name the pinned llama.cpp source directory")
endif()
if(NOT IS_DIRECTORY "${LLAMA_CPP_BUILD}")
    message(FATAL_ERROR "LLAMA_CPP_BUILD must name its completed build directory")
endif()

find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${LLAMA_CPP_ROOT}" rev-parse HEAD
    OUTPUT_VARIABLE LLAMA_CPP_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
if(NOT LLAMA_CPP_COMMIT STREQUAL ATLAS_LLAMA_CPP_COMMIT)
    message(FATAL_ERROR
        "llama.cpp source must be pinned to ${ATLAS_LLAMA_CPP_COMMIT}; got ${LLAMA_CPP_COMMIT}")
endif()

set(LLAMA_BUILD_CACHE "${LLAMA_CPP_BUILD}/CMakeCache.txt")
if(NOT EXISTS "${LLAMA_BUILD_CACHE}")
    message(FATAL_ERROR "LLAMA_CPP_BUILD has no CMakeCache.txt")
endif()
file(READ "${LLAMA_BUILD_CACHE}" LLAMA_BUILD_CACHE_TEXT)
string(FIND "${LLAMA_BUILD_CACHE_TEXT}"
    "CMAKE_HOME_DIRECTORY:INTERNAL=${LLAMA_CPP_ROOT}" LLAMA_BUILD_ROOT_MATCH)
if(LLAMA_BUILD_ROOT_MATCH EQUAL -1)
    message(FATAL_ERROR "LLAMA_CPP_BUILD was not configured from LLAMA_CPP_ROOT")
endif()

set(LLAMA_BIN "${LLAMA_CPP_BUILD}/bin")
set(LLAMA_LIBRARY "${LLAMA_BIN}/libllama.so.0.1.0")
set(COMMON_LIBRARY "${LLAMA_BIN}/libllama-common.so.0.1.0")
set(GGML_LIBRARY "${LLAMA_BIN}/libggml.so.0.20.0")
set(GGML_BASE_LIBRARY "${LLAMA_BIN}/libggml-base.so.0.20.0")

foreach(required_file IN ITEMS
        "${LLAMA_LIBRARY}" "${COMMON_LIBRARY}" "${GGML_LIBRARY}" "${GGML_BASE_LIBRARY}")
    if(NOT EXISTS "${required_file}")
        message(FATAL_ERROR "Required pinned shared library is missing: ${required_file}")
    endif()
endforeach()

add_executable(llama-atlas-capture atlas_capture.cpp)
target_compile_features(llama-atlas-capture PRIVATE cxx_std_17)
target_compile_options(llama-atlas-capture PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_include_directories(llama-atlas-capture PRIVATE
    "${LLAMA_CPP_ROOT}/include"
    "${LLAMA_CPP_ROOT}/common"
    "${LLAMA_CPP_ROOT}/ggml/include"
    "${LLAMA_CPP_ROOT}/src"
    "${LLAMA_CPP_ROOT}/vendor"
)
target_link_libraries(llama-atlas-capture PRIVATE
    "${COMMON_LIBRARY}"
    "${LLAMA_LIBRARY}"
    "${GGML_LIBRARY}"
    "${GGML_BASE_LIBRARY}"
)
set_target_properties(llama-atlas-capture PROPERTIES
    BUILD_RPATH "${LLAMA_BIN}"
    INSTALL_RPATH "${LLAMA_BIN}"
)
