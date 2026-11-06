cmake_minimum_required(VERSION 3.25)
project(Stockfish LANGUAGES CXX ASM)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build -G Ninja")
endif()

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" OR
   NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "This branch supports only 64-bit x86 Linux on the Ryzen 9800X3D.")
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
    message(FATAL_ERROR "Use GCC or Clang to build this branch.")
endif()

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14) OR
   (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19))
    message(FATAL_ERROR "Zen 5 requires GCC 14+ or Clang 19+.")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=znver5" HAS_ZEN5)
if(NOT HAS_ZEN5)
    message(FATAL_ERROR "The compiler must support -march=znver5.")
endif()

find_package(Threads REQUIRED)
include(cmake/Network.cmake)
fetch_network()

set(SOURCES
    src/attacks.cpp
    src/benchmark.cpp
    src/bitboard.cpp
    src/engine.cpp
    src/evaluate.cpp
    src/main.cpp
    src/memory.cpp
    src/misc.cpp
    src/movegen.cpp
    src/movepick.cpp
    src/position.cpp
    src/score.cpp
    src/search.cpp
    src/thread.cpp
    src/timeman.cpp
    src/tt.cpp
    src/tune.cpp
    src/uci.cpp
    src/ucioption.cpp
    src/syzygy/tbprobe.cpp
    src/nnue/network.cpp
    src/nnue/nnue_accumulator.cpp
    src/nnue/nnue_misc.cpp
    src/nnue/features/full_threats.cpp
    src/nnue/features/k32h3.cpp
    src/nnue/features/pp_3wide.cpp
)

set(GIT_SHA "nogit")
string(TIMESTAMP GIT_DATE "%Y%m%d" UTC)
set(GIT_MODIFIED false)
find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" log -1 --format=%h
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE SHA OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE GIT_RESULT)
    if(NOT GIT_RESULT EQUAL 0)
        message(FATAL_ERROR "Could not read the Git revision.")
    endif()
    set(GIT_SHA "${SHA}")
    execute_process(COMMAND "${GIT_EXECUTABLE}" log -1 --format=%cs
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE DATE OUTPUT_STRIP_TRAILING_WHITESPACE
                    COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "-" "" GIT_DATE "${DATE}")
    execute_process(COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    RESULT_VARIABLE DIRTY)
    if(DIRTY EQUAL 1)
        set(GIT_MODIFIED true)
    elseif(NOT DIRTY EQUAL 0)
        message(FATAL_ERROR "Could not read the Git worktree state.")
    endif()
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
             ${SOURCES} src/evaluate.h)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git/HEAD")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS .git/HEAD .git/index)
endif()
configure_file(cmake/build_info.h.in generated/build_info.h @ONLY)
configure_file(cmake/nnue_embed.S.in generated/nnue_embed.S @ONLY)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/generated/nnue_embed.S"
                            PROPERTIES OBJECT_DEPENDS "${NNUE_FILE}")

add_executable(stockfish ${SOURCES} "${CMAKE_CURRENT_BINARY_DIR}/generated/nnue_embed.S")
set_target_properties(stockfish PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)
target_include_directories(stockfish PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(stockfish PRIVATE Threads::Threads rt)
target_compile_options(stockfish PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=znver5;-Wall;-Wextra;-Wcast-qual;-Wshadow;-Wmissing-declarations;-pedantic;-fno-exceptions>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3;-funroll-loops>
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(stockfish PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fconstexpr-ops-limit=500000000;-fno-ipa-cp-clone>
    )
else()
    target_compile_options(stockfish PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fconstexpr-steps=500000000>
    )
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR LANGUAGES CXX)
if(NOT IPO_SUPPORTED)
    message(FATAL_ERROR "Link-time optimization is required: ${IPO_ERROR}")
endif()
set_property(TARGET stockfish PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS src/*.h)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${HEADERS})

enable_testing()
add_test(NAME signature COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/signature.sh" 1556650)
set_tests_properties(signature PROPERTIES WORKING_DIRECTORY "$<TARGET_FILE_DIR:stockfish>"
                     ENVIRONMENT "EXE=$<TARGET_FILE:stockfish>" TIMEOUT 120)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME instrumented COMMAND "${Python3_EXECUTABLE}"
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/instrumented.py" --none "$<TARGET_FILE:stockfish>")
set_tests_properties(instrumented PROPERTIES TIMEOUT 600)

add_custom_target(bench COMMAND "$<TARGET_FILE:stockfish>" bench
                  DEPENDS stockfish USES_TERMINAL)
find_program(CLANG_FORMAT NAMES clang-format-20)
if(CLANG_FORMAT)
    file(GLOB_RECURSE FORMAT_SOURCES CONFIGURE_DEPENDS src/*.cpp)
    add_custom_target(format COMMAND "${CLANG_FORMAT}" -i ${FORMAT_SOURCES} ${HEADERS})
endif()

install(TARGETS stockfish RUNTIME DESTINATION bin)
