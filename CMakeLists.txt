cmake_minimum_required(VERSION 3.24)
project(streetkit VERSION 0.1.0 LANGUAGES CXX)

# The C++ half of streetkit: reading tiles and the manifest, choosing what to stream, walking, riding and flying,
# and the views over a map (signs, places, night lights, a bike's handlebars). A game fetches this repository and
# links `streetkit`; the headers that draw (tile_stream.h and the *_view.h) also need the game's Strata target,
# strata::static, which streetkit does not fetch itself so that a game pins one Strata for everything.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(streetkit STATIC src/tile.cpp)
add_library(streetkit::streetkit ALIAS streetkit)
target_include_directories(streetkit PUBLIC include include/streetkit)
if(NOT MSVC)
  target_compile_options(streetkit PRIVATE -Wall -Wextra)
endif()

option(STREETKIT_BUILD_TESTS "Build streetkit's unit tests" ${PROJECT_IS_TOP_LEVEL})
if(STREETKIT_BUILD_TESTS AND NOT EMSCRIPTEN)
  enable_testing()
  add_executable(test_streetkit tests/cpp/test_streetkit.cpp)
  target_link_libraries(test_streetkit PRIVATE streetkit)
  if(NOT MSVC)
    target_compile_options(test_streetkit PRIVATE -Wall -Wextra)
  endif()
  add_test(NAME units COMMAND test_streetkit ${CMAKE_CURRENT_BINARY_DIR}/fixture)
endif()
