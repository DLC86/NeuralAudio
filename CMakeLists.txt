cmake_minimum_required(VERSION 3.20)

project(NeuralAudio VERSION 0.1.6)

option(BUILD_PNAM_TESTS "Build PNAM regression tests and example" ON)

add_subdirectory(NeuralAudio)
add_subdirectory(NeuralAudioCAPI)

if(BUILD_PNAM_TESTS AND BUILD_PNAM)
    enable_testing()
    add_executable(pnam_test Tests/ParametricTest.cpp)
    target_link_libraries(pnam_test PRIVATE NeuralAudio)
    add_test(NAME pnam COMMAND pnam_test ${CMAKE_CURRENT_SOURCE_DIR}/Tests/fixtures)
    add_executable(pnam_render Examples/PnamRender.c)
    target_link_libraries(pnam_render PRIVATE NeuralAudioCAPI)
    if(WIN32)
        add_custom_command(TARGET pnam_render POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:NeuralAudioCAPI> $<TARGET_FILE_DIR:pnam_render>)
    endif()
    add_test(NAME pnam_c_example COMMAND pnam_render
        ${CMAKE_CURRENT_SOURCE_DIR}/Tests/fixtures/hyper_a2_low_rank.nam)
endif()

option(BUILD_UTILS "Build NeuralAudio utils" OFF)
if(BUILD_UTILS)
    message(STATUS "Building NeuralAudio Utils")
    add_subdirectory(Utils)
else()
    message(STATUS "NOT Building NeuralAudio Utils")
endif()
