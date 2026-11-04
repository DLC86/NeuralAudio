set(CMAKE_CXX_STANDARD 20)

set(SOURCES NeuralAudioCApi.h NeuralAudioCApi.cpp)
	
add_library(NeuralAudioCAPI SHARED ${SOURCES})

target_include_directories(NeuralAudioCAPI PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../NeuralAudio)
target_compile_definitions(NeuralAudioCAPI PRIVATE NEURALAUDIO_CAPI_BUILD=1)

target_link_libraries(NeuralAudioCAPI PRIVATE NeuralAudio)

source_group(NeuralAudioCAPI ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})
