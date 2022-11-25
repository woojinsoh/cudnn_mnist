cmake_minimum_required(VERSION 3.11)

file(GLOB_RECURSE MODEL_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cu)
file(GLOB_RECURSE UTILS_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cu)

add_library(cudnn_models STATIC ${MODEL_FILES} ${UTILS_FILES})
set_property(TARGET cudnn_models PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(cudnn_models PUBLIC ${CMAKE_SOURCE_DIR}/include/models ${CMAKE_SOURCE_DIR}/include/utils)
target_link_libraries(cudnn_models PUBLIC -lcudnn -lcublas)

add_executable(run_mnist main.cu)
target_include_directories(run_mnist PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(run_mnist PUBLIC cudnn_models)