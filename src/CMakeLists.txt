cmake_minimum_required(VERSION 3.11)

file(GLOB_RECURSE MODEL_CU_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cu)
file(GLOB_RECURSE MODEL_CPP_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/model/*.cpp)
file(GLOB_RECURSE UTILS_CU_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cu)
file(GLOB_RECURSE UTILS_CPP_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)


add_library(cudnn_models STATIC ${MODEL_CU_FILES} ${MODEL_CPP_FILES} ${UTILS_CU_FILES} ${UTILS_CPP_FILES})
set_property(TARGET cudnn_models PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_include_directories(cudnn_models PUBLIC ${CMAKE_SOURCE_DIR}/include/models ${CMAKE_SOURCE_DIR}/include/utils /usr/local/cuda/include)
target_link_libraries(cudnn_models PUBLIC -lcudnn -lcublas)

add_executable(run_mnist main.cu)
target_include_directories(run_mnist PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(run_mnist PUBLIC cudnn_models)

