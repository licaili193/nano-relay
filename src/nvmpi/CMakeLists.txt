set(CMAKE_C_FLAGS“$ {CMAKE_C_FLAGS} -fPIC”)
set(CMAKE_CXX_FLAGS“$ {CMAKE_CXX_FLAGS} -fPIC”)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link=/usr/lib/aarch64-linux-gnu/tegra")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link=/usr/local/cuda/lib64")

find_library(LIB_NVBUF nvbuf_utils PATHS /usr/lib/aarch64-linux-gnu/tegra)
find_library(LIB_V4L2 nvv4l2 PATHS /usr/lib/aarch64-linux-gnu/tegra)
find_package(Threads REQUIRED)

add_library(nvmpi SHARED
    nvmpi_enc.cpp
    # nvmpi_dec.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvV4l2Element.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvV4l2ElementPlane.cpp
    # /usr/src/jetson_multimedia_api/samples/common/classes/NvVideoDecoder.cpp
    /usr/src/jetson_multimedia_api/samples/common/classes/NvVideoEncoder.cpp
)

target_link_libraries(nvmpi PUBLIC ${CMAKE_THREAD_LIBS_INIT} ${LIB_NVBUF}  ${LIB_V4L2})
target_include_directories(nvmpi PUBLIC /usr/src/jetson_multimedia_api/include)
target_include_directories(nvmpi PUBLIC /usr/local/cuda/include)
target_include_directories(nvmpi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

