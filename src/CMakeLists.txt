add_executable (camera_test 
    camera_test.cpp
)

target_link_libraries (camera_test PUBLIC 
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
)

cuda_add_executable (stream_test_2 
    stream_test_2.cu
)

target_link_libraries (stream_test_2 ${CUDA_LINK_LIBRARIES_KEYWORD} 
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${CUDA_LIBRARY}
    ${OPENGL_LIBRARIES}
    nvmpi
    fecpp
    practical_socket
    modi_socket
    image_processing
    cuda
    OpenGL::EGL
)

cuda_add_library (camera_grabber SHARED 
  camera_grabber.cu
)

target_include_directories (camera_grabber PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries (camera_grabber ${CUDA_LINK_LIBRARIES_KEYWORD} 
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${CUDA_LIBRARY}
    ${OPENGL_LIBRARIES}
    cuda
    OpenGL::EGL

    nvmpi # include this for nvbuf_utils.h 
    image_processing
)

cuda_add_executable (stream_test_3 
    stream_test_3.cu
)

target_link_libraries (stream_test_3 ${CUDA_LINK_LIBRARIES_KEYWORD} 
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${CUDA_LIBRARY}
    ${OPENGL_LIBRARIES}
    nvmpi
    fecpp
    practical_socket
    modi_socket
    image_processing
    cuda
    OpenGL::EGL
    camera_grabber # TODO: clean up the dependency
)

add_executable (fake_vehicle_node 
    fake_vehicle_node.cpp
)

target_link_libraries (fake_vehicle_node PUBLIC 
    practical_socket
    modi_socket
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
)

target_include_directories(fake_vehicle_node PRIVATE 
    ${CURRENT_CMAKE_SOURCE_DIR}
)

add_subdirectory (nvmpi)
add_subdirectory (fec)
add_subdirectory (practical_socket)
add_subdirectory (socket)
add_subdirectory (image_processing)