add_executable (stream_test 
    stream_test.cpp
)

target_link_libraries (stream_test PUBLIC 
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${OpenCV_LIBS}
    nvmpi
    fecpp
    practical_socket
    modi_socket
)

add_executable (camera_test 
    camera_test.cpp
)

target_link_libraries (camera_test PUBLIC 
    ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
)

add_subdirectory (nvmpi)
add_subdirectory (fec)
add_subdirectory (practical_socket)
add_subdirectory (socket)