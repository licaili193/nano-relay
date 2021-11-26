add_library (modi_socket SHARED 
  udp_send_socket.cpp
)

target_include_directories (modi_socket PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries (modi_socket PUBLIC 
    practical_socket
    fecpp
    ${GLOG_LIBRARIES}
)
