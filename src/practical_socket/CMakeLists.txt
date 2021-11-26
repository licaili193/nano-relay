add_library (practical_socket SHARED 
    practical_socket.cpp
)

target_include_directories (practical_socket PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
