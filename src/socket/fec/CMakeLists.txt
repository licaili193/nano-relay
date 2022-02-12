add_library (fec_codec SHARED 
  fec_decoder.cpp
)

target_include_directories (fec_codec PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries (fec_codec PUBLIC 
    fecpp
    ${GLOG_LIBRARIES}
)
