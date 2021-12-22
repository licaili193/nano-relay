find_package(CUDA REQUIRED)

cuda_add_library (image_processing SHARED 
  image_processing.cu
)

target_include_directories (image_processing PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries (image_processing ${CUDA_LINK_LIBRARIES_KEYWORD} 
  ${CUDA_LIBRARY}    
)
