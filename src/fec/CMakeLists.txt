option( HAS_SSE2 "Compile for SSE2" ON )
option( HAS_SSSE3 "Compile for SSSE3" ON )

if (NOT X86)
  set(HAS_SSE2 OFF)
  set(HAS_SSSE3 OFF)
endif()

set( FECPP_SOURCES fecpp.cpp cpuid.cpp )
if ( HAS_SSE2 )
  add_definitions( -msse2 )
  list( APPEND FECPP_SOURCES fecpp_sse2.cpp )
endif()
if ( HAS_SSSE3 )
  add_definitions( -mssse3 )
  list( APPEND FECPP_SOURCES fecpp_ssse3.cpp )
endif()

add_library( fecpp-object OBJECT ${FECPP_SOURCES} )
set_property( TARGET fecpp-object PROPERTY POSITION_INDEPENDENT_CODE ON )
add_library( fecpp        SHARED $<TARGET_OBJECTS:fecpp-object> )
target_include_directories (fecpp PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

