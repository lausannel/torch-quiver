ADD_LIBRARY(quiver SHARED)
TARGET_ADD_SOURCE_PATTERN(quiver srcs/cpp/src//hybrid/*.cpp)

IF(ENABLE_CUDA)
    TARGET_ADD_SOURCE_PATTERN(quiver srcs/cpp/src/hybrid/sample/*.cu)
    TARGET_INCLUDE_DIRECTORIES(quiver PRIVATE ${CUDA_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(quiver ${CUDA_LIBRARIES})
ENDIF()