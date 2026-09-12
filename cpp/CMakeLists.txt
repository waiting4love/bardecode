cmake_minimum_required(VERSION 3.24)
project(bardecode_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  add_compile_options(/utf-8)  # zxing-cpp sources contain UTF-8 literals
endif()

include(FetchContent)

# NOTE: this network cannot reach github.com directly. All source deps are
# fetched as tarballs from codeload.github.com, which is reachable.
# onnxruntime comes from nuget.org (official Microsoft.ML.OnnxRuntime package).

# ----------------------------------------------------------------------------
# nlohmann/json (header-only)
# ----------------------------------------------------------------------------
FetchContent_Declare(json
  URL https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.11.3
)
FetchContent_MakeAvailable(json)

# ----------------------------------------------------------------------------
# GoogleTest (tests only)
# ----------------------------------------------------------------------------
option(BARDECODE_BUILD_TESTS "Build tests" ON)
if(BARDECODE_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://codeload.github.com/google/googletest/tar.gz/refs/tags/v1.15.2
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
endif()

# ----------------------------------------------------------------------------
# zxing-cpp (build the core library from source)
# ----------------------------------------------------------------------------
set(ZXING_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZXING_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZXING_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ZXING_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
set(ZXING_WRITERS OFF CACHE BOOL "" FORCE)
# zxing-cpp's example/ dir fetches stb from github.com (blocked here); override
# it with a pre-downloaded codeload tarball so FetchContent never hits github.
if(NOT EXISTS "${CMAKE_BINARY_DIR}/stb-override/stb.h")
  file(DOWNLOAD "https://codeload.github.com/nothings/stb/tar.gz/refs/heads/master"
       "${CMAKE_BINARY_DIR}/stb.tar.gz" SHOW_PROGRESS TLS_VERIFY ON)
  file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/stb.tar.gz"
       DESTINATION "${CMAKE_BINARY_DIR}/stb-override")
endif()
file(GLOB _stb_dir "${CMAKE_BINARY_DIR}/stb-override/stb-*")
list(GET _stb_dir 0 _stb_dir)
set(FETCHCONTENT_SOURCE_DIR_STB "${_stb_dir}" CACHE INTERNAL "")
FetchContent_Declare(zxing
  URL https://codeload.github.com/zxing-cpp/zxing-cpp/tar.gz/refs/tags/v2.2.1
)
FetchContent_MakeAvailable(zxing)

# ----------------------------------------------------------------------------
# OpenCV (build core/imgproc/imgcodecs/dnn from source)
# ----------------------------------------------------------------------------
set(BUILD_LIST "core,imgproc,imgcodecs,dnn" CACHE STRING "" FORCE)
set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "" FORCE)  # match our /MD runtime
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_JAVA OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(WITH_ITT OFF CACHE BOOL "" FORCE)
set(WITH_MSMF ON CACHE BOOL "" FORCE)   # Windows Media Foundation for webp/jpg codecs
FetchContent_Declare(opencv
  URL https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.10.0
)
FetchContent_MakeAvailable(opencv)

# ----------------------------------------------------------------------------
# onnxruntime (nuget package Microsoft.ML.OnnxRuntime; no source build needed)
# ----------------------------------------------------------------------------
set(ORT_VERSION 1.20.1)
set(ORT_ROOT "${CMAKE_BINARY_DIR}/ort-${ORT_VERSION}")
if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  set(ORT_URL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${ORT_VERSION}/microsoft.ml.onnxruntime.${ORT_VERSION}.nupkg")
  message(STATUS "Downloading onnxruntime ${ORT_VERSION} from nuget ...")
  file(DOWNLOAD "${ORT_URL}" "${CMAKE_BINARY_DIR}/onnxruntime.nupkg" SHOW_PROGRESS
       TLS_VERIFY ON)
  file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/onnxruntime.nupkg"
       DESTINATION "${CMAKE_BINARY_DIR}/ort-nupkg")
  file(COPY "${CMAKE_BINARY_DIR}/ort-nupkg/build/native/include" DESTINATION "${ORT_ROOT}")
  file(COPY "${CMAKE_BINARY_DIR}/ort-nupkg/runtimes/win-x64/native/" DESTINATION "${ORT_ROOT}/lib")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime.dll"
  IMPORTED_IMPLIB   "${ORT_ROOT}/lib/onnxruntime.lib"
  INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)

# ----------------------------------------------------------------------------
# bardecode library + executable
# ----------------------------------------------------------------------------
add_library(bardecode_lib STATIC
  src/image_io.cpp
  src/detector.cpp
  src/decoder.cpp
  src/pipeline.cpp
)
target_include_directories(bardecode_lib PUBLIC src)
# OpenCV's in-tree module targets don't reliably export interface include
# paths to sibling directories; add them explicitly.
target_include_directories(bardecode_lib SYSTEM PUBLIC
  ${opencv_SOURCE_DIR}/modules/core/include
  ${opencv_SOURCE_DIR}/modules/imgproc/include
  ${opencv_SOURCE_DIR}/modules/imgcodecs/include
  ${opencv_SOURCE_DIR}/modules/dnn/include
  ${CMAKE_BINARY_DIR}  # generated opencv_modules.hpp / cvconfig.h land here
)
target_link_libraries(bardecode_lib PUBLIC
  nlohmann_json::nlohmann_json
  opencv_core opencv_imgproc opencv_imgcodecs opencv_dnn
  ZXing
  onnxruntime
)
# Export for tests/ (FetchContent vars are directory-scoped)
set(BARDECODE_OPENCV_SOURCE_DIR "${opencv_SOURCE_DIR}" CACHE INTERNAL "")

add_executable(bardecode src/main.cpp)
target_link_libraries(bardecode PRIVATE bardecode_lib)

# onnxruntime DLL must sit next to the executables
add_custom_command(TARGET bardecode POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_ROOT}/lib/onnxruntime.dll"
          "$<TARGET_FILE_DIR:bardecode>/onnxruntime.dll")

# Copy the ONNX model next to the executable (mirrors Python package layout)
set(MODEL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../bardecode/models/yolov8s_barcode.onnx")
if(EXISTS "${MODEL_SRC}")
  add_custom_command(TARGET bardecode POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MODEL_SRC}"
            "$<TARGET_FILE_DIR:bardecode>/models/yolov8s_barcode.onnx")
endif()

# tests last: they link onnxruntime / opencv / zxing targets declared above
add_subdirectory(tests)
