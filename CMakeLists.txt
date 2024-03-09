cmake_minimum_required(VERSION 3.0)
project(traffic_light_reco VERSION 1.1.0 LANGUAGES CXX)

# ビルドフラグの設定
set(CMAKE_CXX_STANDARD 17)

# OpenCVの検出と設定
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# インクルードパスの追加
if(WIN32)
  include_directories(C:/local/include)
  link_directories(C:/local/lib)
else()
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
endif()

# ビルドに関する設定
set(EXECUTABLES "traffic_right_reco")
set(DEPENDENT_LIBS ${OpenCV_LIBRARIES} rs_common)

foreach(target IN LISTS EXECUTABLES)
  add_executable(${target} src/${target}.cpp)
  target_link_libraries(${target} ${DEPENDENT_LIBS} stdc++fs)
endforeach()
