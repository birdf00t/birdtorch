cmake_minimum_required(VERSION 3.14)
project(birdtorch)

set(CMAKE_CXX_STANDARD 17)

# 헤더 경로 등록
include_directories(include)

# 라이브러리 빌드 (tensor.cpp → libbirdtorch)
add_library(birdtorch src/tensor.cpp)

# 테스트 실행파일 빌드
add_executable(test_tensor tests/cpp/test_tensor.cpp)
target_link_libraries(test_tensor birdtorch)