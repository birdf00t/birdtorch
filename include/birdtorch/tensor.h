#pragma once //헤더파일을 여러곳에 include 해도 한 번만 읽게 해줌

#include <vector> //std::vector
#include <string> //std::string
#include <stdexcept> // err
#include <numeric> //std::accumulate
#include <iostream> //std::cout

namespace birdtorch { //birdtorch::Tensor

class Tensor {
public:
    std::vector<float> data;    // 실제 값 (1D로 펼쳐서 저장)
    std::vector<int>   shape;   // ex) {2, 3} → 2행 3열
    int                ndim;    // shape.size() 랑 같음

    Tensor();                                          // 빈 텐서
    Tensor(std::vector<int> shape);                    // shape만 주면 0으로 초기화
    Tensor(std::vector<float> data, std::vector<int> shape); // data + shape

    // ─── 기본 정보 ────────────────────────────────
    int    size() const;         // 전체 원소 수 (shape 곱)
    void   print() const;        // 텐서 출력

    // ─── 원소 접근 ────────────────────────────────
    float& at(std::vector<int> indices);               // ex) t.at({1, 2})
    float  at(std::vector<int> indices) const;

    //+,*기호를 쑬 수 있게
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

private:
    // indices → 1D data 배열 인덱스로 변환
    int flat_index(std::vector<int> indices) const;
};

} 