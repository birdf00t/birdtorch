#include "birdtorch/tensor.h"

namespace birdtorch {

// ─── 생성자 구현 ──────────────────────────────────────

// 빈 텐서
Tensor::Tensor() : ndim(0) {}

// shape만 주면 0.0f로 채움
// ex) Tensor({2, 3}) → 2x3 텐서, 전부 0
Tensor::Tensor(std::vector<int> shape) : shape(shape) {
    ndim = shape.size();
    int total = size();           // shape 원소 전부 곱한 값
    data.resize(total, 0.0f);     // total 크기로 0 초기화
}

// data + shape 둘 다 직접 지정
Tensor::Tensor(std::vector<float> data, std::vector<int> shape)
    : data(data), shape(shape) {
    ndim = shape.size();

    // data 크기랑 shape이 맞는지 검증
    if ((int)data.size() != size()) {
        throw std::invalid_argument("data size does not match shape");
    }
}

// ─── 기본 정보 ────────────────────────────────────────

// shape의 모든 원소를 곱함
// ex) shape = {2, 3, 4} → 24
int Tensor::size() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

// 텐서 출력
void Tensor::print() const {
    std::cout << "Tensor(shape=[";
    for (int i = 0; i < (int)shape.size(); i++) {
        std::cout << shape[i];
        if (i < (int)shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (int i = 0; i < (int)data.size(); i++) {
        std::cout << data[i];
        if (i < (int)data.size() - 1) std::cout << ", ";
    }
    std::cout << "])\n";
}

// ─── 원소 접근 ────────────────────────────────────────

// indices → flat index 변환
// ex) shape={2,3}, indices={1,2} → 1*3 + 2 = 5
int Tensor::flat_index(std::vector<int> indices) const {
    if ((int)indices.size() != ndim) {
        throw std::invalid_argument("indices dimension mismatch");
    }
    int idx = 0;
    int stride = 1;
    // 뒤에서부터 계산 (row-major order)
    for (int i = ndim - 1; i >= 0; i--) {
        idx += indices[i] * stride;
        stride *= shape[i];
    }
    return idx;
}

// 쓰기 가능한 접근 (reference 반환)
float& Tensor::at(std::vector<int> indices) {
    return data[flat_index(indices)];
}

// 읽기 전용 접근
float Tensor::at(std::vector<int> indices) const {
    return data[flat_index(indices)];
}

// ─── 기본 연산 ────────────────────────────────────────

// 원소별 덧셈 (element-wise add)
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("shape mismatch for addition");
    }
    Tensor result(shape);
    for (int i = 0; i < size(); i++) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

// 원소별 곱셈 (element-wise multiply)
Tensor Tensor::operator*(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::invalid_argument("shape mismatch for multiplication");
    }
    Tensor result(shape);
    for (int i = 0; i < size(); i++) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

} // namespace birdtorch