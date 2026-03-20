#include <cassert>
#include <iostream>
#include "birdtorch/tensor.h"

using namespace birdtorch;

void test_create() {
    Tensor t({2, 3});
    assert(t.size() == 6);
    assert(t.ndim == 2);
    std::cout << "test_create passed\n";
}

void test_at() {
    Tensor t({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
    assert(t.at({0, 0}) == 1.0f);
    assert(t.at({0, 1}) == 2.0f);
    assert(t.at({1, 0}) == 3.0f);
    assert(t.at({1, 1}) == 4.0f);
    std::cout << "test_at passed\n";
}

void test_add() {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({4.0f, 5.0f, 6.0f}, {3});
    Tensor c = a + b;
    assert(c.data[0] == 5.0f);
    assert(c.data[1] == 7.0f);
    assert(c.data[2] == 9.0f);
    std::cout << "test_add passed\n";
}

void test_mul() {
    Tensor a({1.0f, 2.0f, 3.0f}, {3});
    Tensor b({2.0f, 2.0f, 2.0f}, {3});
    Tensor c = a * b;
    assert(c.data[0] == 2.0f);
    assert(c.data[1] == 4.0f);
    assert(c.data[2] == 6.0f);
    std::cout << "test_mul passed\n";
}

int main() {
    test_create();
    test_at();
    test_add();
    test_mul();
    std::cout << "All tests passed!\n";
    return 0;
}