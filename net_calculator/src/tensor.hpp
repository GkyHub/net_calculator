#pragma once

#include <vector>

typedef std::vector<uint32_t> shape_t;

// the shape of a tensor, also with the sparsity
class Tensor {
private:
    std::vector<uint32_t> _shape;       // the sizes of each dimension of the tensor
    float                 _sparsity;    // ratio of zero values

public:
    Tensor(std::vector<uint32_t> shape, float sparsity = 1.0);
    Tensor() { /* empty constructor */};

    std::vector<uint32_t> shape() const { return _shape; };
    float sparsity() const { return _sparsity; };

    uint64_t Volume() const;    // the number of elements in the tensor
    uint64_t NzVolume() const;  // the number of non-zero elements in the tensor
    void Mask(double nz);       // mask a certain ratio of elements away
    void Fill(double nz);       // fill a certain ratio of elements as non-zeros

    // if two tensors have the same shape
    static bool Match(const Tensor &a, const Tensor &b);

    // concat a tensor to this one along dim
    bool Concat(const Tensor &t, uint32_t dim = 0);

    // return a 1-dim tensor with the same volume as the original tensor
    Tensor Flatten();

    // return if a tensor is empty
    bool Empty();
};