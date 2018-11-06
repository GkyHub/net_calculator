#pragma once

#include <vector>

class Tensor {
public:
    std::vector<uint32_t> _shape;       // the sizes of each dimension of the tensor
    float                 _sparsity;    // ratio of zero values

public:
    uint32_t volume();  // the number of elements in the tensor
    uint32_t nzVolume
}