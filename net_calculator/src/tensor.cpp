#include "tensor.hpp"
#include <assert.h>

Tensor::Tensor(std::vector<uint32_t> shape, float sparsity)
    : _shape(shape), _sparsity(sparsity)
{
    assert(sparsity <= 1.0 && sparsity >= 0.0);
}

uint64_t Tensor::Volume() const
{
    uint64_t vol = 1;
    for (auto d : _shape) {
        vol *= d;
    }
    return vol;
}

uint64_t Tensor::NzVolume() const 
{
    return uint64_t(Volume() * _sparsity);
}

bool Tensor::Match(const Tensor &a, const Tensor &b)
{
    if (a._shape.size() != b._shape.size()) {
        return false;
    }
    for (uint32_t i = 0; i < a._shape.size(); i++) {
        if (a._shape[i] != b._shape[i]) {
            return false;
        }
    }
    return true;
}

bool Tensor::Concat(const Tensor& t, uint32_t dim)
{
    // if this tensor is empty, copy the tensor into this one
    if (Empty()) {
        _shape = t.shape();
        _sparsity = t.sparsity();
        return;
    }

    // otherwise, try to concat it along the dim dimension
    _sparsity = (t.NzVolume() + NzVolume()) / (t.Volume() + Volume());
    if (t._shape.size() != _shape.size())
    {
        return false;
    }

    for (uint32_t i = 0; i < _shape.size(); i++) {
        if (i == dim) {
            _shape[i] += t._shape[i];
        }
        else {
            if (_shape[i] != t._shape[i]) {
                return false;
            }
        }
    }
    return true;
}

void Tensor::Mask(double nz) {
    _sparsity = _sparsity * nz;
}

void Tensor::Fill(double nz) {
    _sparsity = 1 - (1 - _sparsity) * (1 - nz);
}

Tensor Tensor::Flatten()
{
    uint32_t size = Volume();
    return Tensor({size}, _sparsity);
}

bool Tensor::Empty()
{
    return _shape.empty();
}