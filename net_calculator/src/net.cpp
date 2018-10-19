#include "net.hpp"
#include "util.hpp"
#include <assert.h>

//=========================================================
// class Net
//=========================================================
Net::Net(std::string name, std::vector<Net *> src) : name(name), _src(src)
{
}

void Net::addDst(Net *dst)
{
    _dst.push_back(dst);
    return;
}

//=========================================================
// class Input
//=========================================================
Input::Input(tsize_t size) : Net("input", {}) { _input_size = size; }

tsize_t Input::getOutputSize() { return _input_size; }

//=========================================================
// class Conv2D
//=========================================================
Conv2D::Conv2D(std::string name, std::vector<Net *> src, tsize_t param_size, 
    tsize_t stride, nl_t nl) : Net(name, src)
{
    assert(param_size.size() == 2);
    assert(stride.size() == 1);
    _param_size = param_size;
    _stride = stride;
    _nl = nl;

    assert(src.size() > 0);
    _input_size = src[0]->getOutputSize();

    // concat if there are more sources
    if (src.size() > 1) {
        for (int i = 1; i < src.size(); i++) {
            assert(concat(_input_size, src[i]->getOutputSize()));
        }
    }
}

tsize_t Conv2D::getOutputSize()
{
    return {_param_size[0], 
            _input_size[1] / _stride[0], 
            _input_size[2] / _stride[1]};
}

double  Conv2D::getParamNum()
{
    return _param_size[0] * _param_size[1] * _param_size[2];
}

// TODO: consider drop out and weight sparsity
double  Conv2D::getInferenceMacNum()
{
    tsize_t o_size = getOutputSize();
    return o_size[0] * o_size[1] * o_size[2] * _param_size[1] * _param_size[2];
}

// TODO: consider error & weight sparsity
double  Conv2D::getPropagationMacNum()
{
    return getInferenceMacNum();
}

// TODO: consider error & weight sparsity
double  Conv2D::getUpdateMacNum()
{
    return getInferenceMacNum();
}

//=========================================================
// class FC
//=========================================================
FC::FC(std::string name, std::vector<Net *> src, uint32_t neuron_num, nl_t nl)
    : Net(name, src)
{
    // concat and flatten
    _input_size.push_back(0);
    for (auto net : src) {
        _input_size[0] += volume(net->getOutputSize());
    }

    _neuron_num = neuron_num;
    _nl = nl;
}

tsize_t FC::getOutputSize()
{
    return {_neuron_num};
}

double  FC::getParamNum()
{
    return _input_size[0] * _neuron_num;
}

// TODO: consider drop out and weight sparsity
double  FC::getInferenceMacNum()
{
    return _input_size[0] * _neuron_num;
}

// TODO: consider error & weight sparsity
double  FC::getPropagationMacNum()
{
    return _input_size[0] * _neuron_num;
}

// TODO: consider error & weight sparsity
double  FC::getUpdateMacNum()
{
    return _input_size[0] * _neuron_num;
}

//=========================================================
// class Pool
//=========================================================

Pool::Pool(std::string name, Net *src, tsize_t pool_size, tsize_t stride)
    : Net(name, {src})
{
    _input_size = src->getOutputSize();
    _pool_size = pool_size;
    _stride = stride;
}

tsize_t Pool::getOutputSize()
{
    return {
        _input_size[0] / _stride[0],
        _input_size[1] / _stride[1]
    };
}