#include "net.hpp"
#include "util.hpp"
#include <assert.h>
#include <math.h>

//=========================================================
// class Net
//=========================================================
Net::Net(type_t type, std::string name, std::vector<Net *> src) 
    : name(name), _src(src), _type(type)
{
	if (!src.empty()) {
		for (Net *n : src) {
			n->_dst.push_back(n);
		}
	}
}

Tensor Net::input()
{
    if (_src.empty()) {
        return Tensor({});
    }
    return _src[0]->output();
}

void Net::dropout(float nz)
{
    _output.Mask(nz);
}

//=========================================================
// class Input
//=========================================================
Input::Input(shape_t size) : Net(Net::Input, "input", {}) 
{ 
    _output = Tensor(size); 
}

//=========================================================
// class Conv2D
//=========================================================
Conv2D::Conv2D(std::string name, std::vector<Net *> src, shape_t param_shape, 
    shape_t stride, float sparsity) : Net(Net::Conv2D, name, {src})
{
    assert(param_shape.size() == 3);
    assert(stride.size() == 2);
    assert(src.size() > 0);

    _kernel = Tensor(param_shape, sparsity);
    _stride = stride;    
    
    // concat if there are more sources
    for (uint32_t i = 0; i < src.size(); i++) {
        assert(_input.Concat(src[i]->output()));
    }

    _output = Tensor({
        _kernel.shape()[0], 
        _input.shape()[1] / stride[0],
        _input.shape()[2] / stride[1]});
}

uint64_t Conv2D::getParamNum()
{
    return _kernel.NzVolume();
}

uint64_t Conv2D::getInferenceMacNum()
{
    return _output.NzVolume() * 
        _kernel.shape()[1] * 
        _kernel.shape()[2] *
        _kernel.sparsity() *
        _input.shape()[0];
}

uint64_t Conv2D::getPropagationMacNum()
{
    return getInferenceMacNum();
}

// TODO: consider error & weight sparsity
uint64_t Conv2D::getUpdateMacNum()
{
    return getInferenceMacNum();
}

//=========================================================
// class FC
//=========================================================
FC::FC(std::string name, std::vector<Net *> src, uint32_t neuron_num, double sparsity)
    : Net(Net::FC, name, {src})
{
    _output   = Tensor({neuron_num});

    Tensor input;
    for (auto net : src) {
        input.Concat(net->output().Flatten());
    }

    _weights = Tensor({input.shape()[0], neuron_num}, sparsity);
}

uint64_t FC::getParamNum()
{
    return _weights.NzVolume();
}

uint64_t FC::getInferenceMacNum()
{
    return _weights.NzVolume();
}

uint64_t FC::getPropagationMacNum()
{
    return _weights.NzVolume();
}

uint64_t FC::getUpdateMacNum()
{
    return _weights.NzVolume();
}

//=========================================================
// class NL
//=========================================================

NL::NL(std::string name, Net *src, type_t type)
    : Net(Net::NL, name, {src}), _nl_type(type)
{
    _output = src->output();
}

void NL::dropout(float nz)
{
    _output.Mask(nz);
    _src[0]->dropout(nz);
}

//=========================================================
// class Pool
//=========================================================

Pool::Pool(std::string name, Net *src, type_t type, shape_t pool_size, shape_t stride)
    : Net(NL::Pool, name, {src})
{
    assert(src->output().shape().size() == 3);
    _pool_size = pool_size;
    _stride = stride;
    _pool_type = type;
    _output = src->output();
    _output.shape()[1] /= _stride[0];
    _output.shape()[2] /= _stride[1];
}

void Pool::dropout(float nz)
{
    _output.Mask(nz);
    _src[0]->dropout(nz);
}

//=========================================================
// class EleWise
//=========================================================

EleWise::EleWise(std::string name, Net *src1, Net *src2)
    : Net(Net::EleWise, name, {src1, src2})
{
    assert(Tensor::Match(src1->output(), src2->output()));
    _output = src1->output();
    _output.Fill(src2->output().sparsity());
}

void EleWise::dropout(float nz)
{
    _output.Mask(nz);
    _src[0]->dropout(nz);
    _src[1]->dropout(nz);
}