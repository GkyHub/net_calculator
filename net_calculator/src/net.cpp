#include "net.hpp"
#include "util.hpp"
#include <assert.h>

//=========================================================
// class Net
//=========================================================
Net::Net(std::string name, std::vector<Net *> src) : name(name), _src(src)
{
	if (!src.empty()) {
		for (Net *n : src) {
			n->_dst.push_back(n);
		}
	}
}

void Net::addDst(Net *dst)
{
    _dst.push_back(dst);
    return;
}

//=========================================================
// class Input
//=========================================================
Input::Input(shape_t size) : Net("input", {}) { _input_size = size; }

shape_t Input::getOutputShape() { return _input_size; }

//=========================================================
// class Conv2D
//=========================================================
Conv2D::Conv2D(std::string name, std::vector<Net *> src, shape_t param_size, 
    shape_t stride) : Net(name, {src})
{
    assert(param_size.size() == 3);
    assert(stride.size() == 2);
    _param_size = param_size;
    _stride = stride;

    assert(src.size() > 0);
    _input_size = src[0]->getOutputShape();

    // concat if there are more sources
    if (src.size() > 1) {
        for (uint32_t i = 1; i < src.size(); i++) {
            assert(concat(_input_size, src[i]->getOutputShape()));
        }
    }
}

shape_t Conv2D::getOutputShape()
{
    return {_param_size[0], 
            _input_size[1] / _stride[0], 
            _input_size[2] / _stride[1]};
}

double  Conv2D::getParamNum()
{
    return _param_size[0] * _param_size[1] * _param_size[2] * _input_size[0];
}

// TODO: consider drop out and weight sparsity
double  Conv2D::getInferenceMacNum()
{
    shape_t o_size = getOutputShape();
    return o_size[0] * o_size[1] * o_size[2] * _param_size[1] * _param_size[2] * _input_size[0];
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
FC::FC(std::string name, std::vector<Net *> src, uint32_t neuron_num)
    : Net(name, {src})
{
    // concat and flatten
    _input_size.push_back(0);
    for (auto net : src) {
        _input_size[0] += volume(net->getOutputShape());
    }

    _neuron_num = neuron_num;
}

shape_t FC::getOutputShape()
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
// class NL
//=========================================================

NL::NL(std::string name, Net *src, NL::Type type)
    : Net(name, {src}), _type(type)
{
    _input_size = src->getOutputShape();
}

shape_t NL::getOutputShape()
{
    return _input_size;
}

//=========================================================
// class Pool
//=========================================================

Pool::Pool(std::string name, Net *src, Type type, shape_t pool_size, shape_t stride)
    : Net(name, {src})
{
    _input_size = src->getOutputShape();
    _pool_size = pool_size;
    _stride = stride;
    _type = type;
}

shape_t Pool::getOutputShape()
{
    return {
		_input_size[0],
        (_type == Type::GLOBAL) ? 1 : _input_size[1] / _stride[0],
        (_type == Type::GLOBAL) ? 1 : _input_size[2] / _stride[1]
    };
}

//=========================================================
// class EleWise
//=========================================================

EleWise::EleWise(std::string name, Net *src1, Net *src2)
    : Net(name, {src1, src2})
{
    assert(match(src1->getOutputShape(), src2->getOutputShape()));
    _input_size = src1->getOutputShape();
    _input_size.push_back(2);
}

shape_t EleWise::getOutputShape()
{
    return {_input_size[0], _input_size[1], _input_size[2]};
}