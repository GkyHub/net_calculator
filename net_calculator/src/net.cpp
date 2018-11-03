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
    _fp_act_mask = -1.0;
    _bp_err_mask = -1.0;
}

void Net::forwardStaticSparsity()
{
    for (auto l : _src) {
        l->maskAct(_fp_act_mask);
    }
}

void Net::backwardStaticSparsity()
{
    for (auto l : _dst) {
        l->maskErr(_bp_err_mask);
    }
}

void Net::maskAct(double p)
{
    if (_fp_act_mask < 0) {
        _fp_act_mask = p;
    }
    else {
        _fp_act_mask = 1 - (1 - _fp_act_mask) * (1 - p);
    }    
}

void Net::maskErr(double p)
{
    if (_bp_err_mask < 0) {
        _bp_err_mask = p;
    }
    else {
        _bp_err_mask = 1 - (1 - _bp_err_mask) * (1 - p);
    }
}

//=========================================================
// class Input
//=========================================================
Input::Input(shape_t size) : Net(Net::Input, "input", {}) 
{ 
    _input_size = size; 
}

shape_t Input::getOutputShape() 
{ 
    return _input_size; 
}

//=========================================================
// class Conv2D
//=========================================================
Conv2D::Conv2D(std::string name, std::vector<Net *> src, shape_t param_size, 
    shape_t stride, double sparsity) : Net(Net::Conv2D, name, {src})
{
    assert(param_size.size() == 3);
    assert(stride.size() == 2);
    assert(sparsity > 0.0 && sparsity <= 1.0);
    _param_size = param_size;
    _stride = stride;
    _sparsity = sparsity;

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

void Conv2D::forwardStaticSparsity()
{
    for (auto l : _src) {
        l->maskAct(1.0);
    }
}

void Conv2D::backwardStaticSparsity()
{
    for (auto l : _dst) {
        l->maskErr(1.0);
    }
}

//=========================================================
// class FC
//=========================================================
FC::FC(std::string name, std::vector<Net *> src, uint32_t neuron_num, double sparsity)
    : Net(Net::FC, name, {src})
{
    // concat and flatten
    assert(sparsity > 0.0 && sparsity <= 1.0);
    _sparsity = sparsity;

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

void FC::forwardStaticSparsity()
{
    for (auto l : _src) {
        l->maskAct(1.0);
    }
}

void FC::backwardStaticSparsity()
{
    for (auto l : _dst) {
        l->maskErr(1.0);
    }
}

//=========================================================
// class NL
//=========================================================

NL::NL(std::string name, Net *src, type_t type)
    : Net(Net::NL, name, {src}), _nl_type(type)
{
    _input_size = src->getOutputShape();
}

shape_t NL::getOutputShape()
{
    return _input_size;
}

double NL::dynamicActSparsity()
{
    return (_type == RELU) ? 0.5 : 1.0;
}

void NL::backwardStaticSparsity()
{
    for (auto l : _dst) {
        l->maskErr((_type == RELU) ? (0.5 * _bp_err_mask) : _bp_err_mask);
    }
}

//=========================================================
// class Pool
//=========================================================

Pool::Pool(std::string name, Net *src, type_t type, shape_t pool_size, shape_t stride)
    : Net(NL::Pool, name, {src})
{
    _input_size = src->getOutputShape();
    _pool_size = pool_size;
    _stride = stride;
    _pool_type = type;
}

shape_t Pool::getOutputShape()
{
    return {
		_input_size[0],
        (_type == GLOBAL) ? 1 : _input_size[1] / _stride[0],
        (_type == GLOBAL) ? 1 : _input_size[2] / _stride[1]
    };
}

double Pool::dynamicErrSparsity()
{
    return (_type == MAX) ? (1.0 / (_stride[0] * _stride[1])) : 1.0;
}

void Pool::backwardStaticSparsity()
{
    for (auto l : _dst) {
        l->maskErr(1 - pow(1 - _bp_err_mask, 
            _stride[0] * _stride[1]));
    }
}

//=========================================================
// class EleWise
//=========================================================

EleWise::EleWise(std::string name, Net *src1, Net *src2)
    : Net(Net::EleWise, name, {src1, src2})
{
    assert(match(src1->getOutputShape(), src2->getOutputShape()));
    _input_size = src1->getOutputShape();
    _input_size.push_back(2);
}

shape_t EleWise::getOutputShape()
{
    return {_input_size[0], _input_size[1], _input_size[2]};
}

void EleWise::maskErr(double p)
{
    static double p_left = 0;
    if (_bp_err_mask < 0) {
        _bp_err_mask = 1.0;
        p_left = p;
    }
    else {
        _bp_err_mask = 1 - (1 - p_left) * (1 - p);
    }
}

//=========================================================
// class Dropout
//=========================================================

Dropout::Dropout(Net *src, double keep_prob)
    : Net(Net::Dropout, "dropout", {src})
{
    _fp_act_mask = keep_prob;
    _bp_err_mask = keep_prob;
}