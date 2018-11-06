#pragma once
#include <string>
#include "typedef.hpp"
#include "tensor.hpp"

// A network prototype is a acyclic graph with possibly
// multiple sources and multiple destinations
class Net {
public:
    enum type_t { Input, Conv2D, FC, NL, Pool, EleWise, Dropout };
    std::string         name;

protected:
    type_t              _type;
    std::vector<Net *>  _src;           // source nets
    std::vector<Net *>  _dst;           // destination nets
    Tensor              _input;         // activation tensor
    Tensor              _output;        // error tensor

public:
    Net(type_t type, std::string name, std::vector<Net *> src);
    type_t type()  { return _type; };
    Tensor input() { return _input; };
    Tensor output() { return _output; };

    // returns 0 by default, if a layer do not need MAC and parameter
    // do not need to override these methods
    virtual uint64_t  getInferenceMacNum()    { return 0; };
    virtual uint64_t  getPropagationMacNum()  { return 0; };
    virtual uint64_t  getUpdateMacNum()       { return 0; };
    virtual uint64_t  getParamNum()           { return 0; };
    virtual uint64_t  getInputSize()          { return 0; };

    // return the dynamic sparsity 
    virtual double  dynamicActSparsity() { return 1.0; };
    virtual double  dynamicErrSparsity() { return 1.0; };

    // infer forward static sparsity from dropout
    virtual void forwardStaticSparsity() {};
    // infer backward static sparsity from ReLU, pooling and dropout
    virtual void backwardStaticSparsity() {};

    virtual void maskAct(double p);
    virtual void maskErr(double p);
};

// input layer
// name for input layer is always "input"
class Input : public Net {
public:
    Input(shape_t size);
};

// 2D convolutional layer
// automatically concat previous layers
class Conv2D : public Net {
private:
    Tensor      _kernel;    // {C, H, W}
    shape_t     _stride;        // {H, W}

public:
    Conv2D(std::string name, std::vector<Net *> src, shape_t param_size, shape_t stride, float sparsity = 1.0);
    uint64_t  getParamNum();
    uint64_t  getInferenceMacNum();
    uint64_t  getPropagationMacNum();
    uint64_t  getUpdateMacNum();

    void forwardStaticSparsity();
    void backwardStaticSparsity();
};

// fully connected layer
// automatically flatten and concat previous layers
class FC : public Net {
private:
    uint32_t    _neuron_num;
    double      _sparsity;      // ratio of non-zero weights

public:
    FC(std::string name, std::vector<Net *> src, uint32_t neuron_num, double sparsity = 1);
    shape_t getOutputShape();
    double  getParamNum();
    double  getInferenceMacNum();
    double  getPropagationMacNum();
    double  getUpdateMacNum();

    void forwardStaticSparsity();
    void backwardStaticSparsity();

public:
    double _getSrcDynamicSparsity();
    double _getDstDynamicSparsity();
};

class NL : public Net {
public:
    enum type_t { RELU, SIGMOID, TANH };

private:
    type_t  _nl_type;

public:
    NL(std::string name, Net *src, type_t t);
    type_t  getNLType() { return _nl_type; };

    shape_t getOutputShape();
    double dynamicActSparsity();
    void backwardStaticSparsity();
};

// Pooling layer
class Pool : public Net {
public:
    enum type_t { MAX, AVERAGE, GLOBAL };

private:
    shape_t     _pool_size; // {H, W}
    shape_t     _stride;    // {H, W}
    type_t      _pool_type;

public:
    Pool(std::string name, Net *src, type_t type, shape_t pool_size, shape_t stride);
    type_t  getPoolType() { return _pool_type; };
    shape_t getOutputShape();

    double  dynamicErrSparsity();
    void backwardStaticSparsity();
};

// element-wise layer
// add two layers of the same shape together
class EleWise : public Net {
public:
    EleWise(std::string name, Net *src1, Net *src2);
    shape_t getOutputShape();

    void maskErr(double p);
};

// dropout layer
// drop a certain ratio of neurons in training
class Dropout : public Net {
public:
    Dropout(Net *src, double keep_prob);
    shape_t getOutputShape();
}




