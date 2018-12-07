#pragma once
#include <string>
#include "typedef.hpp"
#include "tensor.hpp"

// A network prototype is a acyclic graph with possibly
// multiple sources and multiple destinations
class Net {
public:
    enum type_t { Input, Conv2D, FC, NL, Pool, EleWise };
    std::string         name;

protected:
    type_t              _type;
    std::vector<Net *>  _src;           // source nets
    std::vector<Net *>  _dst;           // destination nets
    Tensor              _output;        // error tensor

public:
    Net(type_t type, std::string name, std::vector<Net *> src);
    type_t type()   { return _type; };
    Tensor output() { return _output; };
    virtual Tensor input();

    // returns 0 by default, if a layer do not need MAC and parameter
    // do not need to override these methods
    virtual uint64_t  getInferenceMacNum()    { return 0; };
    virtual uint64_t  getPropagationMacNum()  { return 0; };
    virtual uint64_t  getUpdateMacNum()       { return 0; };
    virtual uint64_t  getParamNum()           { return 0; };

    // return the ratio of computed zeros
    virtual float fpActSparsity() { return 1.0; };
    virtual float bpErrSparsity() { return 1.0; };

    // drop out function
    virtual void dropout(float nz);
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
    Tensor      _input;     // {C, H, W} just to avoid concat previous layers 
                            // at run time
    Tensor      _kernel;    // {C, H, W}
    shape_t     _stride;    // {H, W}

public:
    Conv2D(std::string name, std::vector<Net *> src, shape_t param_size, shape_t stride, float sparsity = 1.0);
    uint64_t  getParamNum();
    uint64_t  getInferenceMacNum();
    uint64_t  getPropagationMacNum();
    uint64_t  getUpdateMacNum();

    Tensor input() { return _input; };
};

// fully connected layer
// automatically flatten and concat previous layers
class FC : public Net {
private:
    Tensor _weights;        // the weight tensor

public:
    FC(std::string name, std::vector<Net *> src, uint32_t neuron_num, double sparsity = 1);
    uint64_t getParamNum();
    uint64_t getInferenceMacNum();
    uint64_t getPropagationMacNum();
    uint64_t getUpdateMacNum();
};

class NL : public Net {
public:
    enum type_t { RELU, SIGMOID, TANH };

private:
    type_t  _nl_type;

public:
    NL(std::string name, Net *src, type_t t);
    type_t  getNLType() { return _nl_type; };

    void dropout(float nz);
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

    void dropout(float nz);
};

// element-wise layer
// add two layers of the same shape together
class EleWise : public Net {
public:
    EleWise(std::string name, Net *src1, Net *src2);
    void dropout(float nz);
};




