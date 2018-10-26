#pragma once
#include <string>
#include "typedef.hpp"

// A network prototype is a acyclic graph with possibly
// multiple sources and multiple destinations
class Net {
public:
    std::string         name;

protected:
    std::vector<Net *>  _src;           // source nets
    std::vector<Net *>  _dst;           // destination nets
    shape_t             _input_size;

public:
    Net(std::string name, std::vector<Net *> src);
    void    addDst(Net *dst);   // add a destination network
    virtual shape_t getOutputShape() = 0;

    // returns 0 by default, if a layer do not need MAC and parameter
    // do not need to override these methods
    virtual double  getInferenceMacNum() { return 0; };
    virtual double  getPropagationMacNum() { return 0; };
    virtual double  getUpdateMacNum() { return 0; };
    virtual double  getParamNum() { return 0; };
    virtual double  getInputSize() { return 0; };
};

// input layer
// name for input layer is always "input"
class Input : public Net {
public:
    Input(shape_t size);
    shape_t getOutputShape();
};

// 2D convolutional layer
// automatically concat previous layers
class Conv2D : public Net {
private:
    shape_t     _param_size;    // {C, H, W}
    shape_t     _stride;        // {H, W}

public:
    Conv2D(std::string name, std::vector<Net *> src, shape_t param_size, shape_t stride);
    shape_t getOutputShape();
    double  getParamNum();
    double  getInferenceMacNum();
    double  getPropagationMacNum();
    double  getUpdateMacNum();
};

// fully connected layer
// automatically flatten and concat previous layers
class FC : public Net {
private:
    uint32_t    _neuron_num;

public:
    FC(std::string name, std::vector<Net *> src, uint32_t neuron_num);
    shape_t getOutputShape();
    double  getParamNum();
    double  getInferenceMacNum();
    double  getPropagationMacNum();
    double  getUpdateMacNum();
};

class NL : public Net {
public:
    enum class Type {
        RELU, SIGMOID, TANH
    };

private:
    Type _type;

public:
    NL(std::string name, Net *src, NL::Type t);
    shape_t getOutputShape();
};

// Pooling layer
class Pool : public Net {
public:
    enum class Type {
        MAX, AVERAGE, GLOBAL
    };

private:
    shape_t     _pool_size; // {H, W}
    shape_t     _stride;    // {H, W}
    Type        _type;

public:
    Pool(std::string name, Net *src, Type type, shape_t pool_size, shape_t stride);
    shape_t getOutputShape();
};

// element-wise layer
// add two layers of the same shape together
class EleWise : public Net {
public:
    EleWise(std::string name, Net *src1, Net *src2);
    shape_t getOutputShape();
};




