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
    tsize_t             _input_size;

public:
    Net(std::string name, std::vector<Net *> src);
    void    addDst(Net *dst);   // add a destination network
    virtual tsize_t getOutputSize() = 0;

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
    Input(tsize_t size);
    tsize_t getOutputSize();
};

// 2D convolutional layer
// automatically concat previous layers
class Conv2D : public Net {
private:
    tsize_t     _param_size;    // {C, H, W}
    tsize_t     _stride;        // {H, W}
    nl_t        _nl;

public:
    Conv2D(std::string name, std::vector<Net *> src, tsize_t param_size, tsize_t stride, nl_t nl);
    tsize_t getOutputSize();
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
    nl_t        _nl;

public:
    FC(std::string name, std::vector<Net *> src, uint32_t neuron_num, nl_t nl);
    tsize_t getOutputSize();
    double  getParamNum();
    double  getInferenceMacNum();
    double  getPropagationMacNum();
    double  getUpdateMacNum();
};

// Pooling layer
class Pool : public Net {
private:
    tsize_t     _pool_size; // {H, W}
    tsize_t     _stride;    // {H, W}

public:
    Pool(std::string name, Net *src, tsize_t pool_size, tsize_t stride);
    tsize_t getOutputSize();
};

// element-wise layer
// add two layers of the same shape together
class EleWise : public Net {
public:
    EleWise(std::string name, Net *src1, Net *src2)
    tsize_t getOutputSize();
};




