#include "resnet.hpp"
#include <functional>

ResNet::ResNet(uint32_t size[4], uint32_t num[4], bool use_bottleneck)
{
    std::function<Net *(std::string, Net *, uint32_t, uint32_t)> res_unit;
    res_unit = use_bottleneck ? this->Bottleneck : this->PlainBlock;
    Net *x = add(Input({3, 224, 224}));

    x = add(Conv2D("conv1", {x}, {size[0], 7, 7}, {2, 2}));
    x = add(NL("relu1", x, NL::Type::RELU));
    x = add(Pool("pool1", x, Pool::Type::MAX, {2, 2}, {2, 2}));
    
    for (uint32_t i = 0; i < 4; i++) {
        for (uint32_t j = 0; j < num[i]; j++) {
            x = res_unit(
                "res" + std::to_string(i) + std::to_string(j),
                x, 
                size[i],
                ((i > 0 && j == 0) ? 2 : 1) // the first layer of each block downsamples
                );
        }
    }
    x = add(Pool("avg_pool", x, Pool::Type::GLOBAL, {1, 1}, {0, 0}));
    x = add(FC("fc", {x}, 1000));
}

ResNet::~ResNet()
{
    for (Net *l : _layers) {
        delete l;
    }
}

Net *ResNet::PlainBlock(std::string name, Net *src, uint32_t out_channel, uint32_t stride)
{
    Net *x = src;
    Net *shortcut = src;
    tsize_t s = {stride, stride};

    x = add(Conv2D(name + "_conv1", {x}, {out_channel, 3, 3}, s));
    x = add(NL(name + "_relu1", x, NL::Type::RELU));
    x = add(Conv2D(name + "_conv2", {x}, {out_channel, 3, 3}, {1, 1}));

    if (!match(x->getOutputSize(), src->getOutputSize())) {
        shortcut = add(Conv2D(name + "_short", {src}, {out_channel, 1, 1}, s));
    }

    x = add(EleWise(name + "_add", x, shortcut));
    x = add(NL(name + "_relu2", x, NL::Type::RELU));

    return x;
}

Net *ResNet::Bottleneck(std::string name, Net *src, uint32_t out_channel, uint32_t stride)
{
    Net *x = src;
    Net *shortcut = src;
    tsize_t s = {stride, stride};

    x = add(Conv2D(name + "_conv1", {x}, {out_channel / 4, 1, 1}, s));
    x = add(NL(name + "_relu1", x, NL::Type::RELU));
    x = add(Conv2D(name + "_conv2", {x}, {out_channel / 4, 3, 3}, {1, 1}));
    x = add(NL(name + "_relu2", x, NL::Type::RELU));
    x = add(Conv2D(name + "_conv3", {x}, {out_channel, 1, 1}, {1, 1}));

    if (!match(x->getOutputSize(), src->getOutputSize())) {
        shortcut = add(Conv2D(name + "_short", {src}, {out_channel, 1, 1}, s));
    }

    x = add(EleWise(name + "_add", x, shortcut));
    x = add(NL(name + "_relu3", x, NL::Type::RELU));

    return x;
}