// example model of ResNet v1 series
// reference: https://github.com/tensorflow/models/tree/master/official/resnet

#include "model.hpp"
#include "util.hpp"

class ResNet : public Model {
public:
    ResNet(uint32_t size[4], uint32_t num[4], bool use_bottleneck = false);
    ~ResNet();
    // basic block
    Net *PlainBlock(std::string name, Net *src, uint32_t out_channel, uint32_t stride = 1);
    // bottleneck block
    Net *Bottleneck(std::string name, Net *src, uint32_t out_channel, uint32_t stride = 1);
};

// examples
// resnet18 = ResNet({2, 2, 2, 2}, {64, 128, 256, 512});
// resnet34 = ResNet({3, 4, 6, 3}, {64, 128, 256, 512});
// resnet50 = ResNet({3, 4, 6, 3}, {256, 512, 1024, 2048}, true);
// resnet101 = ResNet({3, 4, 23, 3}, {256, 512, 1024, 2048}, true);
// resnet152 = ResNet({3, 8, 36, 3}, {256, 512, 1024, 2048}, true);