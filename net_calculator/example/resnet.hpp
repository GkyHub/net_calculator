// ResNet v1 series models
// reference: https://github.com/tensorflow/models/tree/master/official/resnet
//
// examples:
// ResNet resnet18 = ResNet({2, 2, 2, 2}, {64, 128, 256, 512});
// ResNet resnet34 = ResNet({3, 4, 6, 3}, {64, 128, 256, 512});
// ResNet resnet50 = ResNet({3, 4, 6, 3}, {256, 512, 1024, 2048}, true);
// ResNet resnet101 = ResNet({3, 4, 23, 3}, {256, 512, 1024, 2048}, true);
// ResNet resnet152 = ResNet({3, 8, 36, 3}, {256, 512, 1024, 2048}, true);

#include "../src/model.hpp"
#include "../src/util.hpp"
#include <functional>

class ResNet : public Model {
public:
	ResNet(std::vector<uint32_t> num, std::vector<uint32_t> size, bool use_bottleneck = false)
    {
        //std::function<Net *(std::string, Net *, uint32_t, uint32_t)> res_unit;
        // auto res_unit = use_bottleneck ? &ResNet::Bottleneck : &ResNet::PlainBlock;
		auto res_unit = [&](std::string name, Net *x, uint32_t channel, uint32_t stride = 1)
		{
			if (use_bottleneck) {
				return Bottleneck(name, x, channel, stride);
			}
			else {
				return PlainBlock(name, x, channel, stride);
			}
		};

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

    // basic block
    Net *PlainBlock(std::string name, Net *src, uint32_t out_channel, uint32_t stride = 1)
    {
        Net *x = src;
        Net *shortcut = src;
        shape_t s = {stride, stride};

        x = add(Conv2D(name + "_conv1", {x}, {out_channel, 3, 3}, s));
        x = add(NL(name + "_relu1", x, NL::Type::RELU));
        x = add(Conv2D(name + "_conv2", {x}, {out_channel, 3, 3}, {1, 1}));

        if (!match(x->getOutputShape(), src->getOutputShape())) {
            shortcut = add(Conv2D(name + "_short", {src}, {out_channel, 1, 1}, s));
        }

        x = add(EleWise(name + "_add", x, shortcut));
        x = add(NL(name + "_relu2", x, NL::Type::RELU));

        return x;
    }

    // bottleneck block
    Net *Bottleneck(std::string name, Net *src, uint32_t out_channel, uint32_t stride = 1)
    {
        Net *x = src;
        Net *shortcut = src;
        shape_t s = {stride, stride};

        x = add(Conv2D(name + "_conv1", {x}, {out_channel / 4, 1, 1}, s));
        x = add(NL(name + "_relu1", x, NL::Type::RELU));
        x = add(Conv2D(name + "_conv2", {x}, {out_channel / 4, 3, 3}, {1, 1}));
        x = add(NL(name + "_relu2", x, NL::Type::RELU));
        x = add(Conv2D(name + "_conv3", {x}, {out_channel, 1, 1}, {1, 1}));

        if (!match(x->getOutputShape(), src->getOutputShape())) {
            shortcut = add(Conv2D(name + "_short", {src}, {out_channel, 1, 1}, s));
        }

        x = add(EleWise(name + "_add", x, shortcut));
        x = add(NL(name + "_relu3", x, NL::Type::RELU));

        return x;
    }
};

