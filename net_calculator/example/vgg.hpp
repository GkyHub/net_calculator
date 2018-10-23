// VGG model
// reference: https://arxiv.org/pdf/1409.1556.pdf
//
// example:
// VGG vgg11 = VGG({1, 1, 2, 2, 2});
// VGG vgg16 = VGG({2, 2, 3, 3, 3});
// VGG vgg19 = VGG({2, 2, 4, 4, 4});

#include "model.hpp"
#include "util.hpp"


class VGG : public Model {
public:
    VGG(uint32_t num[5])
    {
        uint32_t size[5] = {64, 128, 256, 512, 512};

        Net *x = add(Input({3, 224, 224}));
        for (uint32_t i = 1; i < 6; i++) {
            for (uint32_t j = 1; j <= num[i]; j++) {
                x = add(Conv2D(
                    "conv" + std::to_string(i) + std::to_string(j),
                    {x},
                    {size[i], 3, 3},
                    {1, 1}
                ));
                x = add(NL(
                    "relu" + std::to_string(i) + std::to_string(j),
                    x,
                    NL::Type::RELU
                ));
            }

            x = add(Pool("pool" + std::to_string(i), x, Pool::Type::MAX, {2, 2}, {2, 2}));
        }

        x = add(FC("fc1", {x}, 4096));
        x = add(NL("relu6", x, NL::Type::RELU));
        x = add(FC("fc2", {x}, 4096));
        x = add(NL("relu7", x, NL::Type::RELU));
        x = add(FC("fc3", {x}, 1000));
        x = add(NL("relu8", x, NL::Type::RELU));
    }
};