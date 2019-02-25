// VGG model
// reference: https://arxiv.org/pdf/1409.1556.pdf
//
// example:
// VGG vgg11 = VGG({1, 1, 2, 2, 2});
// VGG vgg16 = VGG({2, 2, 3, 3, 3});
// VGG vgg19 = VGG({2, 2, 4, 4, 4});

#include "../src/model.hpp"
#include "../src/util.hpp"


class VGG : public Model {
public:
    VGG(std::vector<uint32_t> num)
    {
        uint32_t size[5] = {64, 128, 256, 512, 512};

        Net *x = add(Input({3, 224, 224}));
        for (uint32_t i = 0; i < 5; i++) {
            for (uint32_t j = 0; j < num[i]; j++) {
                x = add(Conv2D(
                    "conv" + std::to_string(i + 1) + std::to_string(j + 1),
                    {x},
                    {size[i], 3, 3},
                    {1, 1}
                ));
                x = add(NL(
                    "relu" + std::to_string(i + 1) + std::to_string(j + 1),
                    x,
                    NL::RELU
                ));
            }

            x = add(Pool("pool" + std::to_string(i + 1), x, Pool::MAX, {2, 2}, {2, 2}));
        }

        x = add(FC("fc1", {x}, 4096));
        x = add(NL("relu6", x, NL::RELU));
        x = add(FC("fc2", {x}, 4096));
        x = add(NL("relu7", x, NL::RELU));
        x = add(FC("fc3", {x}, 1000));
        x = add(NL("relu8", x, NL::RELU));
    }
};