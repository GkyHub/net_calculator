#include "model.hpp"
#include "util.hpp"

Net *ResUnit(Module *m, Net *src, uint32_t width, uint32_t res_id)
{
    Net *conv1 = m->add(Conv2D(
       "conv" + std::to_string(res_id) + "_1",
       src,
       {width, 3, 3},
       {1, 1},
       nl_t::RELU 
    ));

    Net *conv2 = m->add(Conv2D(
       "conv" + std::to_string(res_id) + "_2",
       src,
       {width, 3, 3},
       {1, 1},
       nl_t::RELU 
    ));

    Net *res = m->add(EleWise(
        "res" + std::to_string(res_id),
        conv2, src
    ));

    return res;
}

Model *ResNet18()
{
    Model *resnet = new Model();

    Net *input = resnet.add(Input({3, 224, 224}));
    Net *conv1 = resnet.add(Conv2D("conv1", {input}, {64, 7, 7}, {2, 2}, nl_t:RELU));
    Net *pool1 = resnet.add(Pool("pool1", conv1, {2, 2}, {2, 2}));

    Net *res1[4];
    res1[0] = pool1;
    for (uint32_t i = 0; i < 3; i++) {
        res1[i + 1] = ResUnit(resnet, res1[i], 64, i + 2);
    }

    Net *res2[4]

    return resnet;
}