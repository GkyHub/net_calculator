#include "model.hpp"
#include "util.hpp"

Model *VGG11()
{
    Model *vgg11 = new Model();
	Net *input = vgg11->add(Input({3, 224, 224}));

    Net *conv1 = vgg11->add(Conv2D("conv1", {input}, {64, 3, 3}, {1, 1}, nl_t::RELU));
    Net *pool1 = vgg11->add(Pool("pool1", conv1, {2, 2}, {2, 2}));

    Net *conv2 = vgg11->add(Conv2D("conv2", {pool1}, {128, 3, 3}, {1, 1}, nl_t::RELU));
    Net *pool2 = vgg11->add(Pool("pool2", conv2, {2, 2}, {2, 2}));

    Net *conv3_1 = vgg11->add(Conv2D("conv3_1", {pool2}, {256, 3, 3}, {1, 1}, nl_t::RELU));
    Net *conv3_2 = vgg11->add(Conv2D("conv3_2", {conv3_1}, {256, 3, 3}, {1, 1}, nl_t::RELU));
    Net *pool3 = vgg11->add(Pool("pool3", conv3_2, {2, 2}, {2, 2}));

    Net *conv4_1 = vgg11->add(Conv2D("conv4_1", {pool3}, {512, 3, 3}, {1, 1}, nl_t::RELU));
    Net *conv4_2 = vgg11->add(Conv2D("conv4_2", {conv4_1}, {512, 3, 3}, {1, 1}, nl_t::RELU));
    Net *pool4 = vgg11->add(Pool("pool4", conv4_2, {2, 2}, {2, 2}));

    Net *conv5_1 = vgg11->add(Conv2D("conv5_1", {pool4}, {512, 3, 3}, {1, 1}, nl_t::RELU));
    Net *conv5_2 = vgg11->add(Conv2D("conv5_2", {conv5_1}, {512, 3, 3}, {1, 1}, nl_t::RELU));
    Net *pool5 = vgg11->add(Pool("pool5", conv5_2, {2, 2}, {2, 2}));

    Net *fc1 = vgg11->add(FC("fc1", {pool5}, 4096, nl_t::RELU));
    Net *fc2 = vgg11->add(FC("fc2", {fc1}, 4096, nl_t::RELU));
    Net *fc3 = vgg11->add(FC("fc3", {fc2}, 1000, nl_t::RELU));

    return vgg11;
}

Model *VGG16()
{
    Model *vgg16 = new Model();
    return vgg16;
}

Model *VGG19()
{
    Model *vgg19 = new Model();
    return vgg19;
}