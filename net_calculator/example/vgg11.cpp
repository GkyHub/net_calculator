#include "vgg11.hpp"

VGG11::VGG11(tsize_t input_size, uint32_t class_num)
{
    _layers.push_back(new Input(input_size));
}