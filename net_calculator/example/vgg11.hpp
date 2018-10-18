#pragma once

#include "net.hpp"
#

class VGG11 {
private:
    std::vector<Net *> _layers;

public:
    VGG11(tsize_t input_size, uint32_t class_num);
    ~VGG11();
};