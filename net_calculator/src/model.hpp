#pragma once

#include <string>
#include "net.hpp"

class Model {
private:
    std::vector<Net *> _layers;
    std::vector<bool>  _flag;

public:
    Model();
    ~Model();
    template<typename T>
    Net *add(const T &layer);
    void Profile(std::string file_name);
}