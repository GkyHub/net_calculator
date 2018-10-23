#pragma once

#include <string>
#include "net.hpp"

class Model {
protected:
    std::vector<Net *> _layers;
    std::vector<bool>  _flag;

public:
    Model();
    ~Model();

    template<typename T>
    Net *add(const T &layer)
	{
		Net *l = (Net *)(new T(layer));
		_layers.push_back(l);
		return l;
	}

    void Profile(std::string file_name);
};