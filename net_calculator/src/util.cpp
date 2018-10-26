#include <string>
#include "typedef.hpp"
#include "util.hpp"

// convert a number to string and add a unit (G, M, B) to it
std::string unit_str(double n)
{
    if (n > 1e9) {
        return (std::to_string(n / 1e9) + "G");
    }
    else if (n > 1e6) {
        return (std::to_string(n / 1e6) + "M");
    }
    else if (n > 1e3) {
        return (std::to_string(n / 1e3) + "K");
    }
    else {
        return std::to_string(n);
    }
}

// concat layer b to layer a, add the first dimension
// return false if the size does not match
bool concat(shape_t a, shape_t b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (uint32_t i = 1; i < a.size(); i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    a[0] += b[0];
    return true;
}

// calculate the volume of a tensor
uint32_t volume(shape_t s)
{
    uint32_t v = 1;
    for (auto dim : s) {
        v *= dim;
    }
    return v;
}

bool match(shape_t s1, shape_t s2)
{
	if (s1.size() != s2.size()) {
		return false;
	}
	for (uint32_t i = 0; i < s1.size(); i++) {
		if (s1[i] != s2[i]) {
			return false;
		}
	}
	return true;
}


