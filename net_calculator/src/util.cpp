#include <string>
#include "typedef.hpp"

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
bool concat(tsize_t a, tsize_t b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (int i = 1; i < a.size(); i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    a[0] += b[0];
    return true;
}

// calculate the volume of a tensor
uint32_t volume(tsize_t s)
{
    uint32_t v = 1;
    for (auto dim : s) {
        v *= dim;
    }
    return v;
}