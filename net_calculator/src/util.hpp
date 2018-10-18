#pragma once

// convert a number to string and add a unit (G, M, B) to it
std::string unit_str(double n);

// concat layer b to layer a, add the first dimension
// return false if the size does not match
bool concat(tsize_t a, tsize_t b);

// calculate the volume of a tensor
uint32_t volume(tsize_t s);
