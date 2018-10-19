#pragma once
#include "typedef.hpp"

// convert a number to string and add a unit (G, M, B) to it
std::string unit_str(double n);

// concat layer b to layer a, add the first dimension
// return false if the size does not match
bool concat(tsize_t a, tsize_t b);

// calculate the volume of a tensor
uint32_t volume(tsize_t s);

// csv format
template<typename T1, typename... T2>
void csvPrintLn(std::ostream &os, T1 arg1, T2... arg2)
{
    os << arg1 << ",";
    csvPrintLn(os, arg2);
}

void csvPrintLn(std::ostream &os)
{
    os << std::endl;
}

