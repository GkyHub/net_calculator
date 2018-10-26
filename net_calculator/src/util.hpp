#pragma once
#include "typedef.hpp"
#include <string>

// convert a number to string and add a unit (G, M, B) to it
std::string unit_str(double n);

// concat layer b to layer a, add the first dimension
// return false if the size does not match
bool concat(shape_t a, shape_t b);

// calculate the volume of a tensor
uint32_t volume(shape_t s);

// match if two tensors are of the same shape
bool match(shape_t s1, shape_t s2);

// csv format
template<typename T1>
void csvPrintLn(std::ostream &os, const T1 &arg1)
{
	os << arg1 << std::endl;
}

template<typename T1, typename... T2>
void csvPrintLn(std::ostream &os, const T1 &arg1, T2... arg2)
{
	os << arg1 << ",";
	csvPrintLn(os, arg2...);
}

template<typename T>
T sum(std::vector<T> vec)
{
	T s = 0;
	for (T &e : vec) {
		s += e;
	}
	return s;
}
