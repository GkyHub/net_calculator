#include<iostream>
#include<vector>

typedef int             int32_t;
typedef short           int16_t;
typedef char            int8_t;
typedef unsigned int    uint32_t;
typedef unsigned short  uint16_t;
typedef unsigned char   uint8_t;

typedef std::vector<uint32_t> tsize_t;  // tensor size

typedef enum {
    RELU, SIGMOID, TANH
} nl_t; 