#include <iostream>
#include "utils.h"


int main() {
    // float a = 1.0;
    // short b = 1.0;
    // std::cout << sizeof(a) << std::endl;
    // std::cout << sizeof(b) << std::endl;
    int r,c,b;
    uint8_t* m;
    read_mat("matrixA",r,c,b, m);

    return 0;
}