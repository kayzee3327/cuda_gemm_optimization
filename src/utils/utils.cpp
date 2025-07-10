#include "utils.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

#include <unistd.h>
#include <string.h>



void read_mat(std::string matname, int& r, int& c, int& b, uint8_t*& matret) {
    int rows, cols, bits;

    std::string matpath = "./data/" + matname + ".bin";

    std::ifstream infile(matpath, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file!" << std::endl;
        exit(EXIT_FAILURE);
    }

    infile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    // std::cout << "bytes read: " << infile.gcount() << std::endl;
    infile.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    // std::cout << "bytes read: " << infile.gcount() << std::endl;
    infile.read(reinterpret_cast<char*>(&bits), sizeof(bits));
    // std::cout << "bytes read: " << infile.gcount() << std::endl;

    // std::cout << rows << ' ' << cols << ' ' << bits << std::endl;
    
    r = rows;
    c = cols;
    b = bits;
    delete[] matret;

    if (bits == 32)
    {
        matret = new uint8_t[rows * cols * sizeof(uint32_t)];
        infile.read(reinterpret_cast<char*>(matret), rows * cols * sizeof(uint32_t));
    }
    else if (bits == 16) {
        matret = new uint8_t[rows * cols * sizeof(uint16_t)];
        infile.read(reinterpret_cast<char*>(matret), rows * cols * sizeof(uint16_t));
    }
    else if (bits == 0) {
        matret = new uint8_t[rows * cols * sizeof(uint32_t)];
        infile.read(reinterpret_cast<char*>(matret), rows * cols * sizeof(uint32_t));
    }
    


    infile.close();
}

void write_mat(std::string matname, int r, int c, int b, uint8_t* mat) {
    std::string matpath = "./data/" + matname + ".bin";
    std::ofstream outfile(matpath, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file!" << std::endl;
        exit(EXIT_FAILURE);
    }

    outfile.write(reinterpret_cast<char*>(&r), sizeof(r));
    outfile.write(reinterpret_cast<char*>(&c), sizeof(c));
    outfile.write(reinterpret_cast<char*>(&b), sizeof(b));


    if (b == 32)
    {
        outfile.write(reinterpret_cast<char*>(mat), r * c * sizeof(uint32_t));
    }
    else if (b == 16)
    {
        outfile.write(reinterpret_cast<char*>(mat), r * c * sizeof(uint16_t));
    }
    else if (b == 0)
    {
        outfile.write(reinterpret_cast<char*>(mat), r * c * sizeof(uint32_t));
    }
    
    outfile.close();
}
