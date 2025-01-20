#include "Color.h"

#include <iostream>

int main() {
    const int imageWidth = 256;
    const int imageHeight = 256;

    std::cout << "P3\n" << imageWidth << " " << imageHeight << "\n255\n";

    for(int i = 0; i < imageHeight; i++) {
        std::clog << "\rScanlines remaining: " << imageHeight - i << " " << std::flush; 
        for(int j = 0; j < imageWidth; j++) {
            write_color(std::cout, Color(static_cast<float>(j) / imageHeight, static_cast<float>(i) / imageWidth, 0.0f));
        }
    }

    std::clog << "\rDone.                 \n";

    return 0;
}