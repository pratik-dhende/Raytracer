#ifndef COLOR_H
#define COLOR_H

#include "vec3f.h"

#include <iostream>

using color = vec3f;

void write_color(std::ostream& out, const color& pixel_color) {
    auto r = pixel_color.r();
    auto g = pixel_color.g();
    auto b = pixel_color.b();

    // Translate the [0,1] component values to the byte range [0,255].
    int rByte = int(255.999 * r);
    int gByte = int(255.999 * g);
    int bByte = int(255.999 * b);

    // Write out the pixel color components.
    out << rByte << ' ' << gByte << ' ' << bByte << '\n';
}

#endif