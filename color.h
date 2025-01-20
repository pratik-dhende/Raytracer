#ifndef COLOR_H
#define COLOR_H

#include "Vec3.h"

#include <iostream>

using Color = Vec3;

void write_color(std::ostream& out, const Color& pixel_color) {
    auto r = pixel_color.r();
    auto g = pixel_color.g();
    auto b = pixel_color.b();

    // Translate the [0,1] component values to the byte range [0,255].
    int rByte = static_cast<int>(255.999 * r);
    int gByte = static_cast<int>(255.999 * g);
    int bByte = static_cast<int>(255.999 * b);

    // Write out the pixel color components.
    out << rByte << ' ' << gByte << ' ' << bByte << '\n';
}

#endif