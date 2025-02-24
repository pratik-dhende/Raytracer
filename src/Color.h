#pragma once

#include "Vec3f.h"
#include "Interval.h"

#include <iostream>

using Color = Vec3f;

void write_color(std::ostream& out, const Color& pixel_color) {
    auto r = pixel_color.r();
    auto g = pixel_color.g();
    auto b = pixel_color.b();

    // Translate the [0,1] component values to the byte range [0,255].
    constexpr Interval interval(0.0f, 1.0f);

    int rByte = static_cast<int>(255.999 * interval.clamp(r));
    int gByte = static_cast<int>(255.999 * interval.clamp(g));
    int bByte = static_cast<int>(255.999 * interval.clamp(b));

    // Write out the pixel color components.
    out << rByte << ' ' << gByte << ' ' << bByte << '\n';
}