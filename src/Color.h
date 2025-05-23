#pragma once

#include "Vec3.h"
#include "Interval.h"

#include <iostream>

using Color = Vec3;

static inline double linearToGamma(const double linearValue) {
    if (linearValue > 0.0) {
        return std::sqrt(linearValue);
    }
    return 0.0;
}

void write_color(std::ostream& out, const Color& pixel_color) {
    auto r = pixel_color.r();
    auto g = pixel_color.g();
    auto b = pixel_color.b();

    r = linearToGamma(r);
    g = linearToGamma(g);
    b = linearToGamma(b);

    // Translate the [0,1] component values to the byte range [0,255].
    constexpr Interval interval(0.0, 0.999);

    int rByte = static_cast<int>(256 * interval.clamp(r));
    int gByte = static_cast<int>(256 * interval.clamp(g));
    int bByte = static_cast<int>(256 * interval.clamp(b));

    // Write out the pixel color components.
    out << rByte << ' ' << gByte << ' ' << bByte << '\n';
}