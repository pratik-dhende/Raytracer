#pragma once

#include <limits>

inline constexpr float F_INFINITY = std::numeric_limits<float>::infinity();
inline constexpr float F_PI = 3.1415926535897932385f;

inline float degreesToRadians(const float degree) {
    return degree * F_PI / 180.0f;
}