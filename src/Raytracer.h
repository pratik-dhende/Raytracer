#pragma once

#include "Vec3f.h"
#include "Ray.h"
#include "Color.h"

#include <vector>
#include <memory>
#include <iostream>
#include <cmath>
#include <limits>

constexpr float F_INFINITY = std::numeric_limits<float>::infinity();
constexpr float F_PI = 3.1415926535897932385f;

inline float degreesToRadians(const float degree) {
    return degree * F_PI / 180.0f;
}
