#pragma once

#include <limits>
#include <random>

inline constexpr float F_INFINITY = std::numeric_limits<float>::infinity();
inline constexpr float F_PI = 3.1415926535897932385f;

inline float degreesToRadians(const float degree) {
    return degree * F_PI / 180.0f;
}

inline float random() {
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator;
    return distribution(generator);
}

inline float random(float min, float max) {
    return min + (max - min) * random();
}