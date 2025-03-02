#pragma once

#include <limits>
#include <random>

inline constexpr double POSITIVE_INFINITY = std::numeric_limits<double>::infinity();
inline constexpr double PI = 3.1415926535897932385;

static inline double degreesToRadians(const double degree) {
    return degree * PI / 180.0;
}

static inline double random() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

static inline double random(double min, double max) {
    return min + (max - min) * random();
}
