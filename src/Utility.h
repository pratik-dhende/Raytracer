#pragma once

#include <limits>
#include <random>

inline constexpr double POSITIVE_INFINITY = std::numeric_limits<double>::infinity();
inline constexpr double PI = 3.1415926535897932385;

static inline double degreesToRadians(const double degree) {
    return degree * PI / 180.0;
}

static inline double randomDouble() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator(17);
    return distribution(generator);
}

static inline double randomDouble(double min, double max) {
    return min + (max - min) * randomDouble();
}

static inline int randomInt(const int min, const int max) {
    return static_cast<int>(randomDouble(static_cast<double>(min), static_cast<double>(max + 1)));
}
