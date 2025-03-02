#pragma once

#include "Utility.h"

class Interval {
  public:
    double min, max;

    Interval() : min(POSITIVE_INFINITY), max(-POSITIVE_INFINITY) {} // Default interval is empty

    constexpr Interval(const double _min, const double _max) : min(_min), max(_max) {}

    double size() const {
        return max - min;
    }

    bool contains(const double x) const {
        return min <= x && x <= max;
    }

    bool surrounds(const double x) const {
        return min < x && x < max;
    }

    double clamp(const double x) const {
        return std::max(this->min, std::min(x, this->max));
    }

    static const Interval EMPTY, UNIVERSE;
};

const Interval Interval::EMPTY    = Interval(POSITIVE_INFINITY, -POSITIVE_INFINITY);
const Interval Interval::UNIVERSE = Interval(-POSITIVE_INFINITY, POSITIVE_INFINITY);