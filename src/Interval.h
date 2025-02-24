#pragma once

#include "Utility.h"

class Interval {
  public:
    float min, max;

    Interval() : min(F_INFINITY), max(-F_INFINITY) {} // Default interval is empty

    Interval(const float _min, const float _max) : min(_min), max(_max) {}

    float size() const {
        return max - min;
    }

    bool contains(const float x) const {
        return min <= x && x <= max;
    }

    bool surrounds(const float x) const {
        return min < x && x < max;
    }

    static const Interval EMPTY, UNIVERSE;
};

const Interval Interval::EMPTY    = Interval(F_INFINITY, -F_INFINITY);
const Interval Interval::UNIVERSE = Interval(-F_INFINITY, F_INFINITY);