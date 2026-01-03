#pragma once

#include "Utility.h"

class Interval {
  public:
    double min, max;

    static constexpr bool overlap(const Interval& i1, const Interval& i2) {
        return i1.min <= i2.max && i2.max >= i1.min;
    }

    static constexpr Interval intersection(const Interval& i0, const Interval& i1) {
        return Interval(std::max(i0.min, i1.min), std::min(i0.max, i1.max));
    }

    static constexpr Interval merge(const Interval& i0, const Interval& i1) {
        return Interval(std::min(i0.min, i1.min), std::max(i0.max, i1.max));
    }

    constexpr Interval() noexcept : min(POSITIVE_INFINITY), max(-POSITIVE_INFINITY) {} // Default interval is empty

    constexpr Interval(const double _min, const double _max) noexcept : min(_min), max(_max) {}

    constexpr Interval(const Interval& i0, const Interval& i1) noexcept : Interval(merge(i0, i1)) {}

    double size() const noexcept {
        return max - min;
    }

    bool contains(const double x) const noexcept {
        return min <= x && x <= max;
    }

    bool surrounds(const double x) const noexcept {
        return min < x && x < max;
    }

    bool empty() const noexcept {
        return min > max;
    }

    double clamp(const double x) const noexcept {
        return std::max(this->min, std::min(x, this->max));
    }

    Interval expand(const double padding) const noexcept {
        const double delta = padding / 2;
        return Interval(min - delta, max + delta);
    }

    bool operator<(const Interval& other) {
        return min < other.min;
    }

    static const Interval EMPTY, UNIVERSE;
};

const Interval Interval::EMPTY    = Interval(POSITIVE_INFINITY, -POSITIVE_INFINITY);
const Interval Interval::UNIVERSE = Interval(-POSITIVE_INFINITY, POSITIVE_INFINITY);