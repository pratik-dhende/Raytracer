#pragma once

#include "Utility.h"
#include "Cuda.h"

class Interval {
  public:
    double min, max;

    __host__ __device__ Interval() : min(POSITIVE_INFINITY), max(-POSITIVE_INFINITY) {} // Default interval is empty

    __host__ __device__ Interval(const double _min, const double _max) : min(_min), max(_max) {}

    __host__ __device__ double size() const {
        return max - min;
    }

    __host__ __device__ bool contains(const double x) const {
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(const double x) const {
        return min < x && x < max;
    }

    __host__ __device__ double clamp(const double x) const {
        return Cuda::max(this->min, Cuda::min(x, this->max));
    }

    static const Interval EMPTY, UNIVERSE;
};

const Interval Interval::EMPTY    = Interval(POSITIVE_INFINITY, -POSITIVE_INFINITY);
const Interval Interval::UNIVERSE = Interval(-POSITIVE_INFINITY, POSITIVE_INFINITY);