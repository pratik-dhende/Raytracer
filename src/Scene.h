#pragma once

#include "Raytracer.h"
#include "Hittable.h"

#include <vector>
#include <memory>
#include <cassert>

class Scene : public Hittable {
public:
    __device__ Scene(const int size) : freeIndex(0), size(size){
        hittables = new Hittable*[size];
    }

    __device__ void add(Hittable* hittable) {
        assert(freeIndex < size);

        hittables[freeIndex++] = hittable;
    }

    __device__ bool hit(const Ray& ray, const Interval& rayTInterval, HitInfo& hitInfo) const override {
        double closestT = rayTInterval.max;
        HitInfo tempHitInfo;
        bool anyHit = false;

        for(int i = 0; i < freeIndex; ++i) {
            if (hittables[i]->hit(ray, Interval(rayTInterval.min, closestT), tempHitInfo)) {
                closestT = tempHitInfo.t;
                hitInfo = tempHitInfo;
                anyHit = true;
            }
        }

        return anyHit;
    }

    __device__ ~Scene() {
        delete[] hittables;
    }

private:
    Hittable** hittables;
    int freeIndex;
    int size;
};