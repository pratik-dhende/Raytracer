#pragma once

#include "Vec3.h"
#include "Color.h"
#include "Hittable.h"
#include "Ray.h"

class Material {
public:
    virtual ~Material() = default;

    virtual bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const {
        return false;
    }
};

class Lambertian : public Material {
public:
    Lambertian(const Color& albedo) : albedo(albedo) {}

    bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const override {
        auto scatteringDirection = hitInfo.getNormal() + Vec3::randomUnitVector();

        if (scatteringDirection.nearZero()) {
            scatteringDirection = hitInfo.getNormal();
        }

        scatteredRay = Ray(hitInfo.p, scatteringDirection);
        attenuation = albedo;
        return true;
    }

private:
    Color albedo;
};

class Metal : public Material {
public:
    Metal(const Color& albedo) : albedo(albedo) {}

    bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const override {
        auto scatteringDirection = Vec3::reflect(rayIn.direction(), hitInfo.getNormal());

        if (scatteringDirection.nearZero()) {
            scatteringDirection = hitInfo.getNormal();
        }

        scatteredRay = Ray(hitInfo.p, scatteringDirection);
        attenuation = albedo;
        return true;
    }

private:
    Color albedo;
};