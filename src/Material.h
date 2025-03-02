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
        attenuation = this->albedo;
        return true;
    }

private:
    Color albedo;
};

class Metal : public Material {
public:
    Metal(const Color& albedo, const double fuzz) : albedo(albedo), fuzz(std::min(1.0, fuzz)) {}

    bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const override {
        auto scatteringDirection = Vec3::reflect(rayIn.direction(), hitInfo.getNormal());
        scatteringDirection = scatteringDirection.normalized() + this->fuzz * Vec3::randomUnitVector();
        scatteredRay = Ray(hitInfo.p, scatteringDirection);
        attenuation = this->albedo;
        return Vec3::dot(hitInfo.getNormal(), scatteringDirection) > 0.0;
    }

private:
    Color albedo;
    double fuzz;
};

class Dielectric : public Material {
    public:
        Dielectric(const double refractiveIndex) : refractiveIndex(refractiveIndex) {}

        bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay) const override {
            double etaSurroundingOverEtaMaterial = hitInfo.getFront() ? 1.0 / this->refractiveIndex : this->refractiveIndex;
            auto scatteringDirection = Vec3::refract(rayIn.direction().normalized(), hitInfo.getNormal(), etaSurroundingOverEtaMaterial);
            scatteredRay = Ray(hitInfo.p, scatteringDirection);
            attenuation = Color(1.0);
            return true;
        }

    private:
        double refractiveIndex;
};