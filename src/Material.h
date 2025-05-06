#pragma once

#include "Vec3.h"
#include "Color.h"
#include "Hittable.h"
#include "Ray.h"
#include "Utility.h"

class Material {
public:
    __device__ virtual ~Material() {};

    __device__ virtual bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay, curandState& randState) const {
        return false;
    }
};

class Lambertian : public Material {
public:
    __device__ Lambertian(const Color& albedo) : albedo(albedo) {}

    __device__ bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay, curandState& randState) const override {
        auto scatteringDirection = hitInfo.getNormal() + Vec3::randomUnitVector(randState);

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
    __device__ Metal(const Color& albedo, const double fuzz) : albedo(albedo), fuzz(Cuda::min(1.0, fuzz)) {}

    __device__ bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay, curandState& randState) const override {
        auto scatteringDirection = Vec3::reflect(rayIn.direction(), hitInfo.getNormal());
        scatteringDirection = scatteringDirection.normalized() + this->fuzz * Vec3::randomUnitVector(randState);
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
        __device__ Dielectric(const double refractiveIndex) : refractiveIndex(refractiveIndex) {}

        __device__ bool scatter(const Ray& rayIn, const HitInfo& hitInfo, Color& attenuation, Ray& scatteredRay, curandState& randState) const override {
            double refractiveIndexReciprocal = hitInfo.getFront() ? (1.0 / this->refractiveIndex) : this->refractiveIndex;

            auto unitRayInDirection = rayIn.direction().normalized();

            double cosTheta = Cuda::min(Vec3::dot(-unitRayInDirection, hitInfo.getNormal()), 1.0);
            double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
            
            Vec3 scatteringDirection;
            if (refractiveIndexReciprocal * sinTheta > 1.0 || reflectance(cosTheta, refractiveIndexReciprocal) > Cuda::random(randState)) {
                scatteringDirection = Vec3::reflect(unitRayInDirection, hitInfo.getNormal());
            }
            else {
                scatteringDirection = Vec3::refract(unitRayInDirection, hitInfo.getNormal(), refractiveIndexReciprocal);
            }

            scatteredRay = Ray(hitInfo.p, scatteringDirection);
            attenuation = Color(1.0);
            return true;
        }

    private:
        double refractiveIndex;

        __device__ static double reflectance(const double cosine, const double refractiveIndexReciprocal) {
            double r0 = (1 - refractiveIndexReciprocal) / (1 + refractiveIndexReciprocal);
            r0 = r0 * r0;
            return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
        }
};