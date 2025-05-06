#pragma once

#include "Cuda.h"
#include "Utility.h"

#include <cmath>
#include <curand_kernel.h>

class Vec3 {
private:
    double m_x;
    double m_y;
    double m_z;

public:
    __host__ __device__ Vec3(double x, double y, double z) : m_x(x), m_y(y), m_z(z) {

    }

    __host__ __device__ Vec3(double scalar) : Vec3(scalar, scalar, scalar) {

    }

    __host__ __device__ Vec3() : Vec3(0.0) {

    }

    __host__ __device__ void normalize() {
        double mag = magnitude();
        if (mag == 0.0) {
            return;
        }

        m_x /= mag;
        m_y /= mag;
        m_z /= mag;
    }

    __host__ __device__ Vec3 normalized() const {
        double mag = magnitude();
        if (mag == 0.0) {
            return Vec3(0.0);
        }
        return Vec3(m_x / mag, m_y / mag, m_z / mag);
    }

    __host__ __device__ double x() const {
        return m_x;
    }

    __host__ __device__ double y() const {
        return m_y;
    }

    __host__ __device__ double z() const {
        return m_z;
    }

    __host__ __device__ double r() const {
        return m_x;
    }

    __host__ __device__ double g() const {
        return m_y;
    }

    __host__ __device__ double b() const {
        return m_z;
    }

    __host__ __device__ double magnitude() const {
        return sqrt(m_x * m_x + m_y * m_y + m_z * m_z);
    }

    __host__ __device__ double magnitudeSquared() const {
        return m_x * m_x + m_y * m_y + m_z * m_z;
    }

    __host__ __device__ Vec3 operator-() const {
        return Vec3(-m_x, -m_y, -m_z);
    }

    __host__ __device__ Vec3 operator+(const Vec3& other) const {
        return Vec3(m_x + other.x(), m_y + other.y(), m_z + other.z());
    }

    __host__ __device__ Vec3 operator-(const Vec3& other) const {
        return Vec3(m_x - other.x(), m_y - other.y(), m_z - other.z());
    }

    __host__ __device__ Vec3 operator*(const Vec3& other) const {
        return Vec3(m_x * other.x(), m_y * other.y(), m_z * other.z());
    }

    __host__ __device__ Vec3& operator+=(const Vec3& other) {
        m_x += other.x();
        m_y += other.y();
        m_z += other.z();
        return *this;
    }

    template<typename T>
    __host__ __device__ Vec3& operator*=(const T scalar) {
        m_x *= scalar;
        m_y *= scalar;
        m_z *= scalar;
        return *this;
    }
    
    __host__ __device__ bool nearZero() const {
        auto eps = 1e-8;
        return (abs(m_x) < eps) && (abs(m_y) < eps) && (abs(m_z) < eps);
    }

    static Vec3 random() {
        return Vec3(::random(), ::random(), ::random());
    }

    static Vec3 random(const double min, const double max) {
        return Vec3(::random(min, max), ::random(min, max), ::random(min, max));
    }

    static Vec3 randomUnitVector() {
        while(true) {
            auto p = Vec3::random(-1.0, 1.0);
            double magnitudeSquared = p.magnitudeSquared();
            if (1.0e-160 < magnitudeSquared && magnitudeSquared <= 1.0) {
                return p.normalized();
            }
        }
    }
    
    static Vec3 randomVectorInUnitCircle() {
        while(true) {
            auto v = Vec3(::random(-1.0, 1.0), ::random(-1.0, 1.0), 0.0);
            if (v.magnitude() < 1.0) {
                return v;
            }
        }
    }

    static Vec3 randomUnitVectorInHemisphere(const Vec3& normal) {
        Vec3 unitSphereVector = randomUnitVector();
        if (Vec3::dot(unitSphereVector, normal) > 0.0) {
            return unitSphereVector;
        }
        else {
            return -unitSphereVector;
        }
    }

    __device__ static Vec3 random(curandState& randState) {
        return Vec3(Cuda::random(randState), Cuda::random(randState), Cuda::random(randState));
    }

    __device__ static Vec3 random(const double min, const double max, curandState& randState) {
        return Vec3(Cuda::random(min, max, randState), Cuda::random(min, max, randState), Cuda::random(min, max, randState));
    }

    __device__ static Vec3 randomUnitVector(curandState& randState) {
        while(true) {
            auto p = Vec3::random(-1.0, 1.0, randState);
            double magnitudeSquared = p.magnitudeSquared();
            if (1.0e-160 < magnitudeSquared && magnitudeSquared <= 1.0) {
                return p.normalized();
            }
        }
    }
    
    __device__ static Vec3 randomVectorInUnitCircle(curandState& randState) {
        while(true) {
            auto v = Vec3(Cuda::random(-1.0, 1.0, randState), Cuda::random(-1.0, 1.0, randState), 0.0);
            if (v.magnitude() < 1.0) {
                return v;
            }
        }
    }

    __device__ static Vec3 randomUnitVectorInHemisphere(const Vec3& normal, curandState& randState) {
        Vec3 unitSphereVector = randomUnitVector(randState);
        if (Vec3::dot(unitSphereVector, normal) > 0.0) {
            return unitSphereVector;
        }
        else {
            return -unitSphereVector;
        }
    }

    __host__ __device__ static Vec3 reflect(const Vec3& v, const Vec3& normal);
    __host__ __device__ static Vec3 refract(const Vec3& v, const Vec3& normal, const double refractiveIndexReciprocal); 
    __host__ __device__ static double dot(const Vec3& v1, const Vec3& v2); 
    __host__ __device__ static Vec3 cross(const Vec3& v1, const Vec3& v2);
};

__host__ __device__ static Vec3 operator+(const double scalar, const Vec3& rhs) {
    return Vec3(rhs.x() + scalar, rhs.y() + scalar, rhs.z() + scalar);
}

__host__ __device__ static Vec3 operator+(const Vec3& lhs, const double scalar) {
    return Vec3(lhs.x() + scalar, lhs.y() + scalar, lhs.z() + scalar);
}

__host__ __device__ static Vec3 operator-(const double scalar, const Vec3& rhs) {
    return Vec3(rhs.x() - scalar, rhs.y() - scalar, rhs.z() - scalar);
}

__host__ __device__ static Vec3 operator-(const Vec3& lhs, const double scalar) {
    return Vec3(lhs.x() - scalar, lhs.y() - scalar, lhs.z() - scalar);
}

__host__ __device__ static Vec3 operator*(const double scalar, const Vec3& rhs) {
    return Vec3(rhs.x() * scalar, rhs.y() * scalar, rhs.z() * scalar);
}

__host__ __device__ static Vec3 operator*(const Vec3& lhs, const double scalar) {
    return Vec3(lhs.x() * scalar, lhs.y() * scalar, lhs.z() * scalar);
}

__host__ __device__ static Vec3 operator/(const double scalar, const Vec3& rhs) {
    if (scalar == 0.0)
        return Vec3(0.0);
    return Vec3(rhs.x() / scalar, rhs.y() / scalar, rhs.z() / scalar);
}

__host__ __device__ static Vec3 operator/(const Vec3& lhs, const double scalar) {
    if (scalar == 0.0)
        return Vec3(0.0);
    return Vec3(lhs.x() / scalar, lhs.y() / scalar, lhs.z() / scalar);
}


__host__ __device__ Vec3 Vec3::reflect(const Vec3& v, const Vec3& normal) {
    return v - 2.0 * dot(v, normal) * normal;
}

__host__ __device__ Vec3 Vec3::refract(const Vec3& v, const Vec3& normal, const double refractiveIndexReciprocal) {
    auto cosTheta = Cuda::min(dot(-v, normal), 1.0);
    const Vec3 vOutPerpendicular = refractiveIndexReciprocal * (v + cosTheta * normal);
    // TODO: Why use abs?
    const Vec3 vOutParallel = -sqrt(abs(1.0 - vOutPerpendicular.magnitudeSquared())) * normal;
    return vOutPerpendicular + vOutParallel;
}

__host__ __device__ double Vec3::dot(const Vec3& v1, const Vec3& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

__host__ __device__ Vec3 Vec3::cross(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.y() * v2.z() - v2.y() * v1.z(), v2.x() * v1.z() - v1.x() * v2.z(), v1.x() * v2.y() - v2.x() * v1.y());
}

using Point3 = Vec3;