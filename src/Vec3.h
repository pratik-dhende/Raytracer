#pragma once

#include "Utility.h"

#include <cmath>

class Vec3;

Vec3 operator+(const double scalar, const Vec3& rhs);
Vec3 operator+(const Vec3& lhs, const double scalar);
Vec3 operator-(const double scalar, const Vec3& rhs);
Vec3 operator-(const Vec3& lhs, const double scalar);
Vec3 operator*(const double scalar, const Vec3& rhs);
Vec3 operator*(const Vec3& lhs, const double scalar);
Vec3 operator/(const double scalar, const Vec3& rhs);
Vec3 operator/(const Vec3& lhs, const double scalar);

Vec3 operator*(const int scalar, const Vec3& rhs);
Vec3 operator*(const Vec3& lhs, const int scalar);

class Vec3 {
private:
    double m_x;
    double m_y;
    double m_z;

public:
    constexpr Vec3(double x, double y, double z) : m_x(x), m_y(y), m_z(z) {

    }

    constexpr Vec3(double scalar) : Vec3(scalar, scalar, scalar) {

    }

    constexpr Vec3() : Vec3(0.0) {

    }

    void normalize() {
        double mag = magnitude();
        if (mag == 0.0) {
            return;
        }

        m_x /= mag;
        m_y /= mag;
        m_z /= mag;
    }

    Vec3 normalized() const {
        double mag = magnitude();
        if (mag == 0.0) {
            return Vec3(0.0);
        }
        return Vec3(m_x / mag, m_y / mag, m_z / mag);
    }

    double x() const {
        return m_x;
    }

    double y() const {
        return m_y;
    }

    double z() const {
        return m_z;
    }

    double r() const {
        return m_x;
    }

    double g() const {
        return m_y;
    }

    double b() const {
        return m_z;
    }

    double operator[](int index) const {
        return (index == 0) ? m_x : (index == 1 ? m_y : m_z);
    }

    double magnitude() const {
        return std::sqrt(m_x * m_x + m_y * m_y + m_z * m_z);;
    }

    double magnitudeSquared() const {
        return m_x * m_x + m_y * m_y + m_z * m_z;
    }

    Vec3 operator-() const {
        return Vec3(-m_x, -m_y, -m_z);
    }

    Vec3 operator+(const Vec3& other) const {
        return Vec3(m_x + other.x(), m_y + other.y(), m_z + other.z());
    }

    Vec3 operator-(const Vec3& other) const {
        return Vec3(m_x - other.x(), m_y - other.y(), m_z - other.z());
    }

    Vec3 operator*(const Vec3& other) const {
        return Vec3(m_x * other.x(), m_y * other.y(), m_z * other.z());
    }

    Vec3& operator+=(const Vec3& other) {
        m_x += other.x();
        m_y += other.y();
        m_z += other.z();
        return *this;
    }

    Vec3& operator*=(const double scalar) {
        m_x *= scalar;
        m_y *= scalar;
        m_z *= scalar;
        return *this;
    }

    Vec3& operator*=(const int scalar) {
        m_x *= scalar;
        m_y *= scalar;
        m_z *= scalar;
        return *this;
    }
    
    bool nearZero() const {
        auto eps = 1e-8;
        return (std::abs(m_x) < eps) && (std::abs(m_y) < eps) && (std::abs(m_z) < eps);
    }

    static Vec3 randomDouble() {
        return Vec3(::randomDouble(), ::randomDouble(), ::randomDouble());
    }

    static Vec3 randomDouble(const double min, const double max) {
        return Vec3(::randomDouble(min, max), ::randomDouble(min, max), ::randomDouble(min, max));
    }

    static Vec3 randomUnitVector() {
        while(true) {
            auto p = Vec3::randomDouble(-1.0, 1.0);
            double magnitudeSquared = p.magnitudeSquared();
            if (1.0e-160 < magnitudeSquared && magnitudeSquared <= 1.0) {
                return p.normalized();
            }
        }
    }
    
    static Vec3 randomVectorInUnitCircle() {
        while(true) {
            auto v = Vec3(::randomDouble(-1.0, 1.0), ::randomDouble(-1.0, 1.0), 0.0);
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


    static Vec3 reflect(const Vec3& v, const Vec3& normal) {
        return v - 2.0 * dot(v, normal) * normal;
    }

    static Vec3 refract(const Vec3& v, const Vec3& normal, const double refractiveIndexReciprocal) {
        auto cosTheta = std::min(dot(-v, normal), 1.0);
        const Vec3 vOutPerpendicular = refractiveIndexReciprocal * (v + cosTheta * normal);
        // TODO: Why use abs?
        const Vec3 vOutParallel = -std::sqrt(std::abs(1.0 - vOutPerpendicular.magnitudeSquared())) * normal;
        return vOutPerpendicular + vOutParallel;
    }

    static double dot(const Vec3& v1, const Vec3& v2) {
        return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
    }

    static Vec3 cross(const Vec3& v1, const Vec3& v2) {
        return Vec3(v1.y() * v2.z() - v2.y() * v1.z(), v2.x() * v1.z() - v1.x() * v2.z(), v1.x() * v2.y() - v2.x() * v1.y());
    }

    static Vec3 floor(const Vec3& v) {
        return Vec3(std::floor(v.m_x), std::floor(v.m_y), std::floor(v.m_z));
    }
};

using Point3 = Vec3;