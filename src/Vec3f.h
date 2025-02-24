#pragma once

#include <cmath>

class Vec3f {
private:
    float m_x;
    float m_y;
    float m_z;

public:
    constexpr Vec3f(float x, float y, float z) : m_x(x), m_y(y), m_z(z) {

    }

    constexpr Vec3f(float scalar) : Vec3f(scalar, scalar, scalar) {

    }

    constexpr Vec3f() : Vec3f(0.0) {

    }

    void normalize() {
        float mag = magnitude();
        if (mag == 0.0) {
            return;
        }

        m_x /= mag;
        m_y /= mag;
        m_z /= mag;
    }

    Vec3f normalized() const {
        float mag = magnitude();
        if (mag == 0.0) {
            return Vec3f(0.0);
        }
        return Vec3f(m_x / mag, m_y / mag, m_z / mag);
    }

    float x() const {
        return m_x;
    }

    float y() const {
        return m_y;
    }

    float z() const {
        return m_z;
    }

    float r() const {
        return m_x;
    }

    float g() const {
        return m_y;
    }

    float b() const {
        return m_z;
    }

    float magnitude() const {
        return std::sqrt(m_x * m_x + m_y * m_y + m_z * m_z);;
    }

    float magnitudeSquared() const {
        return m_x * m_x + m_y * m_y + m_z * m_z;
    }

    Vec3f operator-() const {
        return Vec3f(-m_x, -m_y, -m_z);
    }

    Vec3f operator+(const Vec3f& other) const {
        return Vec3f(m_x + other.x(), m_y + other.y(), m_z + other.z());
    }

    Vec3f operator-(const Vec3f& other) const {
        return Vec3f(m_x - other.x(), m_y - other.y(), m_z - other.z());
    }

    Vec3f& operator+=(const Vec3f& other) {
        m_x += other.x();
        m_y += other.y();
        m_z += other.z();
        return *this;
    }

    Vec3f& operator*=(const int scalar) {
        m_x *= scalar;
        m_y *= scalar;
        m_z *= scalar;
        return *this;
    }
};

float dot(const Vec3f& v1, const Vec3f& v2);

Vec3f operator+(const float scalar, const Vec3f& rhs);
Vec3f operator+(const Vec3f& lhs, const float scalar);
Vec3f operator-(const float scalar, const Vec3f& rhs);
Vec3f operator-(const Vec3f& lhs, const float scalar);
Vec3f operator*(const float scalar, const Vec3f& rhs);
Vec3f operator*(const Vec3f& lhs, const float scalar);
Vec3f operator/(const float scalar, const Vec3f& rhs);
Vec3f operator/(const Vec3f& lhs, const float scalar);

Vec3f operator*(const int scalar, const Vec3f& rhs);
Vec3f operator*(const Vec3f& lhs, const int scalar);

using Point3f = Vec3f;