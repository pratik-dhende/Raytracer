#include "Vec3f.h"

float dot(const Vec3f& v1, const Vec3f& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}


Vec3f operator+(const float scalar, const Vec3f& rhs) {
    return Vec3f(rhs.x() + scalar, rhs.y() + scalar, rhs.z() + scalar);
}

Vec3f operator+(const Vec3f& lhs, const float scalar) {
    return Vec3f(lhs.x() + scalar, lhs.y() + scalar, lhs.z() + scalar);
}

Vec3f operator-(const float scalar, const Vec3f& rhs) {
    return Vec3f(rhs.x() - scalar, rhs.y() - scalar, rhs.z() - scalar);
}

Vec3f operator-(const Vec3f& lhs, const float scalar) {
    return Vec3f(lhs.x() - scalar, lhs.y() - scalar, lhs.z() - scalar);
}

Vec3f operator*(const float scalar, const Vec3f& rhs) {
    return Vec3f(rhs.x() * scalar, rhs.y() * scalar, rhs.z() * scalar);
}

Vec3f operator*(const Vec3f& lhs, const float scalar) {
    return Vec3f(lhs.x() * scalar, lhs.y() * scalar, lhs.z() * scalar);
}

Vec3f operator/(const float scalar, const Vec3f& rhs) {
    if (scalar == 0.0f)
        return Vec3f(0.0f);
    return Vec3f(rhs.x() / scalar, rhs.y() / scalar, rhs.z() / scalar);
}

Vec3f operator/(const Vec3f& lhs, const float scalar) {
    if (scalar == 0.0f)
        return Vec3f(0.0f);
    return Vec3f(lhs.x() / scalar, lhs.y() / scalar, lhs.z() / scalar);
}