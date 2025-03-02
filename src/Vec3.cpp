#include "Vec3.h"

Vec3 operator+(const double scalar, const Vec3& rhs) {
    return Vec3(rhs.x() + scalar, rhs.y() + scalar, rhs.z() + scalar);
}

Vec3 operator+(const Vec3& lhs, const double scalar) {
    return Vec3(lhs.x() + scalar, lhs.y() + scalar, lhs.z() + scalar);
}

Vec3 operator-(const double scalar, const Vec3& rhs) {
    return Vec3(rhs.x() - scalar, rhs.y() - scalar, rhs.z() - scalar);
}

Vec3 operator-(const Vec3& lhs, const double scalar) {
    return Vec3(lhs.x() - scalar, lhs.y() - scalar, lhs.z() - scalar);
}

Vec3 operator*(const double scalar, const Vec3& rhs) {
    return Vec3(rhs.x() * scalar, rhs.y() * scalar, rhs.z() * scalar);
}

Vec3 operator*(const Vec3& lhs, const double scalar) {
    return Vec3(lhs.x() * scalar, lhs.y() * scalar, lhs.z() * scalar);
}

Vec3 operator/(const double scalar, const Vec3& rhs) {
    if (scalar == 0.0)
        return Vec3(0.0);
    return Vec3(rhs.x() / scalar, rhs.y() / scalar, rhs.z() / scalar);
}

Vec3 operator/(const Vec3& lhs, const double scalar) {
    if (scalar == 0.0)
        return Vec3(0.0);
    return Vec3(lhs.x() / scalar, lhs.y() / scalar, lhs.z() / scalar);
}