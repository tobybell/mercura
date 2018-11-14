#include "quat.h"

#include <stdio.h>
#include <math.h>

void quat_ham(quat_t *q1, quat_t *q2) {
  flt_t w = q1->w * q2->w - q1->x * q2->x - q1->y * q2->y - q1->z * q2->z;
  flt_t x = q1->w * q2->x + q1->x * q2->w + q1->y * q2->z - q1->z * q2->y;
  flt_t y = q1->w * q2->y - q1->x * q2->z + q1->y * q2->w + q1->z * q2->x;
  flt_t z = q1->w * q2->z + q1->x * q2->y - q1->y * q2->x + q1->z * q2->w;
  *q1 = (quat_t) {w, x, y, z};
}

void quat_inv(quat_t *q) {
  q->x *= -1;
  q->y *= -1;
  q->z *= -1;
}

void quat_set_axisang(quat_t *q, vec3_t *x, flt_t a) {
  flt_t sin_ = sin(a / 2);
  *q = (quat_t) {cos(a / 2), x->x * sin_, x->y * sin_, x->z * sin_};
}

/**
 * Scale the rotation of a quaternion to have a particular angle `a`.
 *
 * Does not change the axis of rotation.
 */
void quat_scale(quat_t *q, flt_t a) {
  flt_t s = sin(a / 2) / sqrt(1 - q->w * q->w);
  *q = (quat_t) {cos(a / 2), q->x * s, q->y * s, q->z * s};
}

flt_t quat_ang(quat_t *q) {
  return 2 * atan2(sqrt(q->x * q->x + q->y * q->y + q->z * q->z), q->w);
}

void quat_axis(quat_t *q, vec3_t *x) {
  flt_t s = sqrt(1 - q->w * q->w);
  *x = (vec3_t) {q->x / s, q->y / s, q->z / s};
}

void quat_print(quat_t *q) {
  printf("{%f %f %f %f}", q->w, q->x, q->y, q->z);
}

void quat_println(quat_t *q) {
  quat_print(q);
  printf("\n");
}
