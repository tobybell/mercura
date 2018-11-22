#include "vec3.h"

#include <stdio.h>
#include <math.h>

void vec3_add(vec3_t *v1, vec3_t *v2) {
  v1->x += v2->x;
  v1->y += v2->y;
  v1->z += v2->z;
}

void vec3_sub(vec3_t *v1, vec3_t *v2) {
  v1->x -= v2->x;
  v1->y -= v2->y;
  v1->z -= v2->z;
}

void vec3_mul(vec3_t *v, flt_t a) {
  v->x *= a;
  v->y *= a;
  v->z *= a;
}

void vec3_div(vec3_t *v, flt_t a) {
  v->x /= a;
  v->y /= a;
  v->z /= a;
}

void vec3_scale(vec3_t *v, flt_t a) {
  vec3_mul(v, a / vec3_len(v));
}

void vec3_rot(vec3_t *v, quat_t *q) {
  quat_t v_ = {0, v->x, v->y, v->z};
  quat_t q_ = *q;
  quat_t q_i = *q;
  quat_inv(&q_i);
  quat_ham(&q_, &v_);
  quat_ham(&q_, &q_i);
  *v = (vec3_t) {q_.x, q_.y, q_.z};
}

flt_t vec3_dot(vec3_t *v1, vec3_t *v2) {
  return v1->x * v2->x + v1->y * v2->y + v1->z * v2->z;
}

void vec3_cross(vec3_t *v1, vec3_t *v2) {
  flt_t x = v1->y * v2->z - v1->z * v2->y;
  flt_t y = v1->z * v2->x - v1->x * v2->z;
  flt_t z = v1->x * v2->y - v1->y * v2->x;
  *v1 = (vec3_t) {x, y, z};
}

flt_t vec3_len_sq(vec3_t *v) {
  return vec3_dot(v, v);
}

flt_t vec3_len(vec3_t *v) {
  return sqrt(vec3_len_sq(v));
}

void vec3_print(vec3_t *v) {
  printf("%f,%f,%f", v->x, v->y, v->z);
}

void vec3_println(vec3_t *v) {
  vec3_print(v);
  printf("\n");
}
