#ifndef __VEC3_H
#define __VEC3_H

#include "common.h"

struct vec3 {
  flt_t x, y, z;
};

typedef struct vec3 vec3_t;

void vec3_add(vec3_t *v1, vec3_t *v2);
void vec3_sub(vec3_t *v1, vec3_t *v2);
void vec3_mul(vec3_t *v, flt_t a);
void vec3_div(vec3_t *v, flt_t a);
void vec3_scale(vec3_t *v, flt_t a);
void vec3_cross(vec3_t *v1, vec3_t *v2);

flt_t vec3_dot(vec3_t *v1, vec3_t *v2);
flt_t vec3_len(vec3_t *v);
flt_t vec3_len_sq(vec3_t *v);

void vec3_print(vec3_t *v);
void vec3_println(vec3_t *v);

#endif
