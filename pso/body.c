#include "body.h"

#include <stdio.h>

#include "vec3.h"

void n_body_reset(body_t *bodies, int_t n) {
  for (int_t i = 0; i < n; i += 1) {
    bodies[i].acc = (vec3_t) {};
  }
}

void n_body_gravity(body_t *bodies, int_t n) {
  for (int_t i = 0; i < n; i += 1) {
    for (int_t j = i + 1; j < n; j += 1) {
      vec3_t r = bodies[j].pos;
      vec3_sub(&r, &bodies[i].pos);
      flt_t r_sq = vec3_len_sq(&r);
      vec3_scale(&r, bodies[j].sgp / r_sq);
      vec3_add(&bodies[i].acc, &r);
      vec3_scale(&r, -bodies[i].sgp / r_sq);
      vec3_add(&bodies[j].acc, &r);
    }
  }
}

void n_body_step(body_t *bodies, int_t n, flt_t dt) {
  for (int_t i = 0; i < n; i += 1) {
    vec3_t acc = bodies[i].acc;
    vec3_mul(&acc, dt);
    vec3_add(&bodies[i].vel, &acc);
    vec3_t vel = bodies[i].vel;
    vec3_mul(&vel, dt);
    vec3_add(&bodies[i].pos, &vel);
  }
}

void body_print(body_t *b) {
  printf("%s,%f,", b->name, b->sgp);
  vec3_print(&b->pos);
  printf(",");
  vec3_print(&b->vel);
}

void body_println(body_t *b) {
  body_print(b);
  printf("\n");
}
