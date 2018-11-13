#include "body.h"
#include "vec3.h"

#define DURATION (20 * 365 * 24 * 60 * 60)
#define TIMESTEP (1)
#define N (sizeof (bodies) / sizeof (*bodies))

body_t bodies[] = {
  {"Sol", 1.327124400189e20, {-4.4910405450108775e5}, {-4.4569063372411534e-10, -8.947881775228357e-2}},
  {"Earth", 3.9860044188e14, {1.4952741801817593e11}, {1.4839093307604534e-4, 2.9791618338162836e4}},
};

void step(flt_t dt) {
  vec3_t acc[N] = {};
  for (int_t i = 0; i < N; i += 1) {
    for (int_t j = i + 1; j < N; j += 1) {
      vec3_t r = bodies[j].pos;
      vec3_sub(&r, &bodies[i].pos);
      flt_t r_sq = vec3_len_sq(&r);
      vec3_scale(&r, bodies[j].sgp / r_sq);
      vec3_add(&acc[i], &r);
      vec3_scale(&r, -bodies[i].sgp / r_sq);
      vec3_add(&acc[j], &r);
    }
    vec3_mul(&acc[i], dt);
    vec3_add(&bodies[i].vel, &acc[i]);
    vec3_t vel = bodies[i].vel;
    vec3_mul(&vel, dt);
    vec3_add(&bodies[i].pos, &vel);
  }
}

int main() {
  int n_steps = (flt_t) DURATION / TIMESTEP;
  body_println(&bodies[1]);
  for (int_t i = 0; i < n_steps; i += 1) {
    step(TIMESTEP);
  }
  body_println(&bodies[1]);
  return 0;
}
