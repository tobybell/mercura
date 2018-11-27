#include "body.h"
#include "eoe.h"
#include "quat.h"
#include "vec3.h"

#define DURATION (100000)
#define TIMESTEP (1)
#define N (sizeof (bodies) / sizeof (*bodies))

// Moon orbit radius around Earth: 389499761.0518769
// 411600841.1563506 for 365/12 day period

// body_t bodies[] = {
//   {"Sol", 1.327124400189e20, {-4.4910405450108775e5}, {-4.4569063372411534e-10, -8.947881775228357e-2}},
//   {"Earth", 3.9860044188e14, {1.4952741801817593e11}, {1.4839093307604534e-4, 2.9791618338162836e4}},
//   {"Luna", 4.90486959e12, {1.49939018859332e11}, {1.4839093307604534e-4, 3.07756991439366e4}},
// };

body_t bodies[] = {
  {"Earth", 3.9860044188e14},
  {"Satellite", 1, {42241095.67708342}, {0.017776962751035255, 3071.8591633446}},
};

flt_t policy(int_t t) {
  if (t < 5900) return -1;
  if (t > 26700 && t < 34400) return -1;
  return 0;
}

int_t main() {
  int n_steps = (flt_t) DURATION / TIMESTEP;
  printf("[");
  for (int_t i = 0; i < n_steps; i += 1) {
    n_body_reset(bodies, N);
    n_body_gravity(bodies, N);

    // Consult policy.
    flt_t thrust_acc = 0.1 * policy(i);

    // Apply thrust acceleration.
    vec3_t thrust = bodies[1].vel;
    vec3_scale(&thrust, thrust_acc);
    vec3_add(&bodies[1].acc, &thrust);

    if (i % 60 == 0) {
      if (i > 0) {
        printf(",\n");
      }
      printf("[");
      vec3_print(&bodies[1].pos);
      printf(",%d,%f]", i, thrust_acc);
    }

    n_body_step(bodies, N, TIMESTEP);
  }
  printf("]");
  return 0;
}
