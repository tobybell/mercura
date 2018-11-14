#include "body.h"
#include "eoe.h"
#include "quat.h"
#include "vec3.h"

#define DURATION (365 * 24 * 60 * 60)
#define TIMESTEP (1)
#define N (sizeof (bodies) / sizeof (*bodies))

body_t bodies[] = {
  {"Sol", 1.327124400189e20, {-4.4910405450108775e5}, {-4.4569063372411534e-10, -8.947881775228357e-2}},
  {"Earth", 3.9860044188e14, {1.4952741801817593e11}, {1.4839093307604534e-4, 2.9791618338162836e4}},
};

int main() {
  eoe_t eoe;
  eoe_from_pv(&eoe, &bodies[1].pos, &bodies[1].vel, bodies[0].sgp);
  eoe_println(&eoe);
  return 0;

  int n_steps = (flt_t) DURATION / TIMESTEP;
  body_println(&bodies[1]);
  for (int_t i = 0; i < n_steps; i += 1) {
    n_body_reset(bodies, N);
    n_body_gravity(bodies, N);
    n_body_step(bodies, N, TIMESTEP);
  }
  body_println(&bodies[1]);
  return 0;
}
