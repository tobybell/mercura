#include "body.h"
#include "eoe.h"
#include "quat.h"
#include "vec3.h"

#define DURATION (365 * 24 * 60 * 60)
#define TIMESTEP (1)
#define N (sizeof (bodies) / sizeof (*bodies))

// Moon orbit radius around Earth: 389499761.0518769
// 411600841.1563506 for 365/12 day period

body_t bodies[] = {
  {"Sol", 1.327124400189e20, {-4.4910405450108775e5}, {-4.4569063372411534e-10, -8.947881775228357e-2}},
  {"Earth", 3.9860044188e14, {1.4952741801817593e11}, {1.4839093307604534e-4, 2.9791618338162836e4}},
  {"Luna", 4.90486959e12, {1.49939018859332e11}, {1.4839093307604534e-4, 3.07756991439366e4}},
};

int main() {
  int n_steps = (flt_t) DURATION / TIMESTEP;
  body_println(&bodies[1]);
  for (int_t i = 0; i < n_steps; i += 1) {
    n_body_reset(bodies, N);
    n_body_gravity(bodies, N);
    n_body_step(bodies, N, TIMESTEP);
    if (i % 1000 == 0) {
      body_print(&bodies[0]);
      printf(",");
      body_print(&bodies[1]);
      printf(",");
      body_println(&bodies[2]);
    }
  }
  return 0;
}
