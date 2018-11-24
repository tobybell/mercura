#include "body.h"
#include "eoe.h"
#include "quat.h"
#include "vec3.h"

#define DURATION (24 * 60 * 60)
#define TIMESTEP (1)
#define N (sizeof (bodies) / sizeof (*bodies))

#define MAX_ACC (1)

body_t bodies[] = {
  {"Earth", 3.9860044188e14},
  {"Cubesat", 1, {42241095.67708342}, {0.017776962751035255, 3071.8591633446}},
};

int main() {
  while (1) {
    flt_t thrust_acc;
    flt_t duration;

    // eoe_t eoe;
    // eoe_from_pv(&eoe, &bodies[1].pos, &bodies[1].vel, bodies[0].sgp);
    // eoe_println(&eoe);
    vec3_print(&bodies[1].pos);
    printf(",");
    vec3_println(&bodies[1].vel);
    fflush(stdout);


    char c;
    scanf(" %c", &c);
    if (c == 'r') {
      scanf(
        " %lf %lf %lf %lf %lf %lf",
        &bodies[1].pos.x,
        &bodies[1].pos.y,
        &bodies[1].pos.z,
        &bodies[1].vel.x,
        &bodies[1].vel.y,
        &bodies[1].vel.z
      );
      printf("\n");
    } else if (c == 'a') {
      scanf(" %lf %lf", &thrust_acc, &duration);

      thrust_acc = min(max(thrust_acc, -MAX_ACC), MAX_ACC);
      int_t n_steps = duration / TIMESTEP;
      for (int_t i = 0; i < n_steps; i += 1) {
        n_body_reset(bodies, N);
        n_body_gravity(bodies, N);

        // Apply thrust acceleration.
        vec3_t thrust = bodies[1].vel;
        vec3_scale(&thrust, thrust_acc);
        vec3_add(&bodies[1].acc, &thrust);

        n_body_step(bodies, N, TIMESTEP);
      }
    } else if (c == 'q') {
      break;
    }
  }
  return 0;
}
