#include "sim.h"

#include <math.h>

#include "eoe.h"

#define R (6.955874e6)
#define MU (3.9860044188e14)

void sim_rand(sim_t *sim) {
  eoe_t eoe = {
    7e6 + 1e8 * flt_rand(),
    -0.5 + flt_rand(),
    -0.5 + flt_rand(),
    0., 0.,
    2 * M_PI * flt_rand(),
  };
  eoe_to_pv(&eoe, &sim->pos, &sim->vel, MU);
}

void sim_geo(sim_t *sim) {
  flt_t T = 86400;
  flt_t r = cbrt(MU * T * T / (4 * M_PI * M_PI));
  flt_t v = cbrt(2 * M_PI * MU / T);
  flt_t dt = -SIM_TIMESTEP / 2 / T;
  sim->pos = (vec3_t) {r};
  sim->vel = (vec3_t) {v * sin(dt), v * cos(dt)};
}

void sim_leo(sim_t *sim) {
  flt_t r = R + 2e5;
  flt_t v = sqrt(MU / r);
  flt_t T = 2 * M_PI * r * sqrt(r / MU);
  flt_t dt = -SIM_TIMESTEP / 2 / T;
  sim->pos = (vec3_t) {r};
  sim->vel = (vec3_t) {v * sin(dt), v * cos(dt)};
}

flt_t sim_run(sim_t *sim, policy_t *policy) {
  int_t n_steps = SIM_LENGTH / SIM_TIMESTEP;
  flt_t cost = 0;
  for (int_t step_i = 0; step_i < n_steps; step_i += 1) {
    cost += sim_step(sim, policy);
    if (vec3_len(&sim->pos) < R) return 1000000;
  }
  flt_t r_norm = vec3_len(&sim->pos);
  flt_t v_norm = vec3_len(&sim->vel);
  flt_t a = 1 / (2 / r_norm - v_norm * v_norm / MU);
  eoe_t eoe;
  eoe_from_pv(&eoe, &sim->pos, &sim->vel, MU);
  flt_t e = sqrt(eoe.h * eoe.h + eoe.k * eoe.k);
  cost += 10000 * (fabs((a - 2e7)) / 2e7 + e);
  return cost;
}

flt_t sim_step(sim_t *sim, policy_t *policy) {
  vec3_t a = sim->pos;
  flt_t r_sq = vec3_len_sq(&sim->pos);
  vec3_scale(&a, -MU / r_sq);

  // Get acceleration from policy.
  flt_t thrust_acc;
  eoe_t eoe;
  eoe_from_pv(&eoe, &sim->pos, &sim->vel, MU);
  policy_forward(policy, &eoe, &thrust_acc);
  thrust_acc *= SIM_MAX_ACCELERATION;

  // Apply acceleration.
  vec3_t thrust = sim->vel;
  vec3_scale(&thrust, thrust_acc);
  vec3_add(&a, &thrust);

  vec3_mul(&a, SIM_TIMESTEP);
  vec3_add(&sim->vel, &a);

  vec3_t v = sim->vel;
  vec3_mul(&v, SIM_TIMESTEP);
  vec3_add(&sim->pos, &v);

  return fabs(thrust_acc) * SIM_TIMESTEP;
}
