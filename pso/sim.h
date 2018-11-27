#ifndef __SIM_H
#define __SIM_H

#include "common.h"
#include "vec3.h"
#include "policy.h"

#define SIM_LENGTH (200000)
#define SIM_TIMESTEP (2)
#define SIM_MAX_ACCELERATION (0.1)

struct sim {
  vec3_t pos;
  vec3_t vel;
};

typedef struct sim sim_t;

void sim_rand(sim_t *sim);
void sim_geo(sim_t *sim);
void sim_leo(sim_t *sim);
flt_t sim_run(sim_t *sim, policy_t *policy);
flt_t sim_step(sim_t *sim, policy_t *policy);

#endif
