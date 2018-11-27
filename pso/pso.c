#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "policy.h"
#include "sim.h"

#define N_PARTICLES (20)
#define N_GENERATIONS (1000)
#define N_SIMS (1)

#define W (0.9)
#define C1 (1.5)
#define C2 (1.5)

struct particle {
  policy_t curr;
  policy_t vel;
  policy_t best;
  flt_t best_val;
};

typedef struct particle particle_t;

void particle_rand(particle_t *particle);
void particle_eval(particle_t *particle, sim_t sims[N_SIMS]);
void particle_move(particle_t *particle);

particle_t particles[N_PARTICLES];
particle_t *champion = particles;

int_t main() {
  srand(time(NULL));

  for (int_t p_i = 0; p_i < N_PARTICLES; p_i += 1) {
    particle_rand(&particles[p_i]);
  }
  for (int_t gen_i = 0; gen_i < N_GENERATIONS; gen_i += 1) {
    sim_t sims[N_SIMS];
    sim_leo(&sims[0]);
    // for (int_t sim_i = 0; sim_i < N_SIMS; sim_i += 1) {
    //   sim_geo(&sims[sim_i]);
    // }
    for (int_t p_i = 0; p_i < N_PARTICLES; p_i += 1) {
      sim_t sims_cp[N_SIMS];
      memcpy(sims_cp, sims, sizeof (sims));
      particle_eval(&particles[p_i], sims_cp);
    }
    for (int_t p_i = 0; p_i < N_PARTICLES; p_i += 1) {
      if (&particles[p_i] != champion && flt_rand() < 0.01) {
        particle_rand(&particles[p_i]);
      } else {
        particle_move(&particles[p_i]);
      }
    }
    printf("gen %d, champ cost %f\n", gen_i, champion->best_val);
    policy_println(&champion->best);
  }
  return 0;
}

void particle_rand(particle_t *particle) {
  policy_rand(&particle->curr);
  policy_rand(&particle->vel);
  policy_mul(&particle->vel, 0.2);
  particle->best_val = 1e16;
}

void particle_eval(particle_t *particle, sim_t sims[N_SIMS]) {
  flt_t val = 0;
  flt_t mean = 0;
  for (int_t sim_i = 0; sim_i < N_SIMS; sim_i += 1) {
    flt_t score = sim_run(&sims[sim_i], &particle->curr);
    mean += score;
    val = max(val, score);
  }
  mean /= N_SIMS;
  val += mean;

  if (val < particle->best_val) {
    // printf("particle %d surpassed itself, new cost %f\n", (int) (particle - particles), val);
    particle->best = particle->curr;
    particle->best_val = val;
  }
  if (val < champion->best_val) {
    champion = particle;
  }
}

void particle_move(particle_t *particle) {
  policy_t curr_vel = particle->vel;
  policy_t self_delta, champ_delta;
  policy_sub(&particle->best, &particle->curr, &self_delta);
  policy_sub(&champion->best, &particle->curr, &champ_delta);
  policy_mul(&particle->vel, W);
  policy_mul(&self_delta, flt_rand() * C1);
  policy_mul(&champ_delta, flt_rand() * C2);
  policy_add(&particle->vel, &self_delta, &particle->vel);
  policy_add(&particle->vel, &champ_delta, &particle->vel);
  policy_add(&particle->curr, &particle->vel, &particle->curr);
}
