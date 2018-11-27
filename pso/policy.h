#ifndef __POLICY_H
#define __POLICY_H

#include "common.h"
#include "eoe.h"

#define POLICY_N_HIDDEN (6)
#define POLICY_N_OUTPUT (1)

struct policy {
  flt_t params[(6 + 1) * POLICY_N_HIDDEN + (POLICY_N_HIDDEN + 1) * POLICY_N_OUTPUT];
};

typedef struct policy policy_t;

void policy_rand(policy_t *p);

void policy_add(policy_t *p1, policy_t *p2, policy_t *out);
void policy_sub(policy_t *p1, policy_t *p2, policy_t *out);
void policy_mul(policy_t *p, flt_t a);
void policy_forward(policy_t *policy, eoe_t *eoe, flt_t output[POLICY_N_OUTPUT]);

void policy_print(policy_t *policy);
void policy_println(policy_t *policy);

#endif
