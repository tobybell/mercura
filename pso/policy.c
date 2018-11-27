#include "policy.h"

#include <stdio.h>
#include <math.h>

#include "eoe.h"

#define N_INPUT (6)
#define N_HIDDEN (POLICY_N_HIDDEN)
#define N_OUTPUT (POLICY_N_OUTPUT)
#define N_PARAMS ((N_INPUT + 1) * N_HIDDEN + (N_HIDDEN + 1) * N_OUTPUT)

#define HW_OFFSET (0)
#define HB_OFFSET (HW_OFFSET + N_INPUT * N_HIDDEN)
#define OW_OFFSET (HB_OFFSET + N_HIDDEN)
#define OB_OFFSET (OW_OFFSET + N_HIDDEN * N_OUTPUT)

void policy_rand(policy_t *p) {
  for (int_t i = 0; i < N_PARAMS; i += 1) {
    p->params[i] = -0.5 + 1 * flt_rand();
  }
}

void policy_forward(policy_t *p, eoe_t *eoe, flt_t output[N_OUTPUT]) {
  flt_t hidden[N_HIDDEN];

  flt_t *hidden_weights = &p->params[HW_OFFSET];
  flt_t *hidden_biases = &p->params[HB_OFFSET];
  flt_t *output_weights = &p->params[OW_OFFSET];
  flt_t *output_biases = &p->params[OB_OFFSET];

  // Input to hidden.
  for (int_t i = 0; i < N_HIDDEN; i += 1) {
    hidden[i] = hidden_biases[i];
    hidden[i] += hidden_weights[N_INPUT * i + 0] * eoe->l / 1e7;
    hidden[i] += hidden_weights[N_INPUT * i + 1] * eoe->h;
    hidden[i] += hidden_weights[N_INPUT * i + 2] * eoe->k;
    hidden[i] += hidden_weights[N_INPUT * i + 3] * eoe->p;
    hidden[i] += hidden_weights[N_INPUT * i + 4] * eoe->q;
    hidden[i] += hidden_weights[N_INPUT * i + 5] * eoe->L;
    hidden[i] = tanh(hidden[i]);
  }
  
  // Hidden to output.
  for (int_t i = 0; i < N_OUTPUT; i += 1) {
    output[i] = output_biases[i];
    for (int_t j = 0; j < N_HIDDEN; j += 1) {
      output[i] += output_weights[N_HIDDEN * i + j] * hidden[j];
    }
    output[i] = tanh(output[i]);
  }
}

void policy_add(policy_t *p1, policy_t *p2, policy_t *out) {
  for (int_t i = 0; i < N_PARAMS; i += 1) {
    out->params[i] = p1->params[i] + p2->params[i];
  }
}

void policy_sub(policy_t *p1, policy_t *p2, policy_t *out) {
  for (int_t i = 0; i < N_PARAMS; i += 1) {
    out->params[i] = p1->params[i] - p2->params[i];
  }
}

void policy_mul(policy_t *p, flt_t a) {
  for (int_t i = 0; i < N_PARAMS; i += 1) {
    p->params[i] *= a;
  }
}


void policy_print(policy_t *policy) {
  printf("%f", policy->params[0]);
  for (int_t i = 1; i < N_PARAMS; i += 1) {
    printf(",%f", policy->params[i]);
  }
}

void policy_println(policy_t *policy) {
  policy_print(policy);
  printf("\n");
}
