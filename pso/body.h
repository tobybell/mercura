#ifndef __BODY_H
#define __BODY_H

#include <stdio.h>

#include "common.h"
#include "vec3.h"

typedef struct body body_t;

struct body {
  str_t name;
  flt_t sgp;
  vec3_t pos, vel, acc;
};

void n_body_reset(body_t *bodies, int_t n);
void n_body_gravity(body_t *bodies, int_t n);
void n_body_step(body_t *bodies, int_t n, flt_t dt);

void body_print(body_t *b);
void body_println(body_t *b);

#endif
