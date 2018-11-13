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

void body_print(body_t *b);
void body_println(body_t *b);

#endif
