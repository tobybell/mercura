#ifndef __QUAT_H
#define __QUAT_H

#include "common.h"

struct quat {
  flt_t w, x, y, z;
};

typedef struct quat quat_t;

#include "vec3.h"

void quat_ham(quat_t *q1, quat_t *q2);
void quat_inv(quat_t *q);
void quat_set_axisang(quat_t *q, vec3_t *x, flt_t a);

flt_t quat_ang(quat_t *q);
void quat_axis(quat_t *q, vec3_t *x);

void quat_print(quat_t *q);
void quat_println(quat_t *q);

#endif
