#ifndef __EOE_H
#define __EOE_H

#include "common.h"
#include "vec3.h"

struct eoe {
  flt_t l;
  flt_t h;
  flt_t k;
  flt_t p;
  flt_t q;
  flt_t L;
};

typedef struct eoe eoe_t;

void eoe_from_pv(eoe_t *eoe, vec3_t *r, vec3_t *v, flt_t mu_center);
void eoe_to_pv(eoe_t *eoe, vec3_t *r, vec3_t *v, flt_t mu_center);

void eoe_print(eoe_t *eoe);
void eoe_println(eoe_t *eoe);

#endif
