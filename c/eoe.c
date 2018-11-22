#include "eoe.h"

#include <stdio.h>
#include <math.h>

#include "common.h"
#include "vec3.h"

void eoe_from_pv(eoe_t *eoe, vec3_t *r, vec3_t *v, flt_t mu_center) {
  flt_t r_norm = vec3_len(r);
  flt_t v_norm = vec3_len(v);

  // Compute the specific angular momentum vector.
  vec3_t hbar = *r; vec3_cross(&hbar, v);

  // Compute the perpendicular equinoctial basis vector by normalizing the
  // specific angular momentum.
  vec3_t w = hbar; vec3_scale(&w, 1);

  // Compute the ascending node vector components in the equinoctial frame.
  flt_t p = w.x / (1 + w.z);
  flt_t q = -w.y / (1 + w.z);

  // Compute the two planar equinoctial frame basis vectors.
  flt_t C = 1 + p * p + q * q;
  vec3_t f = {(1 - p * p + q * q) / C, 2 * p * q / C, -2 * p / C};
  vec3_t g = {2 * p * q / C, (1 + p * p - q * q) / C, 2 * q / C};

  // Compute the eccentricity vector in the ECI frame.
  vec3_t r_unit = *r; vec3_scale(&r_unit, 1);
  vec3_t e = *v;
  vec3_cross(&e, &hbar);
  vec3_div(&e, mu_center);
  vec3_sub(&e, &r_unit);

  // Project the eccentricity vector into the equinoctial frame.
  flt_t h = vec3_dot(&e, &g);
  flt_t k = vec3_dot(&e, &f);

  // Project the position into the equinoctial frame and compute the true
  // longitude.
  flt_t X = vec3_dot(r, &f);
  flt_t Y = vec3_dot(r, &g);
  flt_t L = fmod(atan2(Y, X), 2 * M_PI);

  // Compute the semi-latus rectum.
  flt_t a = 1 / (2 / r_norm - v_norm * v_norm / mu_center);
  flt_t slr = a * (1 - vec3_len_sq(&e));

  *eoe = (eoe_t) {slr, h, k, p, q, L};
}

void eoe_print(eoe_t *eoe) {
  printf("{%f %f %f %f %f %f}", eoe->l, eoe->h, eoe->k, eoe->p, eoe->q, eoe->L);
}

void eoe_println(eoe_t *eoe) {
  eoe_print(eoe);
  printf("\n");
}