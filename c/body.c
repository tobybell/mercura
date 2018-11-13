#include "body.h"

#include <stdio.h>

#include "vec3.h"

void body_print(body_t *b) {
  printf("%s %f ", b->name, b->sgp);
  vec3_print(&b->pos);
  printf(" ");
  vec3_print(&b->vel);
}

void body_println(body_t *b) {
  body_print(b);
  printf("\n");
}
