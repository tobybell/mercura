#include "common.h"

#include <stdlib.h>

flt_t flt_rand(void) {
  return (flt_t) rand() / RAND_MAX;
}
