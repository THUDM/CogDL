#include "rwalk.h"
#include <omp.h>
#include <stdlib.h>


void random_walk(int const* starts, int const* ptr, int const* neighs, int n, int num_walks,
                 int num_steps, int seed, int nthread, float restart_prop, int* walks) {
  if (nthread > 0) {
    omp_set_num_threads(nthread);
  }
#pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
    unsigned int private_seed = (unsigned int)(seed + thread_num);
#pragma omp for
    for (int i = 0; i < n; i++) {
      int offset, num_neighs;
      for (int walk = 0; walk < num_walks; walk++) {
        // int curr = i;
        int curr = starts[i];
        offset = i * num_walks * (num_steps + 1) + walk * (num_steps + 1);
        walks[offset] = starts[i];
        for (int step = 0; step < num_steps; step++) {
          num_neighs = ptr[curr + 1] - ptr[curr];

          if((restart_prop > 0) && (rand_r(&private_seed) / (double)RAND_MAX < restart_prop)){
              curr = starts[i];
          } else {
            if (num_neighs > 0) {
              curr = neighs[ptr[curr] + (rand_r(&private_seed) % num_neighs)];
            }
          }
          walks[offset + step + 1] = curr;
        }

      }
    }
  }
}
