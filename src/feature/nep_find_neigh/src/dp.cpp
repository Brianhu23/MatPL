
#include "dp.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace
{
double get_area_one_direction(const double* a, const double* b)
{
  double s1 = a[1] * b[2] - a[2] * b[1];
  double s2 = a[2] * b[0] - a[0] * b[2];
  double s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

double get_area(const int d, const double* cpu_h)
{
  double area;
  double a[3] = {cpu_h[0], cpu_h[3], cpu_h[6]};
  double b[3] = {cpu_h[1], cpu_h[4], cpu_h[7]};
  double c[3] = {cpu_h[2], cpu_h[5], cpu_h[8]};
  if (d == 0) {
    area = get_area_one_direction(b, c);
  } else if (d == 1) {
    area = get_area_one_direction(c, a);
  } else {
    area = get_area_one_direction(a, b);
  }
  return area;
}

double get_det(const double* cpu_h)
{
  return cpu_h[0] * (cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7]) +
         cpu_h[1] * (cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8]) +
         cpu_h[2] * (cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6]);
}

double get_volume(const double* cpu_h) { return abs(get_det(cpu_h)); }

void get_inverse(double* cpu_h)
{
  cpu_h[9] = cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7];
  cpu_h[10] = cpu_h[2] * cpu_h[7] - cpu_h[1] * cpu_h[8];
  cpu_h[11] = cpu_h[1] * cpu_h[5] - cpu_h[2] * cpu_h[4];
  cpu_h[12] = cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8];
  cpu_h[13] = cpu_h[0] * cpu_h[8] - cpu_h[2] * cpu_h[6];
  cpu_h[14] = cpu_h[2] * cpu_h[3] - cpu_h[0] * cpu_h[5];
  cpu_h[15] = cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6];
  cpu_h[16] = cpu_h[1] * cpu_h[6] - cpu_h[0] * cpu_h[7];
  cpu_h[17] = cpu_h[0] * cpu_h[4] - cpu_h[1] * cpu_h[3];
  double det = get_det(cpu_h);
  for (int n = 9; n < 18; n++) {
    cpu_h[n] /= det;
  }
}

void get_expanded_box(const double rc, const double* box, int* num_cells, double* ebox)
{
  double volume = get_volume(box);
  // printf("=========vlume %f =======\n", volume);
  double thickness_x = volume / get_area(0, box);
  double thickness_y = volume / get_area(1, box);
  double thickness_z = volume / get_area(2, box);
  num_cells[0] = int(ceil(2.0 * rc / thickness_x));
  num_cells[1] = int(ceil(2.0 * rc / thickness_y));
  num_cells[2] = int(ceil(2.0 * rc / thickness_z));

  ebox[0] = box[0] * num_cells[0];
  ebox[3] = box[3] * num_cells[0];
  ebox[6] = box[6] * num_cells[0];
  ebox[1] = box[1] * num_cells[1];
  ebox[4] = box[4] * num_cells[1];
  ebox[7] = box[7] * num_cells[1];
  ebox[2] = box[2] * num_cells[2];
  ebox[5] = box[5] * num_cells[2];
  ebox[8] = box[8] * num_cells[2];

  get_inverse(ebox);

  // printf("===ebox[0-8]  %f %f %f %f %f %f %f %f %f  ===\n", ebox[0], ebox[1], ebox[2], ebox[3], ebox[4], ebox[5], ebox[6], ebox[7], ebox[8]);
  // printf("===ebox[9-17] %f %f %f %f %f %f %f %f %f  ===\n", ebox[9], ebox[10], ebox[11], ebox[12], ebox[13], ebox[14], ebox[15], ebox[16], ebox[17]);

}

void applyMicOne(double& x12)
{
  while (x12 < -0.5)
    x12 += 1.0;
  while (x12 > +0.5)
    x12 -= 1.0;
}

void apply_mic_small_box(const double* ebox, double& x12, double& y12, double& z12)
{
  double sx12 = ebox[9] * x12 + ebox[10] * y12 + ebox[11] * z12;
  double sy12 = ebox[12] * x12 + ebox[13] * y12 + ebox[14] * z12;
  double sz12 = ebox[15] * x12 + ebox[16] * y12 + ebox[17] * z12;
  applyMicOne(sx12);
  applyMicOne(sy12);
  applyMicOne(sz12);
  x12 = ebox[0] * sx12 + ebox[1] * sy12 + ebox[2] * sz12;
  y12 = ebox[3] * sx12 + ebox[4] * sy12 + ebox[5] * sz12;
  z12 = ebox[6] * sx12 + ebox[7] * sy12 + ebox[8] * sz12;
}

void find_neighbor_compute(
  const int ntypes,
  const int N,
  const int MN,
  const double rcut,
  const std::vector<double> rcut_type,
  const std::vector<int> atom_type_map,
  const std::vector<double> box,
  const std::vector<double> position,
  int* num_cells,
  double* ebox,
  std::vector<int>& dp_NL,
  std::vector<double>& dp_r12
  )
{
  get_expanded_box(rcut, box.data(), num_cells, ebox);
  const int size_nl = ntypes * MN;
  const int size_r12 = ntypes * MN * 3;
  
  const double* g_x = position.data();
  const double* g_y = position.data() + N;
  const double* g_z = position.data() + N * 2;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
  std::vector<int> count_t(ntypes);
  for (int n1 = 0; n1 < N; ++n1) {
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for(int i=0; i < ntypes; i++){
        count_t[i] = 0;
    }
    for (int n2 = 0; n2 < N; ++n2) {
      for (int ia = 0; ia < num_cells[0]; ++ia) {
        for (int ib = 0; ib < num_cells[1]; ++ib) {
          for (int ic = 0; ic < num_cells[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            double delta[3];
            delta[0] = box[0] * ia + box[1] * ib + box[2] * ic;
            delta[1] = box[3] * ia + box[4] * ib + box[5] * ic;
            delta[2] = box[6] * ia + box[7] * ib + box[8] * ic;

            double x12 = g_x[n2] + delta[0] - x1;
            double y12 = g_y[n2] + delta[1] - y1;
            double z12 = g_z[n2] + delta[2] - z1;

            apply_mic_small_box(ebox, x12, y12, z12);

            double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            int typej = atom_type_map[n2];

            if (distance_square < rcut_type[typej]*rcut_type[typej]) {
                // if (n1 == 0 or n1 == 10) printf("n1 %d xyz %f %f %f n2 %d tj %d xyz %f %f %f r12 %f %f %f ntypes %d MN %d\n",\
                    n1, x1, y1, z1, n2, typej, g_x[n2], g_y[n2], g_z[n2], x12, y12, z12, ntypes, MN);

              dp_NL[ n1 * size_nl  + typej * MN     + count_t[typej]        ] = n2 + 1; // DP 下标从1开始
              dp_r12[n1 * size_r12 + typej * MN * 3 + count_t[typej] * 3    ] = x12;
              dp_r12[n1 * size_r12 + typej * MN * 3 + count_t[typej] * 3 + 1] = y12;
              dp_r12[n1 * size_r12 + typej * MN * 3 + count_t[typej] * 3 + 2] = z12;
              count_t[typej] += 1;
            }
          }
        }
      }
    }
  }
}
} // namespace

DP_CPU::DP_CPU() {
  num_cells[0] = num_cells[1] = num_cells[2] = 0;
  for (int i = 0; i < 18; ++i) {
    ebox[i] = 0.0;
  }
}

void DP_CPU::find_neigh(
  const int ntypes,
  const int max_neigh_num,
  const double rcut,
  const std::vector<double>  rcut_type,
  const std::vector<int> atom_type_map,
  const std::vector<double> box,
  const std::vector<double> position)
{
  int N = atom_type_map.size();
  dp_NL.assign(N * ntypes * max_neigh_num, 0); // dp 不需要近邻数量
  dp_r12.assign(N * ntypes * max_neigh_num * 3, 0);
  find_neighbor_compute(
    ntypes, 
    N, 
    max_neigh_num, 
    rcut, 
    rcut_type, 
    atom_type_map, 
    box, 
    position, 
    num_cells, 
    ebox, 
    dp_NL, 
    dp_r12
    );
}
