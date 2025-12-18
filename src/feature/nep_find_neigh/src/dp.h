#pragma once
#include <string>
#include <vector>

class DP_CPU
{
public:
  DP_CPU();

  void find_neigh(
    const int ntypes,
    const int max_neigh_num,
    const double rcut,
    const std::vector<double>  rcut_type,
    const std::vector<int> atom_type_map,
    const std::vector<double> box,
    const std::vector<double> position);
  std::vector<int>    dp_NL; // dp 不需要近邻数量
  std::vector<double> dp_r12;

  int num_atoms = 0;
  int num_cells[3];
  double ebox[18];
};
