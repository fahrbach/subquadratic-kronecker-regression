#include <bits/stdc++.h>
using namespace std;

struct InstanceInfo {
  int num_dimensions, step;
  double l2_regularization, epsilon;
  std::vector<int> num_rows, num_cols;
  std::vector<double> factor_norms;
  std::vector<std::vector<double>> leverage_scores;
};

InstanceInfo ReadLeverageScores() {
  ifstream file("leverage_scores.txt");
  InstanceInfo instance_info;
  file >> instance_info.num_dimensions;
  file >> instance_info.l2_regularization;
  file >> instance_info.epsilon;
  file >> instance_info.step;
  for (int n = 0; n < instance_info.num_dimensions; n++) {
    int num_rows, num_cols;
    double factor_norm;
    file >> num_rows >> num_cols >> factor_norm;
    instance_info.num_rows.push_back(num_rows);
    instance_info.num_cols.push_back(num_cols);
    instance_info.factor_norms.push_back(factor_norm);

    std::vector<double> leverage_scores(num_rows);
    for (int i = 0; i < num_rows; i++) file >> leverage_scores[i];
    instance_info.leverage_scores.push_back(leverage_scores);
  }
  return instance_info;
}

double rand_double() {
  return (double)rand() / RAND_MAX;
}

int main() {
  const auto instance = ReadLeverageScores();

  double leverage_score_sum = 1.0;
  for (int n = 0; n < instance.num_dimensions; n++) {
    double dim_sum = 0.0;
    for (const double leverage_score : instance.leverage_scores.at(n)) {
      dim_sum += leverage_score;
    }
    leverage_score_sum *= dim_sum;
  }
  const double delta = 0.01;


  assert(instance.num_dimensions == 3);
  ofstream output_file("sampled_rows.csv");
  
  // TODO(fahrbach): Use better RNG.
  srand(instance.step);
  long long num_sampled_rows = 0;
  std::vector<std::pair<std::vector<int>, double>> buffer;
  for (int i = 0; i < instance.num_rows[0]; i++) {
    for (int j = 0; j < instance.num_rows[1]; j++) {
      for (int k = 0; k < instance.num_rows[2]; k++) {
        double kronecker_leverage_score = instance.leverage_scores[0][i]
          * instance.leverage_scores[1][j] * instance.leverage_scores[2][k];
        double sample_probability = 16 * kronecker_leverage_score * log(leverage_score_sum / delta) / std::pow(instance.epsilon, 2);
        if (sample_probability > rand_double()) {
          //output_file << i << "," << j << "," << k << "," << kronecker_leverage_score << endl;
          buffer.push_back({{i,j,k}, sample_probability});
          num_sampled_rows++;
        }
      }
    }
  }
  // Write sample row indices and sample probability to file.
  output_file << num_sampled_rows << endl;
  for (const auto& row : buffer) {
    for (const auto index : row.first) output_file << index << ",";
    output_file << row.second << endl;
  }
  return 0;
}
