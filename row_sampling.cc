#include <bits/stdc++.h>
using namespace std;

struct InstanceInfo {
  int num_dimensions, step;
  double l2_regularization, epsilon, delta;
  // Factor matrices:
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
  file >> instance_info.delta;
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

// Input:
// - Cumulative density functions for each of the factor matrices.
// - K_leverage_score_sum should equal d if l2_regularization = 0.0. Only useful
//   as we generalize to ridge leverage scores.
// - d is the number of columns (i.e., the rank of the augmented design matrix).
// - Mutable RNG.
//
// Output:
// - If we sample a row of the original design matrix, we return the row indexed
//   by the row indices of its factor matrices (e.g., {i, j, k}).
// - If we sample an augmented row, then we encode it by {-1, row_index} where
//   row_index \in {0, 1, ..., d - 1}.
// - The second part of the pair is the probability with which this row was
//   sampled.
inline pair<vector<int>, double> SampleFromAugmentedDistribution(
    const vector<vector<double>>& factor_matrix_ls_cdf,
    const double& K_leverage_score_sum,
    const long long& d,
    mt19937& rng,
    uniform_real_distribution<double>& uniform_distribution) {
  pair<vector<int>, double> row_indices_and_probability;
  // Flip coin to branch between original rows and augmented rows.
  const double Z = K_leverage_score_sum + d;
  double r = uniform_distribution(rng);
  if (Z * r <= K_leverage_score_sum) {  // r <= K_leverage_score_sum / Z
    vector<int> row_indices;
    double probability = 1.0;
    for (int n = 0; n < factor_matrix_ls_cdf.size(); n++) {
      r = uniform_distribution(rng);
      auto it = upper_bound(factor_matrix_ls_cdf[n].begin(), factor_matrix_ls_cdf[n].end(), r);
      int row_index = it - factor_matrix_ls_cdf[n].begin();
      if (row_index == factor_matrix_ls_cdf[n].size()) row_index--;
      double p = factor_matrix_ls_cdf[n][row_index];
      if (row_index > 0) p -= factor_matrix_ls_cdf[n][row_index - 1];
      //cout << "sample original row: " << n << " " << r << ": " << *it << " " << row_index << " " << p << endl;
      row_indices.push_back(row_index);
      probability *= p;
    }
    row_indices_and_probability.first = row_indices;
    row_indices_and_probability.second = probability;
  } else {
    r = uniform_distribution(rng);
    int row_index = r * d;
    if (row_index == d) row_index--;
    row_indices_and_probability.first = {-1, row_index};
    row_indices_and_probability.second = 1.0 / Z;
    // cout << "sample ridge row: " << row_index << " " << 1.0 / Z << endl;
  }
  return row_indices_and_probability;
}

int main() {
  const auto instance = ReadLeverageScores();

  // Compute number of columns in the design matrix.
  long long d = 1;
  for (int n = 0; n < instance.num_dimensions; n++) {
    d *= instance.num_cols[n];
  }
  const double epsilon = instance.epsilon;
  const double delta = instance.delta;
  double sample_term_1 = 420 * log(4*d / delta);
  double sample_term_2 = pow(delta * epsilon, -1);
  long long num_samples = 4 * d * 2 * max(sample_term_1, sample_term_2);
  // NOTE: Lowering the sample complexity even more to see if we lose solve accuracy.
  num_samples *= 0.001;

  mt19937 rng;
  rng.seed(instance.step);  // Need to sample different rows in each step.
  uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

  // Prepare factor matrix leverage score distributions.
  double K_leverage_score_sum = 1.0;
  vector<vector<double>> factor_matrix_ls_cdf(instance.num_dimensions);
  for (int n = 0; n < instance.num_dimensions; n++) {
    double factor_matrix_leverage_score_sum = 0.0;
    for (int i = 0; i < instance.num_rows[n]; i++) {
      factor_matrix_leverage_score_sum += instance.leverage_scores[n][i];
    }
    K_leverage_score_sum *= factor_matrix_leverage_score_sum;

    factor_matrix_ls_cdf[n] = vector<double>(instance.num_rows[n]);
    for (int i = 0; i < instance.num_rows[n]; i++) {
      factor_matrix_ls_cdf[n][i] = instance.leverage_scores[n][i];
      if (i > 0) factor_matrix_ls_cdf[n][i] += factor_matrix_ls_cdf[n][i - 1];
    }
    for (int i = 0; i < instance.num_rows[n]; i++) {
      factor_matrix_ls_cdf[n][i] /= factor_matrix_leverage_score_sum;
      // cout << n << " " << i << " cdf: " << factor_matrix_ls_cdf[n][i] << endl;
    }
  }

  // TODO(fahrbach): Upgrade to unordered_map.
  map<pair<vector<int>, double>, int> frequency;
  for (int t = 0; t < num_samples; t++) {
    auto row_prob = SampleFromAugmentedDistribution(factor_matrix_ls_cdf,
        K_leverage_score_sum, d, rng, uniform_distribution);
    frequency[row_prob]++;
  }

  ofstream output_file("sampled_rows.csv");
  output_file << frequency.size() << "," << num_samples << endl;
  for (const auto& kv : frequency) {
    const vector<int>& row_indices = kv.first.first;
    const double probability = kv.first.second;
    const double freq = kv.second;
    for (const int row_index : row_indices) {
      output_file << row_index << ",";
    }
    output_file << probability << "," << freq << endl;
  }
  return 0;
}
