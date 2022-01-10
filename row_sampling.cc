// g++ -O2 row_sampling.cc -o row_sampling

#include <bits/stdc++.h>
using namespace std;

struct AlgorithmConfig {
  std::vector<int> input_shape;
  std::vector<int> rank;
  double l2_regularization_strength;
  std::string algorithm;
  int random_seed;
  double epsilon;
  double delta;
  double downsampling_ratio;
  int max_num_steps;
  std::string rre_gap_tol;
  std::string verbose;
  int step;  // This is not part of the original AlgorithmConfig.
};

inline double str_to_double(const std::string& s) {
  stringstream ss(s);
  double ret;
  ss >> ret;
  return ret;
}

inline int str_to_int(const std::string& s) {
  stringstream ss(s);
  int ret;
  ss >> ret;
  return ret;
}

inline std::string int_to_str(const int n) {
  stringstream ss;
  ss << n;
  std::string ret;
  ss >> ret;
  return ret;
}

inline std::string trim_string(const std::string& s) {
  stringstream ss(s);
  std::string ret;
  ss >> ret;
  return ret;
}

inline std::vector<int> str_to_vectorint(const std::string& s) {
  stringstream ss(s);
  std::vector<int> ret;
  std::vector<std::string> tokens;
  while (ss.good()) {
    std::string token;
    std::getline(ss, token, ',');
    tokens.push_back(token);
  }
  for (const auto& token : tokens) {
    std::string clean_token;
    for (const char c : token) {
      if (isdigit(c)) clean_token.push_back(c);
    }
    ret.push_back(str_to_int(clean_token));
  }
  return ret;
}

AlgorithmConfig ReadAlgorithmConfig() {
  ifstream file("tmp/algorithm_config.txt");
  AlgorithmConfig config;
  string line;
  while (getline(file, line)) {
    stringstream ss(line);
    string field, value;
    ss >> field;
    getline(ss, value);
    // cout << " * " << line << " --> " << field << ": " << value << endl;
    if (field == "input_shape") {
      config.input_shape = str_to_vectorint(value);
    } else if (field == "rank") {
      config.rank = str_to_vectorint(value);
    } else if (field == "l2_regularization_strength") {
      config.l2_regularization_strength = str_to_double(value);
    } else if (field == "algorithm") {
      config.algorithm = trim_string(value);
    } else if (field == "random_seed") {
      config.random_seed = str_to_int(value);
    } else if (field == "epsilon") {
      config.epsilon = str_to_double(value);
    } else if (field == "delta") {
      config.delta = str_to_double(value);
    } else if (field == "downsampling_ratio") {
      config.downsampling_ratio = str_to_double(value);
    } else if (field == "max_num_steps") {
      config.max_num_steps = str_to_int(value);
    } else if (field == "rre_gap_tol") {
      config.rre_gap_tol = trim_string(value);
    } else if (field == "verbose") {
      config.verbose = trim_string(value);
    } else if (field == "step") {
      config.step = str_to_int(value);
    } else {
      cout << "Error: Unknown field in ReadAlgorithmConfig()." << endl;
      assert(false);
    }
  }
  return config;
}

std::vector<double> ReadVector(const std::string& filename) {
  std::vector<double> vec;
  ifstream file(filename);
  double value;
  while (file >> value) {
    vec.push_back(value);
  }
  return vec;
}

using Matrix = std::vector<std::vector<double>>;

Matrix ReshapeVectorizedMatrixToMatrix(const std::vector<double>& v, const int
    num_rows, const int num_cols) {
  assert(v.size() == num_rows * num_cols);
  Matrix matrix(num_rows, std::vector<double>(num_cols));
  int idx = 0;
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      matrix[i][j] = v[idx];
      idx++;
    }
  }
  return matrix;
}

// Input:
// - Cumulative density functions for each factor matrix.
// - K_leverage_score_sum equals d if l2_regularization = 0.0. Only useful as we
//   generalize to ridge leverage scores.
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
    row_indices_and_probability = {row_indices, probability};
  } else {
    r = uniform_distribution(rng);
    int row_index = r * d;
    if (row_index == d) row_index--;
    row_indices_and_probability = {{-1, row_index}, 1.0 / Z};
    // cout << "sample ridge row: " << row_index << " " << 1.0 / Z << endl;
  }
  return row_indices_and_probability;
}

// Can't reliably return an int because if the return value is 0, the
// subprocess terminates because it thinks it is main()?
vector<int> ConvertRowIndexToVectorIndex(const vector<int>& coord,
    const vector<int>& input_shape) {
  assert(coord.size() == input_shape.size());
  vector<long long> suffix_products(coord.size());
  for (int i = input_shape.size() - 1; i >= 0; i--) {
    if (i == input_shape.size() - 1) {
      suffix_products[i] = input_shape[i];
    } else {
      suffix_products[i] = input_shape[i] * suffix_products[i + 1];
    }
  }
  long long linear_index = 0;
  for (int i = 0; i < input_shape.size() - 1; i++) {
    linear_index += suffix_products[i + 1] * coord[i];
  }
  linear_index += coord.back();
  std::vector<int> ret;
  ret.push_back(linear_index);
  return ret;
}

int main() {
  // Read algorithm config, leverage scores, and Tucker decomp from `tmp/` dir.
  cout << "[row_sampling.cc] read algorithm config" << endl;
  const AlgorithmConfig config = ReadAlgorithmConfig();
  const int ndim = config.input_shape.size();
  assert(config.rank.size() == ndim);

  cout << "[row_sampling.cc] read leverage scores" << endl;
  vector<vector<double>> leverage_scores(ndim);
  for (int n = 0; n < ndim; n++) {
    string filename = "tmp/leverage_scores_";
    filename += int_to_str(n);
    filename += "_vec.txt";
    leverage_scores[n] = ReadVector(filename);
  }
  cout << "[row_sampling.cc] read factor matrices" << endl;
  vector<Matrix> factor_matrices(ndim);
  for (int n = 0; n < ndim; n++) {
    string filename = "tmp/factor_matrix_";
    filename += int_to_str(n);
    filename += "_vec.txt";
    std::vector<double> vectorized_matrix = ReadVector(filename); 
    factor_matrices[n] = ReshapeVectorizedMatrixToMatrix(vectorized_matrix,
        config.input_shape[n], config.rank[n]);
    cout << "[row_sampling.cc] " << " - dim " << n << ": "
         << factor_matrices[n].size() << " " << factor_matrices[n][0].size() << endl;
  }
  cout << "[row_sampling.cc] read core tensor" << endl;
  vector<double> core_tensor_vec = ReadVector("tmp/core_tensor_vec.txt");

  // Note: This is one approach to constructing the sketched instance. It is
  // almost definitely too slow, though.
  //cout << "[row_sampling.cc] read input tensor (vectorized)" << endl;
  //vector<double> input_tensor_vec = ReadVector("tmp/input_tensor_vec.txt");

  // Start processing data...
  cout << "[row_sampling.cc] start doing stuff..." << endl;
  
  // Compute number of required samples.
  long long d = 1;
  for (int n = 0; n < ndim; n++) {
    d *= config.rank[n];
  }
  const double epsilon = config.epsilon;
  const double delta = config.delta;
  double sample_term_1 = 420 * log(4*d / delta);
  double sample_term_2 = pow(delta * epsilon, -1);
  long long num_samples = 4 * d * 2 * max(sample_term_1, sample_term_2);
  // NOTE: Lower sample complexity even more to see if we lose solve accuracy.
  num_samples *= config.downsampling_ratio;
  cout << "[row_sampling.cc] num_samples: " << num_samples << endl;

  // Set up RNG.
  mt19937 rng;
  rng.seed(config.step);  // Need to sample different rows in each step.
  uniform_real_distribution<double> uniform_distribution(0.0, 1.0);

  // Prepare each factor's leverage score sampling distribution.
  cout << "[row_sampling.cc] prepare leverage score distributions" << endl;
  double K_leverage_score_sum = 1.0;
  vector<vector<double>> factor_matrix_ls_cdf(ndim);
  for (int n = 0; n < ndim; n++) {
    double factor_matrix_leverage_score_sum = 0.0;
    for (int i = 0; i < config.input_shape[n]; i++) {
      factor_matrix_leverage_score_sum += leverage_scores[n][i];
    }
    K_leverage_score_sum *= factor_matrix_leverage_score_sum;

    factor_matrix_ls_cdf[n] = vector<double>(config.input_shape[n]);
    for (int i = 0; i < config.input_shape[n]; i++) {
      factor_matrix_ls_cdf[n][i] = leverage_scores[n][i];
      if (i > 0) factor_matrix_ls_cdf[n][i] += factor_matrix_ls_cdf[n][i - 1];
    }
    for (int i = 0; i < config.input_shape[n]; i++) {
      factor_matrix_ls_cdf[n][i] /= factor_matrix_leverage_score_sum;
      // cout << n << " " << i << " cdf: " << factor_matrix_ls_cdf[n][i] << endl;
    }
  }
  cout << "[row_sampling.cc] K_leverage_score_sum: " << K_leverage_score_sum << endl;

  cout << "[row_sampling.cc] start drawing " << num_samples << " samples..." << endl;
  // TODO(fahrbach): Upgrade to unordered_map?
  auto start = std::chrono::steady_clock::now();
  map<pair<vector<int>, double>, int> sampled_ridge_rows;
  vector<pair<vector<int>, double>> sampled_factor_rows;
  for (int t = 0; t < num_samples; t++) {
    auto row_prob = SampleFromAugmentedDistribution(factor_matrix_ls_cdf,
        K_leverage_score_sum, d, rng, uniform_distribution);
    if (row_prob.first[0] == -1) {
      sampled_ridge_rows[row_prob]++;
    } else {
      sampled_factor_rows.push_back(row_prob);
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  cout << "[row_sampling.cc] total time: " << elapsed_seconds.count() << "s | per sample: " << elapsed_seconds.count() / num_samples << "s" << endl;

  // Write everything to tmp/ separately.
  // - sampled_row_indices_factor_matrix_{n}.txt: row of each factor matrix
  // - sampled_row_indices.txt: combined row index in the Kroneckor matrix K.
  // - sampled_row_indices_probability.txt: sampling probability of row in K.
  // - [similar stats for "fake" ridge row indices"]
  for (int n = 0; n < ndim; n++) {
    string filename = "tmp/sampled_row_indices_factor_matrix_";
    filename += int_to_str(n);
    filename += ".txt";
    ofstream output_file(filename);
    for (const auto& entry : sampled_factor_rows) {
      const vector<int>& row_indices = entry.first;
      output_file << row_indices.at(n) << "\n";
    }
  }
  {
    ofstream output_indices("tmp/sampled_row_indices.txt");
    ofstream output_probability("tmp/sampled_row_probability.txt");
    ofstream output_weight("tmp/sampled_row_weight.txt");
    for (const auto& entry : sampled_factor_rows) {
      const vector<int>& row_indices = entry.first;
      const double probability = entry.second;
      vector<int> vector_index =
        ConvertRowIndexToVectorIndex(row_indices, config.input_shape);
      output_indices << vector_index.front() << "\n";
      output_probability << probability << "\n";
      output_weight << 1 << "\n";  // rows may be duplicated for now.
    }
  }
  {
    ofstream output_indices("tmp/sampled_ridge_indices.txt");
    ofstream output_probability("tmp/sampled_ridge_probability.txt");
    ofstream output_weight("tmp/sampled_ridge_weight.txt");
    for (const auto& kv : sampled_ridge_rows) {
      const vector<int>& row_indices = kv.first.first;
      const double probability = kv.first.second;
      const double freq = kv.second;
      //assert(row_indices.size() == 2 && row_indices[0] == -1);
      output_indices << row_indices[1] << "\n";
      output_probability << probability << "\n";
      output_weight << freq << "\n";
    }
  }
  cout << "[row_sampling.cc] return" << endl;
  return 0;
}
