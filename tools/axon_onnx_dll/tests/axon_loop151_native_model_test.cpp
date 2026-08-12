#include "../include/axon_loop151_native_model.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

int main() {
  std::string error;
  auto logreg = axon_loop151_native::NativeScoreModel::load_document(
      R"({"model_type":"logreg","n_features":2,"coef":[[2.0,-1.0]],"intercept":[0.5],"scaler":{"mean":[0.0,0.0],"scale":[1.0,2.0]}})",
      "logreg",
      error);
  assert(logreg && error.empty());
  const float logreg_score = logreg->predict_probability({1.0f, 2.0f}, &error);
  assert(error.empty() && std::fabs(logreg_score - 0.8175745f) < 1.0e-5f);

  auto pipeline = axon_loop151_native::NativeScoreModel::load_document(
      R"({"model_type":"pipeline","steps":[{"name":"scaler","model_type":"standard_scaler","mean":[0.0,0.0],"scale":[1.0,2.0]},{"name":"classifier","model_type":"logistic_regression","coef":[[2.0,-1.0]],"intercept":[0.5]}]})",
      "pipeline",
      error);
  assert(pipeline && error.empty());
  const float pipeline_score = pipeline->predict_probability({1.0f, 2.0f}, &error);
  assert(error.empty() && std::fabs(pipeline_score - logreg_score) < 1.0e-5f);

  auto hgb = axon_loop151_native::NativeScoreModel::load_document(
      R"({"model_type":"hgb","n_features":1,"baseline_prediction":0.0,"trees":[{"nodes":[{"feature_idx":0,"num_threshold":0.5,"left":1,"right":2,"is_leaf":false},{"value":-2.0,"is_leaf":true},{"value":2.0,"is_leaf":true}]}]})",
      "hgb",
      error);
  assert(hgb && error.empty());
  const float hgb_score = hgb->predict_probability({1.0f}, &error);
  assert(error.empty() && std::fabs(hgb_score - 0.8807971f) < 1.0e-5f);

  auto extra_trees = axon_loop151_native::NativeScoreModel::load_document(
      R"({"model_type":"extra_trees","n_features":1,"trees":[{"nodes":[{"feature_idx":0,"threshold":0.5,"left":1,"right":2,"is_leaf":false},{"value":[3,1],"is_leaf":true},{"value":[1,3],"is_leaf":true}]}]})",
      "extra_trees",
      error);
  assert(extra_trees && error.empty());
  const float extra_score = extra_trees->predict_probability({1.0f}, &error);
  assert(error.empty() && std::fabs(extra_score - 0.75f) < 1.0e-5f);

  auto stack = axon_loop151_native::NativeStackModel::load_document(
      R"({"schema":"axon_loop151_native_stack_v1","n_features":2,"drop_base_prob_features":false,"base_models":[{"native_model":{"model_type":"constant","n_features":2,"probability":0.2}},{"native_model":{"model_type":"constant","n_features":2,"probability":0.8}}],"meta_model":{"native_model":{"model_type":"logreg","n_features":9,"coef":[[0,0,0,0,0,10,0,0,0]],"intercept":[-5]}}})",
      "stack",
      error);
  assert(stack && error.empty());
  const float stack_score = stack->predict_probability({1.0f, 2.0f}, &error);
  assert(error.empty() && stack_score > 0.95f);
  return 0;
}
