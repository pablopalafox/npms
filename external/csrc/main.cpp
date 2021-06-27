#include <torch/extension.h>

#include "cpu/image_proc.h"


// Definitions of all methods in the module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("backproject_depth_ushort", &image_proc::backproject_depth_ushort, "Backproject depth image into 3D points");
  m.def("backproject_depth_float", &image_proc::backproject_depth_float, "Backproject depth image into 3D points");
  m.def("compute_normals", &image_proc::compute_normals, "Compute normals given a point image");
  m.def("compute_normals_via_pca", &image_proc::compute_normals_via_pca, "Compute normals given a point image via PCA");

  m.def("estimate_rigid_pose", &image_proc::estimate_rigid_pose, "Estimates rigid pose using sparse matches, with Procrustes and RANSAC");
  
  m.def("compute_mesh_from_depth", &image_proc::compute_mesh_from_depth, "Estimates rigid pose using sparse matches, with Procrustes and RANSAC");
  
  m.def("compute_validity_and_sign_mask", &image_proc::compute_validity_and_sign_mask, "Estimates rigid pose using sparse matches, with Procrustes and RANSAC");
  
}