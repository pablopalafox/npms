#pragma once

#include <iostream>

#include <torch/extension.h>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace image_proc {

    void backproject_depth_ushort(py::array_t<unsigned short>& in, py::array_t<float>& out, float fx, float fy, float cx, float cy, float normalizer);

    void backproject_depth_float(py::array_t<float>& in, py::array_t<float>& out, float fx, float fy, float cx, float cy);

    void compute_normals(const py::array_t<float>& point_image, py::array_t<float>& normals, float epsilon);

    void compute_normals_via_pca(const py::array_t<float>& point_image, py::array_t<float>& normals, const int kernel_size, const float max_distance);
    
    float estimate_rigid_pose(
         const py::array_t<float>& sourcePointsArray, const py::array_t<float>& targetPointsArray, 
         int numIterationsRANSAC, int numSamplesRANSAC, int minPointsRANSAC, float inlierDistanceRANSAC, float inlierDistanceFinal,
         py::array_t<float>& estimatedPose
    );

    void compute_mesh_from_depth(
        const py::array_t<float>& pointImage, float maxTriangleEdgeDistance, 
        py::array_t<float>& vertexPositions, py::array_t<int>& faceIndices
    );

    void compute_validity_and_sign_mask(
        const py::array_t<float>& grid_points_cam, 
        const py::array_t<bool>& valid_depth_mask_image, 
        const py::array_t<float>& depth_image,
        float fx, float fy, float cx, float cy, int w, int h,
        float truncation,
        py::array_t<bool>& validity_grid_mask,
        py::array_t<float>& sign_grid_mask
    );
    
} // namespace image_proc