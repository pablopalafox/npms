#include "cpu/image_proc.h"

#include <Eigen/Dense>
#include <map>
#include <string>
#include <cmath>
#include <math.h>

#include "procrustes.h"


namespace image_proc {

    using namespace flow_data_gen;

    using Vec2d = Eigen::Vector2d;
    using Vec2f = Eigen::Vector2f;
    using Vec2i = Eigen::Vector2i;
    using Vec3d = Eigen::Vector3d;
    using Vec3f = Eigen::Vector3f;
    using Vec3i = Eigen::Vector3i;
	using Vec4d = Eigen::Vector4d;
	using Vec4f = Eigen::Vector4f;
	using Vec4i = Eigen::Vector4i;

	using Mat2d = Eigen::Matrix2d;
	using Mat2f = Eigen::Matrix2f;
	using Mat2i = Eigen::Matrix2i;
	using Mat3d = Eigen::Matrix3d;
	using Mat3f = Eigen::Matrix3f;
	using Mat3i = Eigen::Matrix3i;
	using Mat4d = Eigen::Matrix4d;
	using Mat4f = Eigen::Matrix4f;
	using Mat4i = Eigen::Matrix4i;


    void backproject_depth_ushort(py::array_t<unsigned short>& in, py::array_t<float>& out, float fx, float fy, float cx, float cy, float normalizer) {
        assert(in.ndim() == 2);
        assert(out.ndim() == 3);

        int width = in.shape(1);
        int height = in.shape(0);
        assert(out.shape(0) == 3);
        assert(out.shape(1) == height);
        assert(out.shape(2) == width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float depth = float(*in.data(y, x)) / normalizer;

                if (depth > 0) {
                    float pos_x = depth * (x - cx) / fx;
                    float pos_y = depth * (y - cy) / fy;
                    float pos_z = depth;

                    *out.mutable_data(0, y, x) = pos_x;
                    *out.mutable_data(1, y, x) = pos_y;
                    *out.mutable_data(2, y, x) = pos_z;
                }
            }
        }
    }

    void backproject_depth_float(py::array_t<float>& in, py::array_t<float>& out, float fx, float fy, float cx, float cy) {
        assert(in.ndim() == 2);
        assert(out.ndim() == 3);

        int width = in.shape(1);
        int height = in.shape(0);
        assert(out.shape(0) == 3);
        assert(out.shape(1) == height);
        assert(out.shape(2) == width);
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float depth = *in.data(y, x);

                if (depth > 0) {
                    float pos_x = depth * (x - cx) / fx;
                    float pos_y = depth * (y - cy) / fy;
                    float pos_z = depth;

                    *out.mutable_data(0, y, x) = pos_x;
                    *out.mutable_data(1, y, x) = pos_y;
                    *out.mutable_data(2, y, x) = pos_z;
                }
            }
        }
    }

    void compute_normals(const py::array_t<float>& point_image, py::array_t<float>& normals, float epsilon) {
    
        int height = point_image.shape(0);
        int width  = point_image.shape(1);
     
        for (int y = 1; y < (height - 1); y++) {
            for (int x = 1; x < (width - 1); x++) {
                
                Vec3f xp(*point_image.data(y, x + 1, 0), *point_image.data(y, x + 1, 1), *point_image.data(y, x + 1, 2));
                Vec3f xn(*point_image.data(y, x - 1, 0), *point_image.data(y, x - 1, 1), *point_image.data(y, x - 1, 2));
                
                Vec3f yp(*point_image.data(y + 1, x, 0), *point_image.data(y + 1, x, 1), *point_image.data(y + 1, x, 2));
                Vec3f yn(*point_image.data(y - 1, x, 0), *point_image.data(y - 1, x, 1), *point_image.data(y - 1, x, 2));

                if (!xp.allFinite() || !xn.allFinite() || !yp.allFinite() || !yn.allFinite()) {
                    continue;
                }

                Vec3f a = xp - xn;
                Vec3f b = yp - yn;

                Vec3f cross = b.cross(a);

                if (!cross.allFinite()) {
                    continue;
                }

                float norm = cross.norm();

                if (norm < epsilon) {
                    continue;
                }

                cross = cross / norm;

                *normals.mutable_data(y, x, 0) = cross.x();
                *normals.mutable_data(y, x, 1) = cross.y();
                *normals.mutable_data(y, x, 2) = cross.z();
            }
        }
    }

    void compute_normals_via_pca(const py::array_t<float>& point_image, py::array_t<float>& normals, const int kernel_size, const float max_distance) {
    
        int height = point_image.shape(0);
        int width  = point_image.shape(1);

        int ex = kernel_size;
        int ey = kernel_size;
     
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                
                Vec3f point(*point_image.data(y, x, 0), *point_image.data(y, x, 1), *point_image.data(y, x, 2));

                if (!point.allFinite()) {
                    continue;
                }

                int counter = 0;
                Vec3f center(0, 0, 0);
                std::vector<Vec3f> valid_points;

                for (int yi = y - ey / 2; yi <= y + ey / 2; ++yi) {
                    for (int xi = x - ex / 2; xi <= x + ex / 2; ++xi) {
                        if (xi >= 0 && xi < width && yi >= 0 && yi < height) {
                            Vec3f point_neighbor(*point_image.data(yi, xi, 0), *point_image.data(yi, xi, 1), *point_image.data(yi, xi, 2));

                            if (point_neighbor.allFinite() && (point - point_neighbor).norm() <= max_distance) {
                                center += point_neighbor;
                                valid_points.push_back(point_neighbor);
                                ++counter;
                            }
                        }
                    }
                }
                center /= (float)counter;

                Eigen::MatrixXf A(3, counter);
                for (int idx = 0; idx < counter; ++idx) {
                    A.col(idx) = valid_points[idx] - center;
                }

                Eigen::Matrix3f cov = A * A.transpose();
                Eigen::EigenSolver<Eigen::Matrix3f> eigen_solver(cov, true);
                const Vec3f& eigen_value = eigen_solver.eigenvalues().real();

                int min_idx = 0;
                eigen_value.minCoeff(&min_idx);
                Vec3f min_eigen_vector = eigen_solver.eigenvectors().real().col(min_idx);

                Vec3f normal;
                // Use eigen vector facing camera as normal.
                if (min_eigen_vector.dot(point) < 0) {
                    normal = min_eigen_vector.normalized();
                }
                else {
                    normal = -min_eigen_vector.normalized();
                }
            
                *normals.mutable_data(y, x, 0) = normal.x();
                *normals.mutable_data(y, x, 1) = normal.y();
                *normals.mutable_data(y, x, 2) = normal.z();
            }
        }
    }

    float estimate_rigid_pose(
         const py::array_t<float>& sourcePointsArray, const py::array_t<float>& targetPointsArray, 
         int numIterationsRANSAC, int numSamplesRANSAC, int minPointsRANSAC, float inlierDistanceRANSAC, float inlierDistanceFinal,
         py::array_t<float>& estimatedPose
    ) {
        if (numSamplesRANSAC > minPointsRANSAC) {
            std::cout << "The minimum number of points should be at least as big as the sample size." << std::endl;
            exit(1);
        }

        // Compute valid 3D matches.
        int nMatches = sourcePointsArray.shape(0);

        std::vector<Vec3f> sourcePoints, targetPoints; 
        for (int i = 0; i < nMatches; i++) {
            Vec3f sourcePoint(
                *sourcePointsArray.data(i, 0),
                *sourcePointsArray.data(i, 1),
                *sourcePointsArray.data(i, 2)
            );
            Vec3f targetPoint(
                *targetPointsArray.data(i, 0),
                *targetPointsArray.data(i, 1),
                *targetPointsArray.data(i, 2)
            );

            sourcePoints.push_back(sourcePoint);
            targetPoints.push_back(targetPoint);
        }

        int nMatches3D = sourcePoints.size();
        if (nMatches3D < minPointsRANSAC) {
            return 0;
        }

        // Execute Procrustes with RANSAC.
        Mat4f globalRigidPose = ProcrustesRANSAC::estimatePose(
            sourcePoints, targetPoints, numIterationsRANSAC, numSamplesRANSAC, inlierDistanceRANSAC
        );

        // Compute the number of inliers.
        int nInliers = 0;
        for (int i = 0; i < nMatches3D; i++) {
            Eigen::Vector3f transformedSourcePoint = globalRigidPose.block(0, 0, 3, 3) * sourcePoints[i] + globalRigidPose.block(0, 3, 3, 1);
            float distance = (transformedSourcePoint - targetPoints[i]).norm();

            if (distance <= inlierDistanceFinal) {
                nInliers++;
            }
        }

        float inlierRatio = float(nInliers) / nMatches3D;

        // Store the estimated pose.
        estimatedPose.resize({ 4, 4 }, false);
        
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                *estimatedPose.mutable_data(i, j) = globalRigidPose(i, j);
            }
        }

        return inlierRatio;
    }

    void compute_mesh_from_depth(
        const py::array_t<float>& pointImage, float maxTriangleEdgeDistance, 
        py::array_t<float>& vertexPositions, py::array_t<int>& faceIndices
    ) {
        int width = pointImage.shape(2);
        int height = pointImage.shape(1);

        // Compute valid pixel vertices and faces.
        // We also need to compute the pixel -> vertex index mapping for 
        // computation of faces.
        // We connect neighboring pixels on the square into two triangles.
        // We only select valid triangles, i.e. with all valid vertices and
        // not too far apart.
        // Important: The triangle orientation is set such that the normals
        // point towards the camera.
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> faces;

        int vertexIdx = 0;
        std::vector<int> mapPixelToVertexIdx(width * height, -1);

        for (int y = 0; y < height - 1; y++) {
            for (int x = 0; x < width - 1; x++) {
                Eigen::Vector3f obs00(*pointImage.data(0, y, x), *pointImage.data(1, y, x), *pointImage.data(2, y, x));
                Eigen::Vector3f obs01(*pointImage.data(0, y + 1, x), *pointImage.data(1, y + 1, x), *pointImage.data(2, y + 1, x));
                Eigen::Vector3f obs10(*pointImage.data(0, y, x + 1), *pointImage.data(1, y, x + 1), *pointImage.data(2, y, x + 1));
                Eigen::Vector3f obs11(*pointImage.data(0, y + 1, x + 1), *pointImage.data(1, y + 1, x + 1), *pointImage.data(2, y + 1, x + 1));

                int idx00 = y * width + x;
                int idx01 = (y + 1) * width + x;
                int idx10 = y * width + (x + 1);
                int idx11 = (y + 1) * width + (x + 1);

                bool valid00 = obs00.z() > 0;
                bool valid01 = obs01.z() > 0;
                bool valid10 = obs10.z() > 0;
                bool valid11 = obs11.z() > 0;

                if (valid00 && valid01 && valid10) {
                    float d0 = (obs00 - obs01).norm();
                    float d1 = (obs00 - obs10).norm();
                    float d2 = (obs01 - obs10).norm();
                    
                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx00];
                        int vIdx1 = mapPixelToVertexIdx[idx01];
                        int vIdx2 = mapPixelToVertexIdx[idx10];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx00] = vertexIdx;
                            vertices.push_back(obs00);
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }

                if (valid01 && valid10 && valid11) {
                    float d0 = (obs10 - obs01).norm();
                    float d1 = (obs10 - obs11).norm();
                    float d2 = (obs01 - obs11).norm();

                    if (d0 <= maxTriangleEdgeDistance && d1 <= maxTriangleEdgeDistance && d2 <= maxTriangleEdgeDistance) {
                        int vIdx0 = mapPixelToVertexIdx[idx11];
                        int vIdx1 = mapPixelToVertexIdx[idx10];
                        int vIdx2 = mapPixelToVertexIdx[idx01];

                        if (vIdx0 == -1) {
                            vIdx0 = vertexIdx;
                            mapPixelToVertexIdx[idx11] = vertexIdx;
                            vertices.push_back(obs11);
                            vertexIdx++;
                        }
                        if (vIdx1 == -1) {
                            vIdx1 = vertexIdx;
                            mapPixelToVertexIdx[idx10] = vertexIdx;
                            vertices.push_back(obs10);
                            vertexIdx++;
                        }
                        if (vIdx2 == -1) {
                            vIdx2 = vertexIdx;
                            mapPixelToVertexIdx[idx01] = vertexIdx;
                            vertices.push_back(obs01);
                            vertexIdx++;
                        }

                        faces.push_back(Eigen::Vector3i(vIdx0, vIdx1, vIdx2));
                    }
                }
            }
        }

        // Convert to numpy array.
        int nVertices = vertices.size();
        int nFaces = faces.size();

        if (nVertices > 0 && nFaces > 0) {
            // Reference check should be set to false otherwise there is a runtime
            // error. Check why that is the case.
            vertexPositions.resize({ nVertices, 3 }, false);
            faceIndices.resize({ nFaces, 3 }, false);

            for (int i = 0; i < nVertices; i++) {
                *vertexPositions.mutable_data(i, 0) = vertices[i].x();
                *vertexPositions.mutable_data(i, 1) = vertices[i].y();
                *vertexPositions.mutable_data(i, 2) = vertices[i].z();
            }
            
            for (int i = 0; i < nFaces; i++) {
                *faceIndices.mutable_data(i, 0) = faces[i].x();
                *faceIndices.mutable_data(i, 1) = faces[i].y();
                *faceIndices.mutable_data(i, 2) = faces[i].z();
            }
        }
    }

    void compute_validity_and_sign_mask(
        const py::array_t<float>& grid_points_cam, 
        const py::array_t<bool>& valid_depth_mask_image, 
        const py::array_t<float>& depth_image,
        float fx, float fy, float cx, float cy, int w, int h,
        float truncation,
        py::array_t<bool>& validity_grid_mask,
        py::array_t<float>& sign_grid_mask
    ) {
        
        int num_grid_points = grid_points_cam.shape(0);

        for (int pt_id = 0; pt_id < num_grid_points; pt_id++) {

            float x = *grid_points_cam.data(pt_id, 0);
            float y = *grid_points_cam.data(pt_id, 1);
            float z = *grid_points_cam.data(pt_id, 2);

            int px = int(round(fx * x / z + cx));
            int py = int(round(fy * y / z + cy));

            if (px < 0 || px >= w || py < 0 || py >= h) {
                continue;
            }

            bool is_valid_pixel = *valid_depth_mask_image.data(py, px);

            if (!is_valid_pixel) {
                continue;
            }
            
            float d = *depth_image.data(py, px);

            // If the z coordinate of the grid point is further away from the camera 
            // than the surface + some truncation, then we invalidate the grid point
            if (z > (d + truncation)) {
                *validity_grid_mask.mutable_data(pt_id) = false;
            }

            if (z > d) {
                *sign_grid_mask.mutable_data(pt_id) = -1.0;
            }
            
        }
    }

} //namespace image_proc