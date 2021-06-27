#ifndef CORRES_OBJECTS_PROCRUSTES_H
#define CORRES_OBJECTS_PROCRUSTES_H

#include <vector>
#include <random>
#include <numeric>
#include <chrono>

#include <Eigen/Dense>


namespace flow_data_gen {

    inline std::vector<int> generateShuffleIndices(int n) {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine e(seed);
		std::vector<int> indices(n);
		std::iota(indices.begin(), indices.end(), 0);
		std::shuffle(indices.begin(), indices.end(), e);
		return indices;
	}

	class Procrustes {
	public:
		static Eigen::Matrix4f estimatePose(const std::vector<Eigen::Vector3f>& sourcePoints, const std::vector<Eigen::Vector3f>& targetPoints) {
            if (sourcePoints.size() != targetPoints.size()) {
                std::cout << "The number of source and target points should be the same, since every source point is matched with corresponding target point." << std::endl;
                exit(1);
            }

			// We estimate the pose between source and target points using Procrustes algorithm.
			// Our shapes have the same scale, therefore we don't estimate scale. We estimated rotation and translation
			// from source points to target points.

			auto sourceMean = Procrustes::computeMean(sourcePoints);
			auto targetMean = Procrustes::computeMean(targetPoints);

			Eigen::Matrix3f rotation = Procrustes::estimateRotation(sourcePoints, sourceMean, targetPoints, targetMean);
			Eigen::Vector3f translation = Procrustes::computeTranslation(sourceMean, targetMean, rotation);

			Eigen::Matrix4f estimatedPose = Eigen::Matrix4f::Identity();
			estimatedPose.block(0, 0, 3, 3) = rotation;
			estimatedPose.block(0, 3, 3, 1) = translation;

			return estimatedPose;
		}

	private:
		static Eigen::Vector3f computeMean(const std::vector<Eigen::Vector3f>& points) {
			const unsigned nPoints = points.size();
			Eigen::Vector3f mean = Eigen::Vector3f::Zero();
			for (int i = 0; i < nPoints; ++i) {
				mean += points[i];
			}
			mean /= nPoints;
			return mean;
		}

		static Eigen::Matrix3f estimateRotation(
			const std::vector<Eigen::Vector3f>& sourcePoints, 
			const Eigen::Vector3f& sourceMean, const std::vector<Eigen::Vector3f>& targetPoints, 
			const Eigen::Vector3f& targetMean
		) {
			const unsigned nPoints = sourcePoints.size();
			Eigen::MatrixXf sourceMatrix(nPoints, 3);
			Eigen::MatrixXf targetMatrix(nPoints, 3);

			for (int i = 0; i < nPoints; ++i) {
				sourceMatrix.block(i, 0, 1, 3) = (sourcePoints[i] - sourceMean).transpose();
				targetMatrix.block(i, 0, 1, 3) = (targetPoints[i] - targetMean).transpose();
			}

			Eigen::Matrix3f A = targetMatrix.transpose() * sourceMatrix;
			Eigen::JacobiSVD<Eigen::Matrix3f> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
			const Eigen::Matrix3f& U = svd.matrixU();
			const Eigen::Matrix3f& V = svd.matrixV();

			const float d = (U * V.transpose()).determinant();
			Eigen::Matrix3f D = Eigen::Matrix3f::Identity();
			D(2, 2) = d;

			Eigen::Matrix3f R = U * D * V.transpose();
			return R;
		}

		static Eigen::Vector3f computeTranslation(const Eigen::Vector3f& sourceMean, const Eigen::Vector3f& targetMean, const Eigen::Matrix3f& rotation) {
			// y - y_mean = R * (x - x_mean) => y = R * x + (y_mean - R * x_mean)
			Eigen::Vector3f translation = targetMean - rotation * sourceMean;
			return translation;
		}
	};


    class ProcrustesRANSAC {
	public:
		static Eigen::Matrix4f estimatePose(
			const std::vector<Eigen::Vector3f>& sourcePoints, 
			const std::vector<Eigen::Vector3f>& targetPoints, 
			int numIterations, 
			int numSamples,
			float inlierDistance
		) {
			if (sourcePoints.size() != targetPoints.size()) {
                std::cout << "The number of source and target points should be the same, since every source point is matched with corresponding target point." << std::endl;
                exit(1);
            }
            if (sourcePoints.size() < numSamples) {
                std::cout << "We cannot sample more than the number of existing point matches." << std::endl;
                exit(1);
            }

			Eigen::Matrix4f bestPose = Eigen::Matrix4f::Identity();
			int maxInliers = 0;

			for (int i = 0; i < numIterations; i++) {
				std::vector<Eigen::Vector3f> sourcePointsSampled, targetPointsSampled;
				ProcrustesRANSAC::sampleInliers(sourcePoints, targetPoints, numSamples, sourcePointsSampled, targetPointsSampled);
				
				Eigen::Matrix4f pose = Procrustes::estimatePose(sourcePointsSampled, targetPointsSampled);

				int numInliers = ProcrustesRANSAC::computeNumInliers(sourcePoints, targetPoints, inlierDistance, pose);
				if (numInliers > maxInliers) {
					bestPose = pose;
				}
			}

			// At the end, we compute the pose using all inliers of the best pose.
			std::vector<Eigen::Vector3f> sourcePointsInliers, targetPointsInliers;
			ProcrustesRANSAC::computeInliers(sourcePoints, targetPoints, inlierDistance, bestPose, sourcePointsInliers, targetPointsInliers);

			Eigen::Matrix4f finalPose = Procrustes::estimatePose(sourcePointsInliers, targetPointsInliers);

			return finalPose;
		}

	private:
		static void sampleInliers(
			const std::vector<Eigen::Vector3f>& sourcePointsAll,
			const std::vector<Eigen::Vector3f>& targetPointsAll,
			int numSamples,
			std::vector<Eigen::Vector3f>& sourcePointsSampled,
			std::vector<Eigen::Vector3f>& targetPointsSampled
		) {
			std::vector<int> shuffledIndices = generateShuffleIndices(sourcePointsAll.size());
			if (shuffledIndices.size() < numSamples) {
                std::cout << "We cannot sample more than the number of existing point matches." << std::endl;
                exit(1);
            } 

			sourcePointsSampled.clear();
			sourcePointsSampled.resize(numSamples);
			targetPointsSampled.clear();
			targetPointsSampled.resize(numSamples);

			for (int i = 0; i < numSamples; i++) {
				int matchIdx = shuffledIndices[i];
				sourcePointsSampled[i] = sourcePointsAll[matchIdx];
				targetPointsSampled[i] = targetPointsAll[matchIdx];
			}
		};

		static int computeNumInliers(
			const std::vector<Eigen::Vector3f>& sourcePointsAll,
			const std::vector<Eigen::Vector3f>& targetPointsAll,
			float inlierDistance,
			const Eigen::Matrix4f& pose
		) {
			int nInliers = 0;
			int nMatches = sourcePointsAll.size();

			for (int i = 0; i < nMatches; i++) {
				Eigen::Vector3f transformedSourcePoint = pose.block(0, 0, 3, 3) * sourcePointsAll[i] + pose.block(0, 3, 3, 1);
				float distance = (transformedSourcePoint - targetPointsAll[i]).norm();
				if (distance <= inlierDistance) {
					nInliers++;
				}
			}

			return nInliers;
		};
		
		static void computeInliers(
			const std::vector<Eigen::Vector3f>& sourcePointsAll,
			const std::vector<Eigen::Vector3f>& targetPointsAll,
			float inlierDistance,
			const Eigen::Matrix4f& pose,
			std::vector<Eigen::Vector3f>& sourcePointsInliers,
			std::vector<Eigen::Vector3f>& targetPointsInliers
		) {
			int nMatches = sourcePointsAll.size();

			sourcePointsInliers.clear();
			sourcePointsInliers.reserve(nMatches);
			targetPointsInliers.clear();
			targetPointsInliers.reserve(nMatches);

			for (int i = 0; i < nMatches; i++) {
				Eigen::Vector3f transformedSourcePoint = pose.block(0, 0, 3, 3) * sourcePointsAll[i] + pose.block(0, 3, 3, 1);
				float distance = (transformedSourcePoint - targetPointsAll[i]).norm();
				if (distance <= inlierDistance) {
					sourcePointsInliers.push_back(sourcePointsAll[i]);
					targetPointsInliers.push_back(targetPointsAll[i]);
				}
			}
		};
	};

} // namespace flow_data_gen

#endif
