import torch

source_cloud = torch.randn(1, 100, 3).cuda()
target_cloud = torch.randn(1, 50, 3).cuda()

print(torch.mean(source_cloud))
print(torch.mean(target_cloud))



# ##############################################################################
# print("######")
# from chamferdist import ChamferDistance
# chamferDist = ChamferDistance()
# # dist_forward = chamferDist(source_cloud, target_cloud)
# dist_bidirectional = chamferDist(source_cloud, target_cloud, bidirectional=True)
# print(dist_bidirectional)


# ##############################################################################
# print("######")
# from pytorch3d.loss import chamfer_distance
# dist_forward = chamfer_distance(source_cloud, target_cloud)
# print(dist_forward)


# ##############################################################################
# print("######")

# from pyTorchChamferDistance.chamfer_distance import ChamferDistance as ChamferDistanceChris
# chamfer_dist = ChamferDistanceChris()

# #...
# # points and points_reconstructed are n_points x 3 matrices

# dist1, dist2 = chamfer_dist(source_cloud, target_cloud)
# print(torch.mean(dist1))
# loss = (torch.mean(dist1)) + (torch.mean(dist2))
# print(loss)



# ##############################################################################
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import external.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
chamLoss = dist_chamfer_3D.chamfer_3DDist()
dist1, dist2, idx1, idx2 = chamLoss(source_cloud, target_cloud)
print(torch.mean(dist1))
loss = (torch.mean(dist1)) + (torch.mean(dist2))
print(loss)
