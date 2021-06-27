import os
import numpy as np
import torch
import torch.nn.functional as fn
from smplx import body_models
from body_model.smpl.rodrigues import rodrigues, derive_rodrigues, axis_angle, derive_axis_angle


class Smpl:
    def __init__(self, device, dtype=torch.float32):
        self.dtype = dtype
        self.device = device
        
        self_dir = os.path.dirname(os.path.realpath(__file__))

        self.model = body_models.create(self_dir, 'smpl', gender='male', dtype=dtype).to(self.device)
        self.model.zero_grad()
        self.faces = self.model.faces
        self.template = self.model.v_template
        self.vertex_joint_selector = self.model.vertex_joint_selector
        self.n_faces = self.model.get_num_faces()
        self.n_vertices = self.model.get_num_verts()

        self.default_transl = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

    def rotate_around_joint(self, rotations, joints):
        batch_size = rotations.shape[0]
        r = torch.eye(4, dtype=self.dtype, device=self.device)[None].repeat(batch_size, 1, 1)
        t = torch.eye(4, dtype=self.dtype, device=self.device)[None].repeat(batch_size, 1, 1)
        r[:, 0:3, 0:3] = rotations
        t[:, 0:3, 3] = joints

        result = torch.einsum('bij,bjk->bik', t, r)
        return result     

    def __call__(self, transl=None, orient=None, betas=None, body_pose=None, offsets=None, t_pose=None):

        if transl is None:
            transl = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

        if orient is None:
            orient = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

        if betas is None:
            betas = torch.zeros((1, 10), dtype=torch.float32, device=self.device)

        if body_pose is None:
            body_pose = torch.zeros((1, 69), dtype=torch.float32, device=self.device)

        assert len(orient.shape) == len(body_pose.shape) and orient.shape[0] == body_pose.shape[0], f"{orient.shape} vs {body_pose.shape}"
        rot_vecs = torch.cat([orient, body_pose], dim=1).view(24, 3)
        rot_mats = rodrigues(axis_angle(rot_vecs)).view(-1, 3, 3)

        # t pose
        if t_pose is not None:
            v_shaped = t_pose
        else:
            v_shaped = self.template + torch.einsum('l,mkl->mk', betas.reshape(-1), self.model.shapedirs)
            if offsets is not None:
                v_shaped += offsets

        # pose offset
        ident = torch.eye(3, dtype=self.dtype, device=self.device)
        pose_feature = (rot_mats[1:, :, :] - ident).view([-1])
        pose_offsets = torch.matmul(pose_feature, self.model.posedirs).view(-1, 3)
        v_posed = pose_offsets + v_shaped

        joints = torch.einsum('ik,ji->jk', v_shaped, self.model.J_regressor)
        num_joints = self.model.J_regressor.shape[0]

        rel_joints = joints.clone()
        rel_joints[1:] -= rel_joints[self.model.parents[1:]]

        part_transforms = self.rotate_around_joint(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3)
        ).reshape(num_joints, 4, 4)

        transform_chain = [part_transforms[0]]

        for i in range(1, self.model.parents.shape[0]):
            curr_res = torch.matmul(transform_chain[self.model.parents[i]], part_transforms[i])
            transform_chain.append(curr_res)

        palette = torch.stack(transform_chain, dim=0)
        W = self.model.lbs_weights
        rel_transforms = palette - fn.pad(
            torch.einsum('bik,bk->bi', palette, fn.pad(joints, [0, 1]))[..., None], [3, 0, 0, 0, 0, 0])
        T = torch.matmul(W, rel_transforms.view(num_joints, 16)).view(-1, 4, 4)

        vertices = torch.einsum('nik,nk->ni', T, fn.pad(v_posed, [0, 1], value=1))[:, 0:3] + transl
        return vertices
        
        # joints = palette[:, :3, 3] + transl
        # joints = self.model.vertex_joint_selector(vertices[None], joints[None])
        # joints = self.joint_mapper(joints)
        # return vertices, joints