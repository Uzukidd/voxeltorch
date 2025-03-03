import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d
from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes, knn_points, knn_gather
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Scale


import skimage

from typing import Union


def bbox_normalize(points: torch.Tensor, bbox: torch.Tensor):
    """
        Args:
            points: centered point cloud [B, N, 3]
            bbox: [l, w, h]

        Return:
            normalized points: [B, N, 3] ([-l/2, l/2] x [-w/2, w/2] x [-h/2, h/2]) -> [-1, 1]^3
    """
    points = 2 * points / bbox
    return points

def tsdf2meshes(tsdf_grid: Union[torch.Tensor, np.ndarray], resolution: torch.Tensor = None):
    """
        Args:
            tsdf_grid: [B, l, w, h]

        Return:
    """
    if isinstance(tsdf_grid, torch.Tensor):
        tsdf_grid_numpy = tsdf_grid.detach().cpu().numpy()

    list_vertices = []
    list_faces = []
    list_normals = []
    for batch_mask in range(tsdf_grid.size(0)):
        ind_tsdf_grid = tsdf_grid_numpy[batch_mask]
        vertices, faces, normals, _ = skimage.measure.marching_cubes(
            ind_tsdf_grid, gradient_direction='descent', level=0)

        list_vertices.append(torch.from_numpy(
            vertices.copy()).to(tsdf_grid.device))
        list_faces.append(torch.from_numpy(
            faces.copy()).to(tsdf_grid.device))
        list_normals.append(torch.from_numpy(
            normals.copy()).to(tsdf_grid.device))

    res_meshes = Meshes(list_vertices, list_faces, verts_normals=list_normals)
    if resolution is not None:
        scale = Scale(resolution[0]/2, resolution[1]/2, resolution[2] /
                      2, device=res_meshes.device)
        new_verts = scale.transform_points(res_meshes.verts_padded())
        res_meshes = res_meshes.update_padded(new_verts)

    return res_meshes


class points_norm(nn.Module):
    def __init__(self, intercept: torch.Tensor = None,
                 scale: torch.Tensor = None,
                 isotropic: bool = False):
        super().__init__()
        self.intercept = intercept
        self.scale = scale
        self.isotropic = isotropic

    def normalize(self, batch_X: torch.Tensor):
        """
            Args:
                points: [B, N, 3]
                isotropic: bool

            Return:
                [B, N, 3]
        """

        points_min, points_max = batch_X.amin(
            dim=(0, 1)), batch_X.amax(dim=(0, 1))

        if self.intercept is None:
            self.intercept = points_min

        if self.scale is None:
            self.scale = points_max - points_min
            if self.isotropic:
                self.scale = self.scale.max()

        batch_X = 2 * (batch_X - self.intercept) / self.scale - 1

        return batch_X

    def denormalize(self, batch_X: torch.Tensor):
        """
            Args:
                points: [B, N, 3]

            Return:
                [B, N, 3]
        """
        assert self.intercept is not None
        assert self.scale is not None

        batch_X = 0.5 * self.scale * (1 + batch_X) + self.intercept

        return batch_X

    def forward(self, batch_X: torch.Tensor):
        pass


class TSDF:
    def __init__(self, resolution: Union[int, torch.Tensor],
                 sampling_count: int,
                 downsampling_count: int,
                 bbox: torch.Tensor,
                 isotropic: bool = False,
                 trunated_dist: float = 0.3):
        self.resolution = resolution  # int, [l, w, h]
        self.bbox = bbox  # [l, w, h]
        self.isotropic = isotropic
        self.sampling_count = sampling_count
        self.downsampling_count = downsampling_count
        self.trunated_dist = trunated_dist

        # self.points_norm = points_norm(
        #     intercept=torch.zeros(3), isotropic=self.isotropic)

    # def preprocess(self, batch_X: torch.Tensor):
    #     """
    #         Args:
    #             batch_X: [B, N, 3]

    #         Return:
    #             batch_X: [B, N, 3]
    #     """
    #     if self.bbox is not None:
    #         batch_X = bbox_normalize(batch_X, self.bbox)
    #     else:
    #         batch_X = self.points_norm.normalize(batch_X)

    #     return batch_X

    @staticmethod
    def broke_tensor(input_tensor: torch.Tensor):
        res_tuple = tuple()
        assert input_tensor.size().__len__() == 1
        for i in input_tensor:
            res_tuple += (i.item(), )
        return res_tuple

    def tsdf(self, batch_meshes: Meshes):
        """
            Args:
                batch_meshes: Meshes

            Return:

        """
        B = batch_meshes.__len__()

        dense_points, dense_normals = self.meshes_sampling(
            batch_meshes, self.sampling_count)

        downsampled_points, downsampled_normals = self.downsampling(dense_points, dense_normals,
                                                                    self.downsampling_count)

        meshgrid = self.generate_meshgrid(
            self.resolution, bbox=self.bbox).unsqueeze(0).expand(B, -1, -1)  # [B, l*w*h, 3]

        sdf_grid = self.sample_sdf(
            meshgrid, downsampled_points, downsampled_normals)

        # Convert sdf to tsdf
        tsdf_grid = sdf_grid.clamp(-self.trunated_dist, self.trunated_dist)

        # Recover the shape of tsdf grid
        tsdf_grid = self.reshape_tsdf(self.resolution, tsdf_grid)

        return tsdf_grid

        # denormalized_tsdf_grid = self.points_norm.denormalize(sdf_grid)

    def meshes_sampling(self, batch_meshes: Meshes,
                        sampling_count: int):
        """
            Args:
                batch_meshes: Meshes

            Return:
                points, normal: [B, N, 3]
        """
        points, normals = sample_points_from_meshes(
            batch_meshes,
            num_samples=sampling_count,
            return_normals=True
        )

        return points, normals

    def downsampling(self, points: torch.Tensor, normals: torch.Tensor,
                     downsampling_count: int):
        """
            Args:
                batch_meshes: Meshes

            Return:
                points, normal: [B, N, 3]
        """
        downsampled_points, downsampled_idx = sample_farthest_points(
            points,
            K=downsampling_count,
            random_start_point=True
        )
        downsampled_normals = knn_gather(
            normals, downsampled_idx.unsqueeze(2)).squeeze(2)

        return downsampled_points, downsampled_normals

    @staticmethod
    def reshape_tsdf(resolution: Union[int, torch.Tensor], tsdf: torch.Tensor):
        """
            Args:
                resolution: int (if isotropic)/ [3] (if anisotropic)

            Return:
                meshgrid: [l*w*h, 3]
        """
        if isinstance(resolution, torch.Tensor):
            l_res, w_res, h_res = __class__.broke_tensor(resolution)
        else:
            l_res, w_res, h_res = resolution, resolution, resolution

        return tsdf.view(-1, l_res, w_res, h_res)

    @staticmethod
    def generate_meshgrid(resolution: Union[int, torch.Tensor], bbox: torch.Tensor):
        """
            Args:
                resolution: int (if isotropic)/ [3] (if anisotropic)
                bbox: [l, w, h]

            Return:
                meshgrid: [l*w*h, 3]
        """
        l, w, h = bbox[0], bbox[1], bbox[2]
        if isinstance(resolution, torch.Tensor):
            l_res, w_res, h_res = __class__.broke_tensor(resolution)
        else:
            l_res, w_res, h_res = resolution, resolution, resolution

        meshgrid = torch.stack(torch.meshgrid(
            torch.linspace(-l/2, l/2, l_res),
            torch.linspace(-w/2, w/2, w_res),
            torch.linspace(-h/2, h/2, h_res),
            indexing="ij"
        ), dim=-1).view(-1, 3)

        return meshgrid

    @staticmethod
    def sample_sdf(coords: torch.Tensor, points: torch.Tensor, normals: torch.Tensor, K: int = 15, votes_threshold: float = 0.5):
        """
            Args:
                coords: [B, N, 3]
                points: [B, N, 3]
                normals: [B, N, 3]
                K: int
                votes_threshold: flaoat

            Return:
                points, normal: [B, N, 3]
        """
        knn_dists, knn_idx, knn_neighbors = knn_points(
            coords,
            points,
            return_nn=True,
            return_sorted=True,
            K=K
        )  # [B, N, K], [B, N, K], [B, N, K, 3]

        voxel_to_point = coords.unsqueeze(2) - knn_neighbors  # [B, N, K, 3]
        knn_normals = knn_gather(normals, knn_idx)  # [B, N, K, 3]

        cos_angles = torch.sum(
            voxel_to_point * knn_normals,
            dim=-1
        ) / (torch.norm(voxel_to_point, dim=-1) + 1e-6)  # [B, N, K]

        # Compute sign distance of each coordination
        outside_votes = (cos_angles > 0).float().mean(dim=-1)  # [B, N]
        inside_mask = outside_votes < votes_threshold

        closest_dists = knn_dists[:, :, 0]  # [B, N]

        sdf = torch.where(inside_mask, -closest_dists, closest_dists)  # [B, N]

        return sdf


if __name__ == "__main__":
    pass
