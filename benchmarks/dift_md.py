import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dift_sd import SDFeaturizer


class DIFTMeanDistance:
    """
    Helper that computes the Mean Distance (MD) metric.  The class holds a single
    SDFeaturizer instance and the  device on which all tensors are placed.

    Parameters
    ----------
    model_name : str, optional
        Identifier of the StableDiffusion checkpoint to load. The default
        matches the example you gave.
    device : torch.device or str, optional
        Target device. If omitted the class automatically selects CUDA when
        available, otherwise CPU.
    """

    def __init__(self, model_name: str = "sd2-community/stable-diffusion-2-1",
                 device: torch.device | str | None = None):
        # Choose device
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.dift = SDFeaturizer(model_name)
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def __call__(
        self,
        source: torch.Tensor,
        edited: torch.Tensor,
        handle_points: list[tuple[int, int]],
        target_points: list[tuple[int, int]],
        prompt: str,
        t: int = 261,
        up_ft_index: int = 1,
        ensemble_size: int = 8,
    ) -> float:
        return self.mean_distance(
            source,
            edited,
            handle_points,
            target_points,
            prompt,
            t=t,
            up_ft_index=up_ft_index,
            ensemble_size=ensemble_size,
        )

    def mean_distance(
        self,
        source: torch.Tensor,
        edited: torch.Tensor,
        handle_points: list[tuple[int, int]],
        target_points: list[tuple[int, int]],
        prompt: str,
        t: int = 261,
        up_ft_index: int = 1,
        ensemble_size: int = 8,
    ) -> float:
        """
        Compute the Mean Distance (MD) between handle points in ``source`` and
        their corresponding locations in ``edited``.

        Parameters
        ----------
        source : torch.Tensor
            Tensor of shape ``(C, H, W)`` (or ``(1, C, H, W)``) representing the
            original image.  Values are assumed to be in the range ``[-1, 1]``.
        edited : torch.Tensor
            Tensor of the same shape as ``source`` representing the edited
            image.
        handle_points : list[tuple[int, int]]
            Pixel coordinates ``(row, col)`` in the source image that act as
            handles.
        target_points : list[tuple[int, int]]
            Desired pixel coordinates after editing; must be the same length
            as ``handle_points``.
        prompt : str
            Text prompt that was used to generate the source image.
        t, up_ft_index, ensemble_size : int, optional
            Forward-pass hyperparameters passed to ``SDFeaturizer.forward``.

        Returns
        -------
        float
            The mean Euclidean distance (in pixel units) between each target
            point and the location found by DIFT for its corresponding handle.
        """
        if len(handle_points) != len(target_points):
            raise ValueError("handle_points and target_points must have the same length")

        # Ensure batch dimension exists (idft expects NCHW)
        if source.dim() == 3:
            source = source.unsqueeze(0)
        if edited.dim() == 3:
            edited = edited.unsqueeze(0)

        source = source.to(self.device)
        edited = edited.to(self.device)

        _, C, H, W = source.shape

        # Extract diffusion features for both images
        ft_source = self.dift.forward(
            source,
            prompt=prompt,
            t=t,
            up_ft_index=up_ft_index,
            ensemble_size=ensemble_size,
        )
        ft_source = F.interpolate(ft_source, (H, W), mode="bilinear")

        ft_edited = self.dift.forward(
            edited,
            prompt=prompt,
            t=t,
            up_ft_index=up_ft_index,
            ensemble_size=ensemble_size,
        )
        ft_edited = F.interpolate(ft_edited, (H, W), mode="bilinear")

        # For each handle point locate the most similar location in the
        # edited feature map (cosine similarity across the channel dim)
        distances = []
        num_channels = ft_source.size(1)

        for hp, tp in zip(handle_points, target_points):
            # Feature vector at the handle point (shape: 1 × C × 1 × 1)
            src_vec = ft_source[0, :, hp[0], hp[1]].view(1, num_channels, 1, 1)

            # Cosine similarity map over the whole edited feature map
            cos_map = self.cos_sim(src_vec, ft_edited).cpu().numpy()[0]  # (H, W)

            # Position of maximal similarity → final handle point after edit
            max_r, max_c = np.unravel_index(cos_map.argmax(), cos_map.shape)

            # Euclidean distance to the target point (pixel space)
            dist = torch.tensor([tp[0] - max_r, tp[1] - max_c], dtype=torch.float32).norm()
            distances.append(dist.item())

        return float(np.mean(distances))