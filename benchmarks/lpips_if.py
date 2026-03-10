import lpips
import torch


class LPIPSImageFidelity:
    """
    Wrapper around the LPIPS metric that returns an image fidelity score
    (1 - LPIPS).  The model is created once at construction time
    and moved to the requested device.

    Parameters
    ----------
    net : str, optional
        Backbone network for LPIPS. Options are ``'alex'``, ``'vgg'`` or
        ``'squeeze'``. Default is ``'alex'``.
    device : torch.device or str, optional
        Device on which to run the model (e.g. ``'cpu'`` or ``'cuda:0'``).
        If omitted, the function will use ``torch.cuda.current_device()``
        when a GPU is available, otherwise ``'cpu'``.
    """

    def __init__(self, net: str = "alex", device: torch.device | str | None = None):
        # Choose a sensible default device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        # Build the LPIPS model and move it to the target device
        self.lpips_fn = lpips.LPIPS(net=net).to(device)
        self.device = device

    def __call__(self, original: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
        return self.image_fidelity(original, modified)

    def image_fidelity(self, original: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
        """
        Compute a fidelity score between two images.

        The LPIPS distance lies in ``[0, 1]`` (higher = more perceptual difference).
        This method returns ``1 - distance`` so that larger values indicate higher
        similarity.

        Both inputs must be PyTorch tensors of shape ``(N, C, H, W)`` with values
        in the range ``[-1, 1]`` (the range expected by the LPIPS model).  They
        will be automatically moved to the same device as the model.

        Parameters
        ----------
        original : torch.Tensor
            Reference image.
        modified : torch.Tensor
            Image to compare against the reference.

        Returns
        -------
        torch.Tensor
            Fidelity score (scalar per batch element) in the range ``[0, 1]``.
        """
        original = original.to(self.device)
        modified = modified.to(self.device)

        # LPIPS returns a tensor of shape (N, 1, 1, 1); squeeze to (N,)
        lpips_distance = self.lpips_fn(original, modified).squeeze()

        # Convert distance to fidelity
        fidelity = 1.0 - lpips_distance
        return fidelity
