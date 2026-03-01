"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for interpretability.
Works with any model that has a single target convolutional (or spatial) layer.
"""
import torch
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM heatmap and overlay for a given model and target layer."""

    def __init__(self, model, target_layer):
        """
        Args:
            model: nn.Module in eval mode.
            target_layer: nn.Module whose output we use (e.g. model.layer3 for ResNet).
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = None
        self._backward_handle = None

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _register_hooks(self):
        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _remove_hooks(self):
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def __call__(self, x, class_idx=None):
        """
        Compute Grad-CAM for a single image or batch (uses first sample for batch).

        Args:
            x: Input tensor [B, C, H, W]. Should require_grad if model requires it.
            class_idx: Class index to backprop from. If None, uses argmax of prediction.

        Returns:
            heatmap: [H, W] numpy in [0, 1] (spatial size of target layer).
            cam_resized: [input_H, input_W] numpy in [0, 1] (upsampled to input size).
        """
        self.model.eval()
        self.activations = None
        self.gradients = None
        self._register_hooks()

        try:
            out = self.model(x)
            if class_idx is None:
                class_idx = out.argmax(dim=1).item() if out.dim() == 2 else out[0].argmax().item()
            score = out[0, class_idx] if out.dim() == 2 else out[0, class_idx]
            self.model.zero_grad()
            score.backward(retain_graph=False)

            A = self.activations[0]
            G = self.gradients[0]
            weights = G.mean(dim=(1, 2))
            cam = (weights[:, None, None] * A).sum(dim=0)
            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            else:
                cam = cam * 0

            h, w = x.shape[2], x.shape[3]
            cam_resized = (
                F.interpolate(
                    torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).to(x.device),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            if cam_resized.max() > cam_resized.min():
                cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())

            return cam, cam_resized
        finally:
            self._remove_hooks()
