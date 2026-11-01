
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Reusable Grad-CAM implementation for DefectVision AI.
    Works with any CNN by hooking a target conv layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self.hook_handles.append(target_layer.register_forward_hook(self._save_activation))
        self.hook_handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[:, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]      # C,H,W
        activations = self.activations[0]  # C,H,W

        weights = gradients.mean(dim=(1, 2))  # C
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        probs = F.softmax(output, dim=1)[0]
        confidence = probs[class_idx].item()

        return cam, class_idx, confidence

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


def overlay_heatmap(image_rgb_uint8, cam, alpha=0.45):
    """image_rgb_uint8: HxWx3 uint8 RGB image. cam: HxW float in [0,1]."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * image_rgb_uint8).astype(np.uint8)
    return overlay


def defect_area_percentage(cam, threshold=0.5):
    binary = (cam >= threshold).astype(np.uint8)
    return 100.0 * binary.sum() / binary.size


def cam_mask_iou(cam, gt_mask, threshold=0.5):
    """gt_mask: HxW, values >0 = defect. cam resized to same shape as gt_mask expected."""
    cam_bin = (cam >= threshold).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)
    intersection = np.logical_and(cam_bin, gt_bin).sum()
    union = np.logical_or(cam_bin, gt_bin).sum()
    if union == 0:
        return 0.0
    return intersection / union
