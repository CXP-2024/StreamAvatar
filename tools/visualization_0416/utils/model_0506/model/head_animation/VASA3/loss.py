import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


def compute_multiscale_vgg_loss(pred: torch.Tensor, gt: torch.Tensor, vgg_net: torch.nn.Module):
    loss = 0
    downscale_factors = [1, 2, 4, 8]
    for downscale_factor in downscale_factors:
        if downscale_factor == 1:
            level_pred = pred
            level_gt = gt
        else:
            level_pred = torch.nn.functional.interpolate(pred, scale_factor=1 / downscale_factor, mode="bilinear")
            level_gt = torch.nn.functional.interpolate(gt, scale_factor=1 / downscale_factor, mode="bilinear")

        loss = loss + vgg_net(level_pred, level_gt, normalize=True)

    return loss / len(downscale_factors)


def crop_and_resize(images: torch.Tensor, bboxes: torch.Tensor, size: int):
    batch_size = images.shape[0]

    output_images = []
    for i in range(batch_size):
        bbox = bboxes[i]
        output_images.append(
            F.interpolate(
                images[i:i+1, :, bbox[0, 1]:bbox[1, 1], bbox[0, 0]:bbox[1, 0]],
                size=(size, size),
                mode="area",
            )
        )

    return torch.cat(output_images, dim=0)


def compute_face_embedding_loss(face_bboxes: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor, face_net: torch.nn.Module):
    # https://github.com/timesler/facenet-pytorch?tab=readme-ov-file#pretrained-models

    # TODO: If this proves to be useful, we can precompute these embeddings.
    with torch.no_grad():
        gt_embedding = face_net(crop_and_resize(gt, face_bboxes, 160) * 2 - 1)

    return torch.abs(face_net(crop_and_resize(pred, face_bboxes, 160) * 2 - 1) - gt_embedding)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()

        self.vgg_net = lpips.LPIPS(net="vgg").eval()
        for param in self.vgg_net.parameters():
            param.requires_grad = False

    def forward(self, img_recon, img_real, facial_mask=None):

        vgg_loss = compute_multiscale_vgg_loss(img_recon, img_real, self.vgg_net)

        return vgg_loss, None
