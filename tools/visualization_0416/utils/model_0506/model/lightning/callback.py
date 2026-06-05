from lightning import Callback
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision
from einops import rearrange

class VisualizationCallback(Callback):
    def __init__(self, save_freq=2000, output_dir="visualizations"):
        self.save_freq = save_freq
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
    #     # Check if the current step is a multiple of save_freq
    #     if trainer.is_global_zero:
    #         global_step = trainer.global_step
    #         if global_step % self.save_freq == 0:
    #             # Perform your visualization logic here
    #             # Example: save a plot of the current input or output (you can replace it with any other visualization)
    #             self.save_visualization(trainer, model, global_step, batch)
    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        # Check if the current step is a multiple of save_freq
        if trainer.is_global_zero:
            global_step = trainer.global_step
            if global_step % self.save_freq == 0:
                # Perform your visualization logic here
                # Example: save a plot of the current input or output (you can replace it with any other visualization)
                self.save_visualization(trainer, model, global_step, batch)

    def save_visualization(self, trainer, model, global_step, batch):
        # Example visualization: Save a plot of a dummy tensor (replace with your actual data or outputs)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])  # Replace with actual data, such as outputs from the model
        ax.set_title(f"Visualization at Step {global_step}")

        # Save the plot to a file
        plt.savefig(f"{self.output_dir}/visualization_{global_step}.png")
        plt.close(fig)
        print(f"Saved visualization at step {global_step}")


class VisualizationVAECallback(VisualizationCallback):
    def __init__(self, save_freq=2000, output_dir="visualizations"):
        super().__init__(save_freq, output_dir)

    def save_visualization(self, trainer, model, global_step, batch):
        # Example visualization: Save a plot of a dummy tensor (replace with your actual data or outputs)
        model.eval()
        with torch.no_grad():
            x_pred, x_gt = model(batch)

        x_pred = x_pred.cpu()
        x_gt = x_gt.cpu()

        x_pred = torch.clamp(x_pred, min=0.0, max=1.0)
        x_gt = torch.clamp(x_gt, min=0.0, max=1.0)

        B = x_gt.shape[0]
        rows = int(np.ceil(np.sqrt(B)))
        cols = int(np.ceil(B / rows))

        gt_grid = torchvision.utils.make_grid(x_gt, nrow=rows)
        pred_grid = torchvision.utils.make_grid(x_pred, nrow=rows)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(gt_grid.permute(1, 2, 0))
        axes[0].axis('off')
        # axes[0].set_title('Ground Truth')

        axes[1].imshow(pred_grid.permute(1, 2, 0))
        axes[1].axis('off')
        # axes[1].set_title('Prediction')

        plt.tight_layout()
        plt.show()
        plt.savefig(f"{self.output_dir}/image_grid_{global_step}.png")
        plt.close()

        # import pdb; pdb.set_trace()


class Visualization_HeadAnimator_Callback(VisualizationCallback):
    def __init__(self, save_freq=2000, output_dir="visualizations"):
        super().__init__(save_freq, output_dir)

    def save_visualization(self, trainer, model, global_step, batch):
        # Example visualization: Save a plot of a dummy tensor (replace with your actual data or outputs)

        masked_target_vid = batch['pixel_values_vid'] # this is a video batch: [B, T, C, H, W]
        masked_ref_img = batch['pixel_values_ref_img']

        ref_img_original = batch['ref_img_original']
        target_vid_original = batch['pixel_values_vid_original']

        # construct ref-tgt pairs
        masked_ref_img = masked_ref_img[:,None].repeat(1, masked_target_vid.size(1), 1, 1, 1)
        masked_ref_img = rearrange(masked_ref_img, "b t c h w -> (b t) c h w")
        masked_target_vid = rearrange(masked_target_vid, "b t c h w -> (b t) c h w")

        ref_img_original = ref_img_original[:,None].repeat(1, target_vid_original.size(1), 1, 1, 1)
        ref_img_original = rearrange(ref_img_original, "b t c h w -> (b t) c h w")
        target_vid_original = rearrange(target_vid_original, "b t c h w -> (b t) c h w")

        with torch.no_grad():
            # get reconstructed image
            model_out = model.forward(ref_img_original, target_vid_original, masked_ref_img, masked_target_vid)
            x_pred = model_out['recon_img']
        x_gt = target_vid_original

        x_pred = x_pred.cpu()
        x_gt = x_gt.cpu()
        x_ref = ref_img_original.cpu()

        if x_gt.min() < -0.5:
            x_gt = (x_gt + 1) / 2
            x_pred = (x_pred + 1) / 2
            x_ref = (x_ref + 1) / 2

        x_pred = torch.clamp(x_pred, min=0.0, max=1.0)
        x_gt = torch.clamp(x_gt, min=0.0, max=1.0)
        x_ref = torch.clamp(x_ref, min=0.0, max=1.0)

        B = x_gt.shape[0]
        rows = int(np.ceil(np.sqrt(B)))
        cols = int(np.ceil(B / rows))

        ref_grid = torchvision.utils.make_grid(x_ref, nrow=rows)
        gt_grid = torchvision.utils.make_grid(x_gt, nrow=rows)
        pred_grid = torchvision.utils.make_grid(x_pred, nrow=rows)

        diff = (x_pred-x_gt).abs()
        diff_grid = torchvision.utils.make_grid(diff, nrow=rows)

        fig, axes = plt.subplots(1, 4, figsize=(12, 6))
        axes[0].imshow(ref_grid.permute(1, 2, 0))
        axes[0].axis('off')

        axes[1].imshow(gt_grid.permute(1, 2, 0))
        axes[1].axis('off')

        axes[2].imshow(pred_grid.permute(1, 2, 0))
        axes[2].axis('off')

        axes[3].imshow(diff_grid.permute(1, 2, 0), cmap='jet')
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig(f"{self.output_dir}/image_grid_{global_step}.png")
        plt.close()
