import torch
import numpy as np
import utils
import os
import time
import torch.nn.functional as F
from PIL import Image


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=False)
            self.diffusion.model.eval()
        else:
            print('Pre-trained model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):

                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                img_h_64 = int(64 * np.ceil(h / 64.0))
                img_w_64 = int(64 * np.ceil(w / 64.0))
                x_cond = F.pad(x_cond, (0, img_w_64 - w, 0, img_h_64 - h), 'reflect')

                t1 = time.time()
                pred_x = self.diffusion.model(torch.cat((x_cond, x_cond),
                                                        dim=1))["pred_x"][:, :, :h, :w]
                t2 = time.time()

                # === Align to GT true size ===
                try:
                    gt_path = y[0] if isinstance(y, (list, tuple)) else y
                    with Image.open(gt_path) as _im:
                        _w, _h = _im.size
                    pred_x = pred_x[:, :, :_h, :_w]
                except Exception:
                    pass

                # === Handle y as a path and get correct filename ===
                if isinstance(y, (list, tuple)):
                    # Handle cases where y is a list or tuple and ensure we extract just the path string
                    save_name = y[0]
                else:
                    save_name = y
                
                # Ensure save_name is a proper string without any trailing characters
                save_name = os.path.basename(str(save_name).strip().replace("']","").replace("[",""))

                # Debug: print paths to check for any issues
                print(f"save_name before saving: {save_name}")

                # Save the image
                save_path = os.path.join(image_folder, save_name)
                print(f"save_path: {save_path}")  # Debug: print final path to check

                utils.logging.save_image(pred_x, save_path)

                print(f"processing image {save_name}, time={t2 - t1:.4f}s")
