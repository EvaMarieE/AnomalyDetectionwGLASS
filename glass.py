from loss import FocalLoss
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker

import numpy as np
import pandas as pd
import torch.nn.functional as F

import logging
import os
import math
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1

class GLASS(torch.nn.Module):
    def __init__(self, device):
        super(GLASS, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=True,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            rand_aug=1,
            downsampling=8,
            scale=0,
            batch_size=1,
            **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device
        
        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator
        
        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
        
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(
                self.forward_modules["feature_aggregator"].backbone.parameters(),
                lr,
                weight_decay=1e-5,
            )
        
        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)
        
        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin
        
        self.c = torch.tensor(0)
        self.c_ = torch.tensor(0)
        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()
        
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        
        if not evaluation and self.train_backbone:
            print("Training backbone")
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]    
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]
    
        for i in range(1, len(patch_features)):
            _features = patch_features[i]
            patch_dims = patch_shapes[i]
    
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, 4, 5, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            patch_features[i] = _features

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes


    def trainer(self, training_data, val_data, name, writer=None):            
        state_dict = {}
        
        ckpt_path_best = os.path.join(self.ckpt_dir, 'ckpt_best_0.pth')
    
        # Delete old checkpoints to force retraining
        if os.path.exists(ckpt_path_best):
            os.remove(ckpt_path_best)
            
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")

    
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})

        #self.distribution = training_data.dataset.distribution
        dataset = training_data.dataset
        self.distribution = dataset.distribution

        xlsx_path = './datasets/excel/' + name.split('_')[0] + '_distribution.xlsx'
        try:
            if self.distribution == 1:  # rejudge by image-level spectrogram analysis
                self.distribution = 1
                self.svd = 1
            elif self.distribution == 2:  # manifold
                self.distribution = 0
                self.svd = 0
            elif self.distribution == 3:  # hypersphere
                self.distribution = 0
                self.svd = 1
            elif self.distribution == 4:  # opposite choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = 1 - df.loc[df['Class'] == name, 'Distribution'].values[0]
            else:  # choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = df.loc[df['Class'] == name, 'Distribution'].values[0]
        except:
            self.distribution = 1
            self.svd = 1

        # judge by image-level spectrogram analysis
        if self.distribution == 1:
            self.forward_modules.eval()
            with torch.no_grad():
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    batch_mean = torch.mean(img, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            avg_img = utils.torch_format_2_numpy_img(self.c.detach().cpu().numpy())
            self.svd = utils.distribution_judge(avg_img, name)
            os.makedirs(f'./results/judge/avg/{self.svd}', exist_ok=True)
            cv2.imwrite(f'./results/judge/avg/{self.svd}/{name}.png', avg_img)
            return self.svd

        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        best_record = None
        for i_epoch in pbar:
            self.forward_modules.eval()
            with torch.no_grad():  # compute center
                for i, data in enumerate(training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    if self.pre_proj > 0:
                        outputs = self.pre_projection(self._embed(img, evaluation=False)[0])
                        outputs = outputs[0] if len(outputs) == 2 else outputs
                    else:
                        outputs = self._embed(img, evaluation=False)[0]
                    outputs = outputs[0] if len(outputs) == 2 else outputs
                    outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1])

                    batch_mean = torch.mean(outputs, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(training_data)

            pbar_str, pt, pf = self._train_discriminator(training_data, i_epoch, pbar, pbar_str1)
            update_state_dict()

            if (i_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, labels_gt, masks_gt, anomaly_masks = self.predict(val_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                         labels_gt, masks_gt, name)

                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("i-ap", image_ap, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                self.logger.logger.add_scalar("p-ap", pixel_ap, i_epoch)
                self.logger.logger.add_scalar("p-pro", pixel_pro, i_epoch)

                # TensorBoard logging
                if writer:
                    writer.add_scalar('AUROC/Image', image_auroc, i_epoch)
                    writer.add_scalar('AP/Image', image_ap, i_epoch)
                    writer.add_scalar('AUROC/Pixel', pixel_auroc, i_epoch)
                    writer.add_scalar('AP/Pixel', pixel_ap, i_epoch)
                    writer.add_scalar('PRO/Pixel', pixel_pro, i_epoch)

                eval_path = './results/eval/' + name + '/'
                train_path = './results/training/' + name + '/'
                if best_record is None:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                elif image_auroc + pixel_auroc > best_record[0] + best_record[2]:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    os.remove(ckpt_path_best)
                    ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                pbar_str1 = f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})" \
                            f" IAP:{round(image_ap * 100, 2)}({round(best_record[1] * 100, 2)})" \
                            f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_record[2] * 100, 2)})" \
                            f" PAP:{round(pixel_ap * 100, 2)}({round(best_record[3] * 100, 2)})" \
                            f" PRO:{round(pixel_pro * 100, 2)}({round(best_record[4] * 100, 2)})" \
                            f" E:{i_epoch}({best_record[-1]})"
                pbar_str += pbar_str1
                pbar.set_description_str(pbar_str)

            torch.save(state_dict, ckpt_path_save)

        return best_record

    def _train_discriminator(self, input_data, i_epoch, pbar, pbar_str1):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        
        pt, pf = 0, 0
        all_loss, all_p_true, all_p_fake, all_r_t, all_r_g, all_r_f = [], [], [], [], [], []
        sample_num = 0
        for i_iter, data_item in enumerate(input_data):
            #self.backbone_opt.zero_grad()
            
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()
    
            aug = data_item["aug"].to(torch.float).to(self.device)
            img = data_item["image"].to(torch.float).to(self.device)
            
            # Fix 1: Ensure proper gradient tracking
            if self.pre_proj > 0:
                # Get embeddings first
                fake_embed = self._embed(aug, evaluation=False)[0]
                true_embed = self._embed(img, evaluation=False)[0]
                
                # Apply projection and ensure gradients are tracked
                fake_feats = self.pre_projection(fake_embed)
                fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
                fake_feats.requires_grad_(True)  # Ensure gradients are tracked
                
                true_feats = self.pre_projection(true_embed)
                true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
                true_feats.requires_grad_(True)  # Ensure gradients are tracked
            else:
                fake_feats = self._embed(aug, evaluation=False)[0]
                fake_feats.requires_grad_(True)
                true_feats = self._embed(img, evaluation=False)[0]
                true_feats.requires_grad_(True)
    
            # Ensure gaus_feats maintains gradient connection
            noise = torch.normal(0, self.noise, true_feats.shape).to(self.device)
            gaus_feats = true_feats + noise
            gaus_feats.requires_grad_(True)
            
            center = self.c.repeat(img.shape[0], 1, 1).reshape(-1, self.c.shape[-1])
            
            # Distance metrics for statistics
            true_points = torch.cat([fake_feats, true_feats], dim=0)
            c_t_points = torch.cat([center, center], dim=0)
            
            dist_t = torch.norm(true_points - c_t_points, dim=1)
            r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)
            
            dist_f = torch.norm(fake_feats - center, dim=1)
            r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)
            
            # Initialize r_g in case it's not set later (e.g. early break)
            r_g = torch.tensor([0.0]).to(self.device)
            
            # === Adversarial mining steps ===
            for step in range(self.step + 1):
                scores = self.discriminator(torch.cat([true_feats, gaus_feats]))
                true_scores = scores[:len(true_feats)]
                gaus_scores = scores[len(true_feats):]
            
                true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
                gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
                bce_loss = true_loss + gaus_loss
            
                if step == self.step:
                    break
                elif self.mining == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                    break
            
                try:
                    grad = torch.autograd.grad(gaus_loss, [gaus_feats], retain_graph=True, create_graph=False)[0]
                except RuntimeError as e:
                    print(f"Gradient computation failed at step {step}: {e}")
                    break
            
                grad_norm = torch.norm(grad, dim=1).view(-1, 1)
                grad_normalized = grad / (grad_norm + 1e-10)
            
                with torch.no_grad():
                    gaus_feats_new = gaus_feats + 0.001 * grad_normalized
            
                gaus_feats = gaus_feats_new.detach().requires_grad_(True)
            
            # === Final pass ===
            final_scores = self.discriminator(torch.cat([true_feats, gaus_feats]))
            true_scores = final_scores[:len(true_feats)]
            gaus_scores = final_scores[len(true_feats):]
            
            true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
            loss = true_loss + gaus_loss
            
            loss.backward()
            if self.train_backbone:
                self.backbone_opt.step()
            if self.pre_proj > 0:
                self.proj_opt.step()
            self.dsc_opt.step()
            
            # Logging gradient requirement and loss
            outputs, _ = self._embed(img, evaluation=False)
            #print("Backbone outputs require grad:", outputs.requires_grad)
            #print("Loss value:", loss.item())
            
            # === Pixel-level score statistics ===
            mask_s_gt = data_item.get("mask_s", None)
            if mask_s_gt is not None:
                mask_s_gt = mask_s_gt.to(self.device)
            
            fake_scores = self.discriminator(fake_feats)
            
            if mask_s_gt is not None:
                pix_true = torch.cat([
                    fake_scores.detach() * (1 - mask_s_gt),
                    true_scores.detach()
                ])
                pix_fake = torch.cat([
                    fake_scores.detach() * mask_s_gt,
                    gaus_scores.detach()
                ])
                p_true = ((pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
                p_fake = (pix_fake >= self.dsc_margin).sum() / ((mask_s_gt == 1).sum() + gaus_scores.shape[0])
            else:
                p_true = 0.0
                p_fake = 0.0
            
            # === Logging ===
            self.logger.logger.add_scalar("p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar("p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar("r_t", r_t, self.logger.g_iter)
            self.logger.logger.add_scalar("r_g", r_g, self.logger.g_iter)
            self.logger.logger.add_scalar("r_f", r_f, self.logger.g_iter)
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            self.logger.step()
            
            # Collect stats
            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true if isinstance(p_true, float) else p_true.cpu().item())
            all_p_fake.append(p_fake if isinstance(p_fake, float) else p_fake.cpu().item())
            all_r_t.append(r_t.cpu().item())
            all_r_g.append(r_g.cpu().item())
            all_r_f.append(r_f.cpu().item())
            
            sample_num += img.shape[0]
            
            pbar_str = f"epoch:{i_epoch} loss:{np.mean(all_loss):.2e}"
            pbar_str += f" pt:{np.mean(all_p_true) * 100:.2f}"
            pbar_str += f" pf:{np.mean(all_p_fake) * 100:.2f}"
            pbar_str += f" rt:{np.mean(all_r_t):.2f}"
            pbar_str += f" rg:{np.mean(all_r_g):.2f}"
            pbar_str += f" rf:{np.mean(all_r_f):.2f}"
            pbar_str += f" svd:{self.svd}"
            pbar_str += f" sample:{sample_num}"
            pbar_str += pbar_str1
            pbar.set_description_str(pbar_str)
            
            if sample_num > self.limit:
                break
    
        return pbar_str, all_p_true, all_p_fake


    def tester(self, test_data, name):
        
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        
        if len(ckpt_path) != 0:
            print("Skipping checkpoint loading for fresh run")
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)
    
            images, scores, segmentations, labels_gt, masks_gt, anomaly_masks = self.predict(test_data)
    
            # Prefer anomaly mask if available and at least one exists
            if anomaly_masks is not None and any(m is not None for m in anomaly_masks):
                masks_eval = anomaly_masks
            else:
                masks_eval = masks_gt
    
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(
                images, scores, segmentations,
                labels_gt, masks_eval,
                name, path='eval',
                anomaly_masks=anomaly_masks # for visualization
            )
    
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = 0., 0., 0., 0., 0., -1.
            LOGGER.info("No ckpt file found!")
    
        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch
    
    

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training', anomaly_masks=None):
        scores = np.squeeze(np.array(scores))
    
        # kip image-level metrics if only one class in labels_gt
        if len(set(labels_gt)) < 2:
            print("[WARNING] Skipping image-level metrics — only one class present in labels_gt.")
            image_auroc = 0.
            image_ap = 0.
        else:
            image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
            image_auroc = image_scores["auroc"]
            image_ap = image_scores["ap"]
    
        # --- Pixel-level metrics (always computed if masks_gt is available) ---
        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), segmentations)
                except Exception:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            print("[WARNING] Skipping pixel-level metrics — no ground-truth masks found.") 
            pixel_auroc = pixel_ap = pixel_pro = -1.
    
        # ----- Visualization -----
        defects = np.array(images)
        targets = np.array(masks_gt)
    
        full_path = os.path.join('./results', path, name)
        utils.del_remake_dir(full_path, del_flag=False)
    
        for i in range(len(defects)):
            defect = utils.torch_format_2_numpy_img(defects[i])
            target = utils.torch_format_2_numpy_img(targets[i])

            # Get raw anomaly map and lung mask (as numpy)
            mask = segmentations[i]  # shape: H x W, values in [0, 1]
            lung_mask = np.array(masks_gt[i])  # convert list to NumPy array

            # If it’s shape (1, H, W), squeeze the first dim
            if lung_mask.ndim == 3 and lung_mask.shape[0] == 1:
                lung_mask = np.squeeze(lung_mask, axis=0)  # (H, W)
            
            # Restrict anomaly detection to lung region only
            mask = mask * lung_mask
        
            # If anomalies appear blue instead of red → invert the mask
            # mask = 1.0 - mask
        
            # Resize and colorize
            mask = cv2.resize(mask, (defect.shape[1], defect.shape[0]))
            mask = (mask * 255).astype('uint8')
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
            # Include anomaly mask if present
            if anomaly_masks is not None and anomaly_masks[i] is not None:
                anomaly_mask = utils.torch_format_2_numpy_img(anomaly_masks[i])
                anomaly_mask = cv2.resize(anomaly_mask, (defect.shape[1], defect.shape[0]))
                anomaly_mask = (anomaly_mask * 255).astype('uint8')
                anomaly_mask_color = cv2.applyColorMap(anomaly_mask, cv2.COLORMAP_JET)
        
                img_up = np.hstack([defect, target, mask_color, anomaly_mask_color])
                img_up = cv2.resize(img_up, (256 * 4, 256))
            else:
                img_up = np.hstack([defect, target, mask_color])
                img_up = cv2.resize(img_up, (256 * 3, 256))
        
            cv2.imwrite(os.path.join(full_path, str(i + 1).zfill(3) + '.png'), img_up)
    
        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro



    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        anomaly_masks_all = []  # Collect anomaly masks from batches if needed

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            
            for data in data_iterator:
                
                if isinstance(data, dict):
                    
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy().tolist())
                        
                    image = data["image"]
                    images.extend(image.numpy().tolist())
                    img_paths.extend(data["image_path"])

                    # pass anomaly_mask if present in data (optional)
                    anomaly_mask = data.get("anomaly_mask", None)
                else:
                    anomaly_mask = None
                
                _scores, _masks, _anom_masks = self._predict(image, anomaly_mask=anomaly_mask)
                for score, mask, amask in zip(_scores, _masks, _anom_masks):
                    scores.append(score)
                    masks.append(mask)
                    anomaly_masks_all.append(amask)  # collect anomaly masks if needed
    
        return images, scores, masks, labels_gt, masks_gt, anomaly_masks_all


        
    def _predict(self, img, anomaly_mask=None):
        """Infer score and mask for a batch of images."""
        
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()
    
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
    
        with torch.no_grad():
            patch_features, patch_shapes = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features
    
            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)
    
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()
    
            # Prepare anomaly masks if provided
            if anomaly_mask is not None:
                anomaly_masks = [m.cpu() if m is not None else None for m in anomaly_mask]
            else:
                anomaly_masks = [None] * img.shape[0]
        torch.cuda.empty_cache()
    
        return list(image_scores), list(masks), anomaly_masks
    
    
