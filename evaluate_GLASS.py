import os
import glob
import numpy as np
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score



def split_composite_image(image_path):
    """Split the composite result image into original, reconstruction, and anomaly map."""
    img = imread(image_path, as_gray=True).astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    h, w = img.shape
    third = w // 3
    original = img[:, :third]
    # middle = img[:, third:2*third]  # not used here
    anomaly_map = img[:, 2 * third:]
    return original, anomaly_map


def normalize(img):
    """Normalize image to range [0, 1]."""
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)


def save_as_heatmap(img, save_path):
    """Save anomaly map as normalized heatmap."""
    img_norm = normalize(img)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_norm, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_anomaly_heatmaps(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    result_paths = sorted(glob.glob(os.path.join(results_dir, "*.png")))

    for path in tqdm(result_paths, desc="Saving heatmaps"):
        _, anomaly_map = split_composite_image(path)
        filename = os.path.splitext(os.path.basename(path))[0] + ".png"
        save_path = os.path.join(output_dir, filename)
        save_as_heatmap(anomaly_map, save_path)


def load_image(path, target_shape=None):
    img = imread(path, as_gray=True).astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    if target_shape and img.shape != target_shape:
        img = resize(img, target_shape, preserve_range=True, anti_aliasing=True)
    return img


def evaluate(result_dir, anomaly_dir, mask_dir, vis_dir, target_shape=(256, 256)):
    os.makedirs(vis_dir, exist_ok=True)
    original_dir = os.path.join(vis_dir, "original")
    os.makedirs(original_dir, exist_ok=True)

    result_paths = sorted(glob.glob(os.path.join(result_dir, "*.png")))
    anomaly_paths = sorted(glob.glob(os.path.join(anomaly_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    assert len(anomaly_paths) == len(mask_paths), "Mismatch between anomaly maps and ground truth masks"

    all_ssim = []
    all_mse = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_iou = []

    for i, (res_path, anom_path, mask_path) in enumerate(zip(result_paths, anomaly_paths, mask_paths)):
        original, _ = split_composite_image(res_path)
        anomaly_map = load_image(anom_path, target_shape)

        # Apply threshold to binarize the anomaly map
        threshold = 0.95 
        binary_anomaly = (anomaly_map > threshold).astype(np.uint8)

        mask = load_image(mask_path, target_shape)
        mask = (mask > 0.5).astype(np.uint8)

        if mask.shape != anomaly_map.shape:
            mask = resize(mask, anomaly_map.shape, preserve_range=True, anti_aliasing=True)
            mask = (mask > 0.5).astype(np.uint8)

        # Save original image
        plt.imsave(os.path.join(original_dir, f"original_{i:03}.png"), normalize(original), cmap='gray')

        # Evaluate SSIM and MSE
        curr_ssim = ssim(mask, binary_anomaly, data_range=1.0)
        curr_mse = mean_squared_error(mask, binary_anomaly)

        all_ssim.append(curr_ssim)
        all_mse.append(curr_mse)

        # Flatten for sklearn metrics
        y_true = mask.flatten()
        y_pred = binary_anomaly.flatten()

        all_precision.append(precision_score(y_true, y_pred, zero_division=0))
        all_recall.append(recall_score(y_true, y_pred, zero_division=0))
        all_f1.append(f1_score(y_true, y_pred, zero_division=0))
        all_iou.append(jaccard_score(y_true, y_pred, zero_division=0))

        # Save visualization
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].imshow(normalize(original), cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(binary_anomaly, cmap='jet')
        ax[1].set_title('Anomaly Map')
        ax[1].axis('off')

        ax[2].imshow(mask, cmap='gray')
        ax[2].set_title('Ground Truth Mask')
        ax[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"eval_{i:03}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Print results
    print("\n--- Evaluation Results ---")
    print(f"Total evaluated: {len(all_ssim)} images")
    print(f"Average SSIM:     {np.mean(all_ssim):.4f}")
    print(f"Average MSE:      {np.mean(all_mse):.4f}")
    print(f"Average Precision:{np.mean(all_precision):.4f}")
    print(f"Average Recall:   {np.mean(all_recall):.4f}")
    print(f"Average F1-Score: {np.mean(all_f1):.4f}")
    print(f"Average IoU:      {np.mean(all_iou):.4f}")

    
if __name__ == "__main__":

    save_anomaly_heatmaps(
        results_dir="chest_wAnomalyMask/",         # path to composite result images
        output_dir="evaluate/anomaly_maps"         # extracted anomaly maps
    )

    evaluate(
        result_dir="chest_wAnomalyMask/",          # full composite images
        anomaly_dir="evaluate/anomaly_maps",       # normalized anomaly maps
        mask_dir="anomaly_mask",                   # ground truth anomaly masks
        vis_dir="evaluate/compare_results"         # side-by-side outputs
    )
