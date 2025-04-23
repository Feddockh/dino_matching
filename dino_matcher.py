#!/usr/bin/env python3
import random
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision import transforms
import operator


IMG1_PATH         = "rivendale_dataset/firefly_left/images/1739373919_100068096.png"
IMG2_PATH         = "rivendale_dataset/ximea/images/1739373919_100068096.png"
# IMG1_PATH         = "firefly_left.png"
# IMG2_PATH         = "675.png"
# IMG1_PATH         = "dog1.jpg"
# IMG2_PATH         = "dog2.jpg"
DINOV2_REPO       = "facebookresearch/dinov2"
MODEL_NAME        = "dinov2_vitb14"
SMALLER_EDGE      = 1024             # resize shorter edge before patching
HALF_PRECISION    = False           # True if your GPU likes fp16
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
# KEEP_FRACTION     = 0.005            # fraction of matches to draw
RATIO_THRESH      = 0.95     # Lowe ratio threshold (smaller = stricter)
TOP_MATCHES       = 300     # how many of the best matches to draw


class Dinov2Matcher:
    def __init__(
        self,
        repo_root: str | Path,
        model_name: str = MODEL_NAME,
        smaller_edge: int = SMALLER_EDGE,
        half_precision: bool = HALF_PRECISION,
        device: str = DEVICE,
    ):
        repo_root = str(repo_root)
        self.device = device
        self.half = half_precision
        self.model = torch.hub.load(repo_root, model_name)
        if self.half:
            self.model = self.model.half()
        self.model.eval().to(device)

        self.patch = self.model.patch_size  # e.g. 14 for ViT-B/14

        self.tf = transforms.Compose(
            [
                transforms.Resize(
                    smaller_edge,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def _prepare_img(self, rgb: np.ndarray):
        img_pil = Image.fromarray(rgb)
        tens = self.tf(img_pil)  # C×H×W

        # crop so (H, W) are multiples of patch size
        h, w = tens.shape[1:]
        h_crop, w_crop = h - h % self.patch, w - w % self.patch
        tens = tens[:, :h_crop, :w_crop]

        grid_h, grid_w = h_crop // self.patch, w_crop // self.patch
        scale = img_pil.width / w_crop  # crop→orig scale

        return tens, (grid_h, grid_w), scale

    def _tokens(self, tens: torch.Tensor):
        with torch.inference_mode():
            batch = tens.unsqueeze(0).to(self.device)
            if self.half:
                batch = batch.half()
            feats = self.model.get_intermediate_layers(batch)[0].squeeze(0)
        return feats.cpu().numpy()  # [N_patches, dim]

    def _grid_idx_to_xy(self, idx: int, grid, scale: float):
        gy, gx = grid
        row, col = divmod(idx, gx)
        x = (col + 0.5) * self.patch * scale
        y = (row + 0.5) * self.patch * scale
        return x, y

    def extract(self, rgb: np.ndarray):
        tens, grid, scale = self._prepare_img(rgb)
        feats = self._tokens(tens)
        return feats, grid, scale

def load_img_pair(img1_path: str, img2_path: str):
    img1_bgr = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2_bgr = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    if img1_bgr is None or img2_bgr is None:
        raise FileNotFoundError("Could not load one of the input images.")
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
    return img1_rgb, img2_rgb

def gaussian_blur(img, ksize=5):
    if ksize % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    kernel = cv2.getGaussianKernel(ksize, -1)
    kernel = np.outer(kernel, kernel)
    blurred = cv2.filter2D(img, -1, kernel)
    return blurred

def invert_colors(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.bitwise_not(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def gamma_correction(img, gamma=2.0):
    # assume img is BGR; convert to float [0,1]
    imgf = img.astype(np.float32) / 255.0
    # apply gamma
    corrected = np.power(imgf, gamma)
    # back to uint8
    return (corrected * 255).clip(0,255).astype(np.uint8)

def canny_edges(img, low=100, high=200, dilate=True):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high, apertureSize=3, L2gradient=True)
    if dilate:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    edge_rgb = np.stack([edges]*3, axis=-1)
    return edge_rgb

def match(matcher, img1_rgb, img2_rgb):
    feats1, grid1, scale1 = matcher.extract(img1_rgb)
    feats2, grid2, scale2 = matcher.extract(img2_rgb)

    knn = NearestNeighbors(n_neighbors=2).fit(feats1)
    dists, idxs = knn.kneighbors(feats2)

    good = (dists[:, 0] / (dists[:, 1] + 1e-8)) < RATIO_THRESH
    matches = [
        (i2, idxs[i2, 0], dists[i2, 0])
        for i2 in np.nonzero(good)[0]
    ]

    if not matches:
        print("No matches passed the ratio test – try a larger RATIO_THRESH.")
        return []

    matches.sort(key=operator.itemgetter(2))
    matches = matches[:TOP_MATCHES]
    return matches, grid1, scale1, grid2, scale2

def plot_matches(matcher, img1_rgb, img2_rgb, matches, grid1, scale1, grid2, scale2):
    fig = plt.figure(figsize=(20, 10))
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    ax1.imshow(img1_rgb); ax2.imshow(img2_rgb)
    ax1.set_title("Image 1 (reference)")
    ax2.set_title("Image 2 (query)")
    for a in (ax1, ax2): a.set_axis_off()

    for i2, i1, _ in matches:
        xA, yA = matcher._grid_idx_to_xy(int(i1), grid1, scale1)
        xB, yB = matcher._grid_idx_to_xy(int(i2), grid2, scale2)

        ax2.add_artist(
            mpatches.ConnectionPatch(
                xyA=(xB, yB),
                xyB=(xA, yA),
                coordsA="data",
                coordsB="data",
                axesA=ax2,
                axesB=ax1,
                color=np.random.rand(3,), lw=1, alpha=0.8,
            )
        )

    plt.tight_layout()
    plt.show()

def display_images(img1_rgb, img2_rgb):
    fig = plt.figure(figsize=(20, 10))
    ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
    ax1.imshow(img1_rgb); ax2.imshow(img2_rgb)
    ax1.set_title("Image 1 (reference)")
    ax2.set_title("Image 2 (query)")
    for a in (ax1, ax2): a.set_axis_off()
    plt.show()


if __name__ == "__main__":
    
    # Load images
    img1_rgb, img2_rgb = load_img_pair(IMG1_PATH, IMG2_PATH)

    # Gaussian blur
    img1_rgb = gaussian_blur(img1_rgb, ksize=11)
    img2_rgb = gaussian_blur(img2_rgb, ksize=11)
    display_images(img1_rgb, img2_rgb)

    # Invert colors
    img2_rgb = invert_colors(img2_rgb)
    display_images(img1_rgb, img2_rgb)

    # Reduce midtones with gamma correction
    img2_rgb = gamma_correction(img2_rgb, gamma=7.0)
    display_images(img1_rgb, img2_rgb)

    # Canny edges
    # edge1 = canny_edges(img1_rgb, low=100, high=300)
    # edge2 = canny_edges(img2_rgb, low=40, high=60, dilate=True)
    # display_images(img1_rgb, img2_rgb)

    # Create matcher
    matcher = Dinov2Matcher(repo_root=DINOV2_REPO)

    # Extract features and match
    matches, grid1, scale1, grid2, scale2 = match(matcher, img1_rgb, img2_rgb)
    if matches:
        # Plot matches
        plot_matches(matcher, img1_rgb, img2_rgb, matches, grid1, scale1, grid2, scale2)
    else:
        print("No matches found.")
