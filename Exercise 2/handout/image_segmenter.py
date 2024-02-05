import numpy as np

from kmeans import (
    compute_distance,
    kmeans_fit,
    kmeans_predict_idx
)

from extract_patches import extract_patches
from advanced_methods import perform_min_cut


class ImageSegmenter:
    def __init__(self, k_fg=3, k_bg=3, p=1, n_iter=20, tol=1.e-4, alpha=80, mode='kmeans'):
        """ Feel free to add any hyper-parameters to the ImageSegmenter.
            
            But note:
            For the final submission the default hyper-parameteres will be used.
            In particular the segmetation will likely crash, if no defaults are set.
        """
        
        # Number of clusters in FG/BG
        self.k_fg = k_fg
        self.k_bg = k_bg
        self.p = p
        self.n_iter = n_iter
        self.tol = tol
        self.alpha = alpha
        
        self.mode= mode
        
    def extract_features_(self, sample_dd):
        """ Extract features, e.g. p x p neighborhood of pixel, from the RGB image """
        
        img = sample_dd['img']
        H, W, C = img.shape

        # --- extract patches ---
        if self.p == 1:
          return img
        patches = extract_patches(img, self.p)

        return patches
    
    def segment_image_dummy(self, sample_dd):
        return sample_dd['scribble_fg']

    def segment_image_kmeans(self, sample_dd):
        """ Segment images using k means """
        H, W, C = sample_dd['img'].shape
        features = self.extract_features_(sample_dd)

        # --- generate fg and bg masks ---
        n_features = features.shape[-1]

        fg_mask = np.where(sample_dd["scribble_fg"] > 100, True, False)
        bg_mask = np.where(sample_dd["scribble_bg"] > 100, True, False)
        fg_mask_stacked = np.repeat(fg_mask[:, :, np.newaxis], n_features, axis=-1)
        bg_mask_stacked = np.repeat(bg_mask[:, :, np.newaxis], n_features, axis=-1)

        # --- apply k-means for fg ---
        fg_data = features[fg_mask_stacked]
        fg_data = np.reshape(fg_data, (-1, 3*self.p**2))
        fg_centroids = kmeans_fit(fg_data, self.k_fg, n_iter=self.n_iter, tol=self.tol)

        # --- apply k-means for bg ---
        bg_data = features[bg_mask_stacked]
        bg_data = np.reshape(bg_data, (-1, 3*self.p**2))
        bg_centroids = kmeans_fit(bg_data, self.k_bg, n_iter=self.n_iter, tol=self.tol)

        # --- determine fg or bg for each pixel ---

        # --- find distances in intensity space ---
        features_flat = np.reshape(features, (H*W, -1))
        
        distances_intensity = compute_distance(features_flat, np.vstack((fg_centroids, bg_centroids)))
        
        distances_intensity_min = np.empty((H*W, 2))
        distances_intensity_min[:, 0] = np.min(distances_intensity[:self.k_fg])
        distances_intensity_min[:, 1] = np.min(distances_intensity[self.k_fg:])

        
        distances_intensity_min = np.reshape(distances_intensity_min, (H, W, 2))

        # --- find distances in pixel space ---
        distances_spatial = np.empty((H, W, 2))

        fg_indices = np.argwhere(fg_mask)
        bg_indices = np.argwhere(bg_mask)

        h_grid, w_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        distances_spatial = np.empty((H, W, 2))

        # --- fg ---
        fg_differences = np.stack([h_grid[:, :, np.newaxis] - fg_indices[:, 0], w_grid[:, :, np.newaxis] - fg_indices[:, 1]], axis=-1)
        fg_distances = np.linalg.norm(fg_differences, axis=-1)
        distances_spatial[:, :, 0] = np.min(fg_distances, axis=-1)

        # --- bg ---
        bg_differences = np.stack([h_grid[:, :, np.newaxis] - bg_indices[:, 0], w_grid[:, :, np.newaxis] - bg_indices[:, 1]], axis=-1)
        bg_distances = np.linalg.norm(bg_differences, axis=-1)
        distances_spatial[:, :, 1] = np.min(bg_distances, axis=-1)

        # --- combine both distance metrics and weigh according to hyperparameter ---
        distances = np.dstack((distances_intensity_min, distances_spatial))
        predicted_mask = (self.alpha*distances_intensity_min[:, :, 0] + distances_spatial[:, :, 0]) - (self.alpha*distances_intensity_min[:, :, 1] + distances_spatial[:, :, 1])

        predicted = np.where(predicted_mask > 0, False, True)

        return predicted

    def segment_image_grabcut(self, sample_dd):
        """ Segment via an energy minimisation """

        # Foreground potential set to 1 inside box, 0 otherwise
        unary_fg = sample_dd['scribble_fg'].astype(np.float32) / 255

        # Background potential set to 0 inside box, 1 everywhere else
        unary_bg = 1 - unary_fg

        # Pairwise potential set to 1 everywhere
        pairwise = np.ones_like(unary_fg)

        # Perfirm min cut to get segmentation mask
        im_mask = perform_min_cut(unary_fg, unary_bg, pairwise)
        
        return im_mask

    def segment_image(self, sample_dd):
        """ Feel free to add other methods """
        if self.mode == 'dummy':
            return self.segment_image_dummy(sample_dd)
        
        elif self.mode == 'kmeans':
            return self.segment_image_kmeans(sample_dd)
        
        elif self.mode == 'grabcut':
            return self.segment_image_grabcut(sample_dd)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")