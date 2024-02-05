import numpy as np

from calibration import compute_mx_my, estimate_f_b
from extract_patches import extract_patches
from numpy.lib.function_base import corrcoef


def triangulate(u_left, u_right, v, calib_dict):
    """
    Triangulate (determine 3D world coordinates) a set of points given their projected coordinates in two images.
    These equations are according to the simple setup, where C' = (b, 0, 0)

    Args:
        u_left  (np.array of shape (num_points,))   ... Projected u-coordinates of the 3D-points in the left image
        u_right (np.array of shape (num_points,))   ... Projected u-coordinates of the 3D-points in the right image
        v       (np.array of shape (num_points,))   ... Projected v-coordinates of the 3D-points (same for both images)
        calib_dict (dict)                           ... Dict containing camera parameters
    
    Returns:
        points (np.array of shape (num_points, 3)   ... Triangulated 3D coordinates of the input - in units of [mm]
    """

    disparity_mm = np.array((u_left - u_right) / calib_dict['mx'])
    u_left_mm = np.array((u_left - calib_dict['o_x']) / calib_dict['mx'])
    v_mm = np.array((v - calib_dict['o_y']) / calib_dict['my'])

    f = calib_dict['f']
    b = calib_dict['b']

    points = np.empty((u_left.shape[0], 3))
    points[:, 0] = b * u_left_mm /  disparity_mm
    points[:, 1] = b * v_mm / disparity_mm
    points[:, 2] = np.clip(b * f / disparity_mm, 0, None)

    return points


def compute_ncc(img_l, img_r, p):
    """
    Calculate normalized cross-correlation (NCC) between patches at the same row in two images.
    
    The regions near the boundary of the image, where the patches go out of image, are ignored.
    That is, for an input image, "p" number of rows and columns will be ignored on each side.

    For input images of size (H, W, C), the output will be an array of size (H - 2*p, W - 2*p, W - 2*p)

    Args:
        img_l (np.array of shape (H, W, C)) ... Left image
        img_r (np.array of shape (H, W, C)) ... Right image
        p (int):                            ... Defines square neighborhood. Patch-size is (2*p+1, 2*p+1).
                              
    Returns:
        corr    ... (np.array of shape (H - 2*p, W - 2*p, W - 2*p))
                    The value output[r, c_l, c_r] denotes the NCC between the patch centered at (r + p, c_l + p) 
                    in the left image and the patch centered at  (r + p, c_r + p) at the right image.
    """

    # Add dummy channel dimension
    if img_l.ndim == 2:
        img_l = img_l[:, :, None]
        img_r = img_r[:, :, None]
    
    assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
    assert img_l.shape == img_r.shape, "Shape mismatch."
    
    H, W, _ = img_l.shape

    # Extract patches - patches_l/r are NumPy arrays of shape H, W, C * (2*p+1)**2
    patches_l = extract_patches(img_l, 2*p+1)
    patches_r = extract_patches(img_r, 2*p+1)

    # p is here defined as the number of pixels to the right/left, resp. to the top/bottom.
    # if p = 2, the extracted patch will have size 5x5
    # --> the p in the handout refers to 2*p+1 in this file
    
    # Standardize each patch
    patches_l = (patches_l - np.mean(patches_l, axis=2, keepdims=True)) / np.std(patches_l, axis=2, keepdims=True)
    patches_r = (patches_r - np.mean(patches_r, axis=2, keepdims=True)) / np.std(patches_r, axis=2, keepdims=True)
    
    # Compute correlation (using matrix multiplication) - corr will be of shape H, W, W 
    cc_matrix = np.matmul(patches_l, patches_r.swapaxes(1, 2))

    # normalize cc_matrix -> ncc_matrix
    norm_factor_l = np.sqrt(np.sum(patches_l**2, axis=2, keepdims=True))
    norm_factor_r = np.sqrt(np.sum(patches_r**2, axis=2, keepdims=True))

    ncc_matrix = cc_matrix / (norm_factor_l * norm_factor_r)
    ncc_matrix = np.nan_to_num(ncc_matrix)
    
    ncc_matrix = np.array(ncc_matrix * np.triu(np.ones((W, W)), k=-40))
    
    # Ignore boundaries
    return ncc_matrix[p:H-p, p:W-p, p:W-p]


class Stereo3dReconstructor:
    def __init__(self, p=5, w_mode='none'):
        """
        Feel free to add hyper parameters here, but be sure to set defaults
        
        Args:
            p       ... Patch size for NCC computation
            w_mode  ... Weighting mode. I.e. method to compute certainty scores
        """
        self.p = p
        self.w_mode = w_mode

    def fill_calib_dict(self, calib_dict, calib_points):
        """ Fill missing entries in calib dict - nothing to do here """
        calib_dict['mx'], calib_dict['my'] = compute_mx_my(calib_dict)
        calib_dict['f'], calib_dict['b'] = estimate_f_b(calib_dict, calib_points)
        
        return calib_dict

    def recon_scene_3d(self, img_l, img_r, calib_dict):
        """
        Compute point correspondences for two images and perform 3D reconstruction.

        Args:
            img_l (np.array of shape (H, W, C)) ... Left image
            img_r (np.array of shape (H, W, C)) ... Right image
            calib_dict (dict)                   ... Dict containing camera parameters
        
        Returns:
            points3d (np.array of shape (H, W, 4)
                Array containing the re-constructed 3D world coordinates for each pixel in the left image.
                Boundary points - which are not well defined for NCC might be padded with 0s.
                4th dimension holds the certainties.
        """

        # Add dummy channel dimension
        if img_l.ndim == 2:
            img_l = img_l[:, :, None]
            img_r = img_r[:, :, None]
        
        assert img_l.ndim == 3, f"Expected 3 dimensional input. Got {img_l.shape}"
        assert img_l.shape == img_r.shape, "Shape mismatch."
        
        H, W, C = img_l.shape
        
        # During NCC comutation we discard boundaries
        # This shifts the indices of the center pixels
        H_small, W_small = H - 2*self.p, W - 2*self.p
        
        calib_small = calib_dict
        calib_small['height'], calib_small['width'] = H_small, W_small
        calib_small['o_x'] = calib_dict['o_x'] - self.p
        calib_small['o_y'] = calib_dict['o_y'] - self.p
        
        # Create (u, v) pixel grid
        y, u_left = np.meshgrid(
            np.arange(H_small, dtype=float),
            np.arange(W_small, dtype=float),
            indexing='ij'
        )

        # Compute normalized cross correlation & find correspondece
        corr = compute_ncc(img_l, img_r, self.p)

        # Find correspondence
        #
        # TO IMPLEMENT
        #
        indices = np.where(np.arange(corr.shape[1]) >= np.arange(corr.shape[2])[:, None])
        corr_copy = corr.copy()
        corr_copy[:, indices[0], indices[1]] = 0

        u_right = np.argmax(corr_copy, axis=2)
        
        # Set certainty
        #
        # TO IMPLEMENT
        #
        if self.w_mode == 'none':
            certainty_score = np.ones((H_small, W_small), dtype=float)
        else:
            certainty_score = np.mean(corr, axis=2)
        
        # Triangulate the points to get 3D world coordinates
        points = triangulate(
            u_left.flatten(), u_right.flatten(), y.flatten(), calib_small
        )

        # Reshape to image & pad
        points = points.reshape(H_small, W_small, 3)
        points = np.pad(points, ((self.p, self.p), (self.p, self.p), (0, 0)))
        certainty_score = np.pad(certainty_score, self.p)[:, :, None]

        return np.concatenate([points, certainty_score], axis=2)