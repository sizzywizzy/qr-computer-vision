
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import qr, rq
from mpl_toolkits.mplot3d import Axes3D

class CameraCalibration:
    """
    Complete camera calibration implementation using QR decomposition.
    """
    
    def __init__(self):
        self.P = None  # Projection matrix
        self.K = None  # Intrinsic matrix (calibration)
        self.R = None  # Rotation matrix
        self.t = None  # Translation vector
        
    def estimate_projection_matrix(self, pts_3d, pts_2d):
        """
        Estimate projection matrix P using Direct Linear Transform (DLT).
        
        Args:
            pts_3d: Nx3 array of 3D world points
            pts_2d: Nx2 array of corresponding 2D image points
            
        Returns:
            P: 3x4 projection matrix
        """
        n = pts_3d.shape[0]
        
        # Convert to homogeneous coordinates
        pts_3d_h = np.hstack([pts_3d, np.ones((n, 1))])
        
        # Build the constraint matrix A
        A = []
        for i in range(n):
            X, Y, Z, W = pts_3d_h[i]
            u, v = pts_2d[i]
            
            A.append([X, Y, Z, W, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u*W])
            A.append([0, 0, 0, 0, X, Y, Z, W, -v*X, -v*Y, -v*Z, -v*W])
            
        A = np.array(A)
        
        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        P = Vt[-1].reshape(3, 4)
        
        # Normalize so that ||P|| = 1
        P = P / np.linalg.norm(P)
        
        self.P = P
        return P
    
    def rq_decomposition(self, M):
        """
        Perform RQ decomposition using QR decomposition with matrix flips.
        
        Args:
            M: 3x3 matrix to decompose
            
        Returns:
            R: Upper triangular matrix (calibration matrix)
            Q: Orthogonal matrix (rotation matrix)
        """
        # Flip M upside down and transpose
        M_flip = np.flipud(M).T
        
        # Standard QR decomposition
        Q_temp, R_temp = qr(M_flip)
        
        # Flip back to get RQ
        R = np.flipud(R_temp.T)
        R = np.fliplr(R)
        
        Q = Q_temp.T
        Q = np.flipud(Q)
        
        return R, Q
    
    def fix_sign_conventions(self, K, R):
        """
        Fix sign ambiguities to ensure physically meaningful parameters.
        
        Args:
            K: Calibration matrix
            R: Rotation matrix
            
        Returns:
            K_fixed: Calibration matrix with positive diagonal
            R_fixed: Rotation matrix with det(R) = 1
        """
        K_fixed = K.copy()
        R_fixed = R.copy()
        
        # Make diagonal elements of K positive
        for i in range(3):
            if K_fixed[i, i] < 0:
                K_fixed[:, i] *= -1
                R_fixed[i, :] *= -1
        
        # Ensure det(R) = 1 (proper rotation, not reflection)
        if np.linalg.det(R_fixed) < 0:
            K_fixed[:, 2] *= -1
            R_fixed[2, :] *= -1
            
        # Normalize K so that K[2,2] = 1
        K_fixed = K_fixed / K_fixed[2, 2]
        
        return K_fixed, R_fixed
    
    def decompose_projection_matrix(self):
        """
        Decompose projection matrix P into K, R, and t using QR decomposition.
        
        Returns:
            K: 3x3 intrinsic matrix
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        if self.P is None:
            raise ValueError("Projection matrix not estimated yet!")
        
        # Extract M (first 3x3 submatrix) and p4 (4th column)
        M = self.P[:, :3]
        p4 = self.P[:, 3]
        
        # RQ decomposition
        K, R = self.rq_decomposition(M)
        
        # Fix sign conventions
        K, R = self.fix_sign_conventions(K, R)
        
        # Compute translation vector
        t = np.linalg.inv(K) @ p4
        
        self.K = K
        self.R = R
        self.t = t
        
        return K, R, t
    
    def reproject_points(self, pts_3d):
        """
        Reproject 3D points to 2D using extracted camera parameters.
        
        Args:
            pts_3d: Nx3 array of 3D points
            
        Returns:
            pts_2d: Nx2 array of reprojected 2D points
        """
        if self.K is None or self.R is None or self.t is None:
            raise ValueError("Camera not calibrated yet!")
        
        # Build projection matrix from K, R, t
        P_reconstructed = self.K @ np.hstack([self.R, self.t.reshape(-1, 1)])
        
        # Convert 3D points to homogeneous coordinates
        n = pts_3d.shape[0]
        pts_3d_h = np.hstack([pts_3d, np.ones((n, 1))])
        
        # Project
        pts_2d_h = (P_reconstructed @ pts_3d_h.T).T
        
        # Convert from homogeneous to Cartesian
        pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:3]
        
        return pts_2d
    
    def compute_reprojection_error(self, pts_3d, pts_2d_observed):
        """
        Compute mean reprojection error.
        
        Args:
            pts_3d: Nx3 array of 3D points
            pts_2d_observed: Nx2 array of observed 2D points
            
        Returns:
            mean_error: Mean Euclidean distance in pixels
        """
        pts_2d_reprojected = self.reproject_points(pts_3d)
        errors = np.linalg.norm(pts_2d_reprojected - pts_2d_observed, axis=1)
        return np.mean(errors)

# Generate synthetic calibration data
def generate_calibration_data(n_points=20, noise_level=0.5):
    """
    Generate synthetic 3D-2D point correspondences for testing.
    
    Args:
        n_points: Number of point correspondences
        noise_level: Gaussian noise std dev in pixels
        
    Returns:
        pts_3d: Nx3 array of 3D points
        pts_2d: Nx2 array of 2D points with noise
        K_true: Ground truth calibration matrix
        R_true: Ground truth rotation matrix
        t_true: Ground truth translation vector
    """
    # Ground truth camera parameters
    fx, fy = 1000.0, 1000
