```py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class TensorQRDecomposition:
    """
    Tensor QR Decomposition for image processing applications
    Simplified implementation for educational purposes
    """
    
    def __init__(self, mode='M_product'):
        self.mode = mode
    
    def m_product(self, A, B, M):
        """
        Compute M-product of two tensors A and B with transformation matrix M
        A: (m, n, p) tensor
        B: (n, k, p) tensor  
        M: (p, p) transformation matrix
        """
        # Transform via mode-3 product
        A_transformed = self.mode_3_product(A, M)
        B_transformed = self.mode_3_product(B, M)
        
        # Compute transformed product
        result = self.t_product(A_transformed, B_transformed)
        
        # Inverse transform
        result = self.mode_3_product(result, np.linalg.inv(M))
        return result
    
    def mode_3_product(self, A, M):
        """
        Mode-3 product of tensor A with matrix M
        """
        m, n, p = A.shape
        A_flat = A.reshape(m * n, p)
        result_flat = A_flat @ M.T
        result = result_flat.reshape(m, n, p)
        return result
    
    def t_product(self, A, B):
        """
        Tensor product in transformed domain
        """
        m, n, p = A.shape
        n2, k, p2 = B.shape
        assert n == n2 and p == p2
        
        # Convert to Fourier domain
        A_f = np.fft.fft(A, axis=2)
        B_f = np.fft.fft(B, axis=2)
        
        # Perform matrix multiplication in Fourier domain
        C_f = np.zeros((m, k, p), dtype=complex)
        for i in range(p):
            C_f[:, :, i] = A_f[:, :, i] @ B_f[:, :, i]
        
        # Convert back to spatial domain
        C = np.fft.ifft(C_f, axis=2).real
        return C
    
    def tensor_qr(self, A, M=None):
        """
        Compute tensor QR decomposition using M-product
        """
        if M is None:
            M = np.eye(A.shape[2])
        
        m, n, p = A.shape
        
        # Initialize Q and R tensors
        Q = np.zeros((m, n, p))
        R = np.zeros((n, n, p))
        
        # Transform tensor using M
        A_transformed = self.mode_3_product(A, M)
        
        # Process each frontal slice
        for i in range(p):
            # Compute QR decomposition for each slice
            Q_slice, R_slice = np.linalg.qr(A_transformed[:, :, i])
            
            # Store results
            Q[:, :, i] = Q_slice
            R[:, :, i] = R_slice
        
        # Inverse transform Q and R
        Q = self.mode_3_product(Q, np.linalg.inv(M))
        R = self.mode_3_product(R, np.linalg.inv(M))
        
        return Q, R
    
    def compress_image(self, image, compression_ratio=0.5):
        """
        Apply tensor QR decomposition for image compression
        """
        if len(image.shape) == 2:
            # Convert grayscale to "tensor" by adding third dimension
            image_tensor = image[:, :, np.newaxis]
        else:
            image_tensor = image.transpose(2, 0, 1)  # Change to (channels, height, width)
            image_tensor = image_tensor[np.newaxis, :, :, :]  # Add batch dimension
            image_tensor = image_tensor.transpose(0, 2, 3, 1)  # Change to (1, h, w, c)
        
        m, n, p = image_tensor.shape
        
        # Compute tensor QR decomposition
        Q, R = self.tensor_qr(image_tensor)
        
        # Determine rank for compression
        k = int(min(m, n) * compression_ratio)
        
        # Truncate Q and R
        Q_compressed = Q[:, :k, :]
        R_compressed = R[:k, :, :]
        
        # Reconstruct compressed image
        compressed_tensor = self.m_product(Q_compressed, R_compressed, np.eye(p))
        
        # Convert back to image format
        if len(image.shape) == 2:
            compressed_image = compressed_tensor[:, :, 0]
        else:
            compressed_tensor = compressed_tensor.transpose(0, 3, 1, 2)  # Change to (1, c, h, w)
            compressed_image = compressed_tensor[0].transpose(1, 2, 0)  # Change to (h, w, c)
        
        return compressed_image.astype(np.uint8), Q, R

# Demonstration and testing
def demonstrate_tensor_qr():
    """
    Demonstrate tensor QR decomposition for image processing
    """
    # Create sample image data
    image = np.random.rand(64, 64, 3) * 255
    image = image.astype(np.uint8)
    
    # Initialize tensor QR
    tqr = TensorQRDecomposition()
    
    print("Original image shape:", image.shape)
    
    # Apply compression
    compressed, Q, R = tqr.compress_image(image, compression_ratio=0.3)
    
    print("Compressed image shape:", compressed.shape)
    print("Q tensor shape:", Q.shape)
    print("R tensor shape:", R.shape)
    
    # Calculate compression statistics
    original_size = image.nbytes
    compressed_size = compressed.nbytes
    compression_ratio = compressed_size / original_size
    
    print(f"Original size: {original_size} bytes")
    print(f"Compressed size: {compressed_size} bytes") 
    print(f"Compression ratio: {compression_ratio:.2f}")
    
    # Calculate quality metrics
    mse = np.mean((image.astype(float) - compressed.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {psnr:.2f} dB")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    if len(image.shape) == 3:
        axes[0].imshow(image)
        axes[1].imshow(compressed)
    else:
        axes[0].imshow(image, cmap='gray')
        axes[1].imshow(compressed, cmap='gray')
    
    axes[0].set_title(f'Original Image\n{image.shape}')
    axes[1].set_title(f'Compressed Image (Ratio: {compression_ratio:.2f})\nPSNR: {psnr:.2f} dB')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return image, compressed, Q, R

if __name__ == "__main__":
    original, compressed, Q, R = demonstrate_tensor_qr()
```
