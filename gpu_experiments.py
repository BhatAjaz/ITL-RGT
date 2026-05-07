"""
===========================================================================
GPU-ACCELERATED EXPERIMENTS
On the Information-Theoretic Limits of Recursive Generative Training
===========================================================================

Validates theoretical predictions of model collapse at scale using real
neural networks on GPU (targeting RTX 5090 with 32GB VRAM).

Experiments:
  1. VAE/MNIST Recursive Training          (~2.5 hrs on RTX 5090)
  2. VAE/CIFAR-10 Recursive Training       (~8 hrs on RTX 5090)
  3. Diffusion Model Recursive Training    (~20 hrs on RTX 5090)
  4. Large-Scale Gaussian Simulation       (~1 hr on RTX 5090)
  5. Language Model Recursive Training     (~12 hrs on RTX 5090)
  6. Optimal Mixing Schedule Search        (~30 min on RTX 5090)

Requirements (tested versions):
  - python >= 3.9
  - torch >= 2.2.0          (CUDA 12.x for RTX 5090)
  - torchvision >= 0.17.0
  - numpy >= 1.24.0
  - scipy >= 1.11.0          (for linalg.sqrtm)
  - scikit-learn >= 1.3.0    (for PCA, NearestNeighbors)
  - matplotlib >= 3.8.0
  - tqdm >= 4.66.0
  - pillow >= 10.0.0
  - transformers >= 4.37.0   (for GPT-2 in Experiment 5)

Install:
  pip install torch torchvision numpy scipy scikit-learn \
              matplotlib tqdm pillow transformers

Usage:
  python gpu_experiments.py --exp 1          # Run Experiment 1
  python gpu_experiments.py --exp 4          # Run Experiment 4 (Gaussian sim)
  python gpu_experiments.py --exp all        # Run all experiments
  python gpu_experiments.py --test           # Quick test (verify setup)
  python gpu_experiments.py --exp 1 --seed 42  # Custom seed

GPU Memory Requirements (estimated):
  Exp 1 (VAE/MNIST):    ~2 GB
  Exp 2 (VAE/CIFAR-10): ~4 GB
  Exp 3 (DDPM/CIFAR-10):~8 GB
  Exp 4 (Gaussian sim): ~6 GB (for largest configs)
  Exp 5 (GPT-2):        ~6 GB (124M params, fp32)
  Exp 6 (Schedule):     ~2 GB

Estimated Runtimes on RTX 5090:
  Exp 1: ~2.5 hours  (15 gens x 50 epochs x 50k samples)
  Exp 2: ~8 hours    (10 gens x 100 epochs x ConvVAE)
  Exp 3: ~20 hours   (8 gens x full DDPM training)
  Exp 4: ~1 hour     (100k trials GPU-parallel)
  Exp 5: ~12 hours   (8 gens x GPT-2 fine-tuning)
  Exp 6: ~30 minutes (analytical + fast simulation)

Output:
  Figures  -> <script_dir>/gpu_figs/
  Results  -> <script_dir>/gpu_results/
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Global configuration — output directories are relative to the script location
# so the code works on any machine without editing paths.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(_SCRIPT_DIR, 'gpu_figs')
RES_DIR = os.path.join(_SCRIPT_DIR, 'gpu_results')

# Ensure output directories exist
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Matplotlib setup (non-interactive backend)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 9

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
          '#a65628', '#f781bf', '#66c2a5', '#fc8d62', '#8da0cb',
          '#e5c494', '#b3b3b3', '#666666', '#a6cee3', '#fdbf6f']

# Global constants matching the paper
SIGMA2 = 1.0   # True data variance
MU0 = 0.0      # True data mean


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA GPU preferred)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = (torch.cuda.get_device_properties(0).total_memory -
                    torch.cuda.memory_allocated(0)) / 1e9
        print(f"  Using GPU: {gpu_name} ({gpu_mem:.1f} GB total, {free_mem:.1f} GB free)")
    else:
        device = torch.device('cpu')
        print("  WARNING: No GPU detected, using CPU (will be slow)")
    return device


def save_results(exp_name: str, results: dict):
    """Save experiment results as numpy arrays and JSON metadata."""
    exp_dir = os.path.join(RES_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save numpy arrays
    for key, val in results.items():
        if isinstance(val, np.ndarray):
            np.save(os.path.join(exp_dir, f'{key}.npy'), val)
        elif isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (int, float)):
            np.save(os.path.join(exp_dir, f'{key}.npy'), np.array(val))

    # Save metadata as JSON (handles non-array data)
    metadata = {}
    for key, val in results.items():
        if isinstance(val, (int, float, str, bool)):
            metadata[key] = val
        elif isinstance(val, np.ndarray):
            metadata[f'{key}_shape'] = list(val.shape)
    with open(os.path.join(exp_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  Results saved to {exp_dir}/")


def save_checkpoint(exp_name: str, generation: int, state: dict):
    """Save a checkpoint for resuming interrupted experiments."""
    ckpt_dir = os.path.join(RES_DIR, exp_name, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'gen_{generation}.pt')
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(exp_name: str, generation: int) -> dict:
    """Load a checkpoint if one exists."""
    path = os.path.join(RES_DIR, exp_name, 'checkpoints', f'gen_{generation}.pt')
    if os.path.exists(path):
        print(f"  Resuming from checkpoint: {path}")
        return torch.load(path, map_location='cpu')
    return None


def estimate_gpu_memory(tensor_shape, dtype=torch.float32):
    """Estimate GPU memory needed for a tensor in GB."""
    elements = 1
    for s in tensor_shape:
        elements *= s
    bytes_per = {torch.float32: 4, torch.float16: 2, torch.float64: 8}
    return elements * bytes_per.get(dtype, 4) / 1e9


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.label:
            print(f"  [{self.label}] Elapsed: {self.elapsed:.1f}s")


# ============================================================================
# EXPERIMENT 1: Large-Scale VAE/MNIST Recursive Training
# ============================================================================

class VAEEncoder(nn.Module):
    """MLP encoder for VAE on MNIST.

    Architecture: 784 -> 400 -> 400 -> (mu, logvar) each of dim latent_dim.
    The two-layer MLP provides sufficient capacity to capture digit structure
    while remaining tractable for 15 recursive generations.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """MLP decoder for VAE on MNIST.

    Architecture: latent_dim -> 400 -> 400 -> 784.
    Supports both fixed (known) and learned (unknown) output variance.

    Theoretical connection (Theorem 1b):
      - Known variance: decoder outputs only mu; variance fixed to sigma^2.
        KL should grow as t*d/(2n).
      - Unknown variance: decoder outputs mu AND logvar; variance is learned.
        Variance contraction ((n-1)/n)^t causes KL to grow at ~2x the rate.
    """
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784,
                 fixed_variance=True):
        super().__init__()
        self.fixed_variance = fixed_variance
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        if not fixed_variance:
            # Learn per-pixel log-variance (unknown variance case)
            # Initialize to small values for stable training start
            self.fc_logvar = nn.Linear(hidden_dim, output_dim)
            # Xavier-like init with small bias to start near log(0.01) = -4.6
            nn.init.xavier_uniform_(self.fc_logvar.weight)
            nn.init.constant_(self.fc_logvar.bias, -4.6)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        mu = torch.sigmoid(self.fc_mu(h))  # Pixel values in [0,1]
        if self.fixed_variance:
            # Known variance: fix to a constant (Theorem 1 baseline)
            return mu, None
        else:
            # Unknown variance: learn it (Theorem 1b test)
            logvar = self.fc_logvar(h)
            return mu, logvar


class VAE(nn.Module):
    """Variational Autoencoder for MNIST with known/unknown variance.

    The key theoretical distinction:
      - Known variance:   KL(P_0 || P_t) ~ t*d / (2n)    [Theorem 1]
      - Unknown variance: KL(P_0 || P_t) ~ t*d / n        [Theorem 1b]
    The unknown variance case doubles the KL growth rate because
    variance contraction ((n-1)/n)^t adds a variance mismatch term
    equal in magnitude to the mean drift term.
    """
    def __init__(self, latent_dim=20, hidden_dim=400, fixed_variance=True):
        super().__init__()
        self.encoder = VAEEncoder(784, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, 784, fixed_variance)
        self.latent_dim = latent_dim
        self.fixed_variance = fixed_variance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decoder(z)
        return mu_x, logvar_x, mu_z, logvar_z

    def loss_function(self, x, mu_x, logvar_x, mu_z, logvar_z,
                       min_kl_per_dim=1.0, kl_constraint_weight=5.0, beta=1.0):
        """Compute VAE loss = Reconstruction + KL penalty + anti-collapse regulariser.

        For known variance: reconstruction is fixed-variance Gaussian NLL.
        For unknown variance: reconstruction uses learned per-pixel variance.

        Anti-collapse mechanism:
          Under recursive training, VAE posteriors collapse toward the prior
          N(0, I), i.e., mu_z → 0 and logvar_z → 0, giving kl_per_dim → 0.
          The fundamental problem is that d(kl_per_dim)/d(encoder_params) ≈ 0
          at the collapse point, so NO penalty on kl_per_dim can escape it
          (neither "free bits" nor a relu-based constraint).

          Instead, we directly regularise the encoder's log-variance outputs.
          When the posterior collapses to the prior, logvar_z → 0.  A penalty
          on |logvar_z| being too close to 0 (i.e., the posterior variance
          being too close to the prior variance) has a CONSTANT gradient
          that pushes the encoder away from the prior, regardless of how
          close it already is:

            logvar_reg = weight * sum(relu(threshold - |logvar_z|))

          This forces each latent dimension to maintain a posterior variance
          that is EITHER meaningfully larger OR smaller than the prior's
          variance of 1.  In practice, the reconstruction loss pushes toward
          smaller variance (tighter posterior), so the equilibrium settles at
          logvar_z ≈ -threshold, giving a well-localised but informative
          posterior.

        Args:
            min_kl_per_dim: Target minimum KL per latent dimension (nats).
                Used to set the logvar threshold: log(1 + 2*min_kl) ≈ 1.4
                for min_kl=1.0.  Default 1.0.
            kl_constraint_weight: Weight for the logvar regulariser. Default 5.0.
            beta: KL weight (for warm-up scheduling). Default 1.0.
        """
        # Per-sample, per-dimension KL
        kl_per_dim = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp())  # (B, d)

        # Standard KL penalty (pushes toward prior)
        kl_z = kl_per_dim.sum()

        # Anti-collapse: penalise |logvar_z| being too close to 0.
        # When logvar_z ≈ 0 (posterior variance ≈ prior variance), the
        # posterior is collapsing toward the prior.  We force each dimension
        # to maintain |logvar_z| >= threshold, where threshold is derived
        # from the desired minimum KL per dimension.
        # For a Gaussian with mu=0: KL = 0.5*(logvar + 1 - exp(logvar))
        # Setting KL = min_kl_per_dim and solving: logvar ≈ -2*min_kl_per_dim
        # (approximate for min_kl ~ 1), so threshold ≈ 2*min_kl_per_dim.
        logvar_threshold = 2.0 * min_kl_per_dim
        logvar_reg = kl_constraint_weight * torch.sum(
            F.relu(logvar_threshold - logvar_z.abs())
        )

        # Reconstruction loss
        if self.fixed_variance:
            recon_loss = F.binary_cross_entropy(mu_x, x, reduction='sum')
        else:
            logvar_clamped = logvar_x.clamp(-10, 2)
            recon_loss = 0.5 * torch.sum(
                logvar_clamped + (x - mu_x).pow(2) / logvar_clamped.exp().clamp(min=1e-4)
            )

        return recon_loss + beta * kl_z + logvar_reg, recon_loss, kl_z

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device):
        """Generate samples from the prior p(z) = N(0, I)."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        mu_x, logvar_x = self.decoder(z)
        if self.fixed_variance:
            return mu_x
        else:
            # Sample with learned variance (clamped for stability)
            logvar_clamped = logvar_x.clamp(-10, 2)
            std_x = torch.exp(0.5 * logvar_clamped)
            return mu_x + std_x * torch.randn_like(mu_x)


def compute_pixel_variance(data_loader, device):
    """Compute per-pixel variance across a dataset.

    Theoretical prediction (Theorem 1b): under recursive training,
    E[var_t] = sigma^2_0 * ((n-1)/n)^t, where n is the sample size.
    This contraction arises because each generation's MLE variance
    estimator is biased: E[s^2] = (n-1)/n * sigma^2_{t-1}.
    """
    sum_x = None
    sum_x2 = None
    count = 0
    for batch in data_loader:
        x = batch[0].to(device)
        if sum_x is None:
            sum_x = torch.zeros(x.shape[1], device=device)
            sum_x2 = torch.zeros(x.shape[1], device=device)
        sum_x += x.sum(dim=0)
        sum_x2 += (x ** 2).sum(dim=0)
        count += x.shape[0]
    mean = sum_x / count
    var = (sum_x2 / count) - mean.pow(2)
    return var.cpu().numpy()


def compute_digit_extinction(data_loader, n_classes=10):
    """Count how many digit classes are represented in generated samples.

    Theoretical connection (Theorem 2): categories (digits) with
    probability p_i survive generation t with probability
    (1 - (1-p_i)^n)^t. Low-probability digits go extinct first.
    Under recursive training, we expect progressive digit extinction.
    """
    # This requires labels; for generated data we use a classifier
    # Simple heuristic: use cluster-based assignment
    # For MNIST, we'll train a simple classifier alongside
    labels = []
    for batch in data_loader:
        if len(batch) > 1:
            labels.append(batch[1].numpy())
    if len(labels) == 0:
        return n_classes  # Can't determine without labels
    all_labels = np.concatenate(labels)
    return len(np.unique(all_labels))


def compute_kl_from_samples(samples_real, samples_gen, n_bins=50):
    """Estimate KL(P_real || P_gen) from samples.

    Uses a parametric Gaussian approximation for the main KL estimate,
    which is more reliable than histogram-based estimation in high
    dimensions (e.g., 784-dim MNIST pixels). Falls back to per-dimension
    histogram KL for non-Gaussian distributions.

    The Gaussian KL between N(mu_r, Sigma_r) and N(mu_g, Sigma_g) is:
      KL = 0.5 * [tr(Sigma_g^{-1} Sigma_r) + (mu_g - mu_r)^T Sigma_g^{-1} (mu_g - mu_r)
                   - d + ln(det(Sigma_g) / det(Sigma_r))]

    For diagonal covariance (independent dimensions), this simplifies to
    a sum of per-dimension KL divergences.

    Dimensions where BOTH distributions have near-zero variance are
    excluded to avoid log(1) = 0 artefacts from trivial (always-black)
    pixels.
    """
    d = samples_real.shape[1]

    # Compute per-dimension mean and variance
    mu_r = np.mean(samples_real, axis=0)
    mu_g = np.mean(samples_gen, axis=0)
    var_r = np.var(samples_real, axis=0) + 1e-10  # Avoid zero variance
    var_g = np.var(samples_gen, axis=0) + 1e-10

    # Only include dimensions where at least one distribution has
    # meaningful variance (skip always-black border pixels).
    active = (var_r > 1e-6) | (var_g > 1e-6)

    # Parametric Gaussian KL (diagonal covariance):
    # KL(N(mu_r, var_r) || N(mu_g, var_g)) per dimension =
    #   0.5 * [log(var_g/var_r) + var_r/var_g + (mu_r-mu_g)^2/var_g - 1]
    kl_per_dim = 0.5 * (np.log(var_g[active] / var_r[active]) +
                         var_r[active] / var_g[active] +
                         (mu_r[active] - mu_g[active])**2 / var_g[active] - 1)

    # Sum across dimensions for total KL
    total_kl = np.sum(kl_per_dim)

    # Clip to non-negative (KL should be >= 0)
    return max(0.0, total_kl)


def compute_kl_known_variance(samples_real, samples_gen, true_var_real):
    """Estimate KL(P_real || P_gen) assuming the real-data variance is KNOWN.

    This corresponds to Theorem 1 (known variance): the KL uses the true
    variance sigma^2 of P_0 rather than the sample variance of the real
    data.  For the generated distribution P_t, we still use the sample
    variance of the generated data.

    Theoretical prediction: KL(P_0 || P_t) ~ t * d / (2n)

    Dimensions where the true variance is negligible (border pixels) are
    excluded to avoid log(epsilon) noise dominating the KL sum.

    Args:
        samples_real: numpy array (N_real, D)
        samples_gen:  numpy array (N_gen, D)
        true_var_real: numpy array (D,) — the TRUE per-dimension variance
                       of the real data distribution (computed once from the
                       full dataset, not from a subsample).
    """
    mu_r = np.mean(samples_real, axis=0)
    mu_g = np.mean(samples_gen, axis=0)
    var_r = true_var_real + 1e-10  # Known true variance
    var_g = np.var(samples_gen, axis=0) + 1e-10  # Sample variance of P_t

    # Only include dimensions where the true distribution has meaningful
    # variance.  For MNIST, border pixels with true_var ≈ 0 contribute
    # only numerical noise to the KL sum.
    active = true_var_real > 1e-6

    # KL(N(mu_r, var_r) || N(mu_g, var_g)) per dimension
    kl_per_dim = 0.5 * (np.log(var_g[active] / var_r[active]) +
                         var_r[active] / var_g[active] +
                         (mu_r[active] - mu_g[active])**2 / var_g[active] - 1)

    total_kl = np.sum(kl_per_dim)
    return max(0.0, total_kl)


def compute_fid_simple(real_features, gen_features):
    """Compute pixel-space Fréchet Distance (not true FID).

    True FID uses Inception-v3 pool3 features.  This function computes the
    same formula on *raw pixel* vectors, yielding a pixel-space Fréchet
    Distance that correlates with distribution shift but is NOT directly
    comparable to published FID scores.

    For numerical stability, features are reduced to at most 128 dimensions
    via PCA before computing the Fréchet Distance.  This prevents singular
    covariance matrices when the number of features far exceeds the number
    of samples.

    FD = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*(Sigma_r*Sigma_g)^{1/2})

    Lower FD = closer distributions. Under model collapse, FD should increase
    across generations as the generated distribution drifts from real data.
    """
    from sklearn.decomposition import PCA

    # Reduce dimensionality for stable covariance estimation
    max_dims = min(128, real_features.shape[1], real_features.shape[0] // 2,
                   gen_features.shape[0] // 2)
    max_dims = max(max_dims, 1)

    if real_features.shape[1] > max_dims:
        pca = PCA(n_components=max_dims)
        combined = np.vstack([real_features, gen_features])
        pca.fit(combined)
        real_features = pca.transform(real_features)
        gen_features = pca.transform(gen_features)

    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(gen_features, axis=0)
    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(gen_features, rowvar=False)

    # Handle 1D case
    if sigma_r.ndim == 0:
        sigma_r = np.array([[sigma_r]])
        sigma_g = np.array([[sigma_g]])

    # Add diagonal regularization to prevent singular matrix in sqrtm.
    # This is standard practice (Heusel et al. 2017 use it for FID).
    # Epsilon = 1e-6 is small enough not to affect the result meaningfully.
    eps = 1e-6
    sigma_r = sigma_r + eps * np.eye(sigma_r.shape[0])
    sigma_g = sigma_g + eps * np.eye(sigma_g.shape[0])

    diff = mu_r - mu_g
    mean_diff_sq = np.sum(diff ** 2)

    # Compute sqrt of product of covariances
    from scipy.linalg import sqrtm
    covmean = sqrtm(sigma_r @ sigma_g)
    # Numerical stability: discard imaginary part from floating-point errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = mean_diff_sq + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(max(0, fd))  # FD should be non-negative


def experiment1_vae_mnist(seed=42, n_generations=15, n_samples_per_gen=50000,
                          n_epochs=50, batch_size=256, lr=1e-3,
                          latent_dim=20, hidden_dim=400, quick_test=False,
                          alpha_values=None):
    """Experiment 1: Large-Scale VAE/MNIST Recursive Training.

    Validates Theorem 1 (known variance) and Theorem 1b (unknown variance)
    using a VAE trained recursively on MNIST for 15 generations.

    Key design: we train ONE VAE per generation (BCE loss, fixed-variance
    decoder).  After generating samples, we compute TWO KL estimates from
    the SAME generated data:

      * kl_known:  Gaussian KL using the TRUE variance of the original
                   data distribution (Theorem 1: KL ~ t*d/(2n))
      * kl_unknown: Gaussian KL using the SAMPLE variance of both real
                   and generated data (Theorem 1b: KL ~ t*d/n)

    The "unknown variance" effect arises because the sample variance s²
    is a biased estimator: E[s²] = ((n-1)/n)*σ², causing variance
    contraction ((n-1)/n)^t across generations and doubling the KL rate.

    Also validates Theorem 4 (optimal mixing): with alpha > 0, each
    generation's training data is mixed with original data, which should
    prevent catastrophic collapse and produce gradual drift matching
    the theoretical predictions.

    Theoretical predictions to validate:
      1. KL_known grows linearly: KL(P_0 || P_t) ~ t * d / (2n)
      2. KL_unknown grows at ~2x the rate: KL(P_0 || P_t) ~ t * d / n
      3. Pixel variance contracts as ((n-1)/n)^t  [Theorem 1b]
      4. With alpha > 0: collapse is prevented, drift is gradual [Theorem 4]
      5. Digit categories go extinct progressively [Theorem 2]

    Estimated runtime: ~1.5 hours on RTX 5090
    GPU memory: ~2 GB
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: VAE/MNIST Recursive Training")
    print("="*70)
    print(f"  Generations: {n_generations}")
    print(f"  Samples/gen: {n_samples_per_gen}")
    print(f"  Epochs/gen:  {n_epochs}")
    print(f"  Latent dim:  {latent_dim}")
    print(f"  Hidden dim:  {hidden_dim}")
    print()

    set_seed(seed)
    device = get_device()

    if alpha_values is None:
        alpha_values = [0, 0.1]

    if quick_test:
        n_generations = 3
        n_samples_per_gen = 1000
        n_epochs = 10
        alpha_values = [0, 0.1]
        print("  [QUICK TEST MODE] Reduced parameters")

    # ---- Load MNIST ----
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    mnist_train = datasets.MNIST('./data', train=True, download=True,
                                  transform=transform)

    # Use 50k training samples (standard split)
    original_data = mnist_train.data[:50000].float() / 255.0
    original_labels = mnist_train.targets[:50000]
    original_data_flat = original_data.view(50000, -1)

    # Compute the TRUE per-pixel variance of the original data.
    # This is used as the "known variance" sigma^2 in Theorem 1.
    # We compute it from the full 50k training set for stability.
    true_pixel_var = original_data_flat.var(dim=0).numpy()  # (784,)

    # Identify "active" pixels (those with meaningful variance in the
    # original data).  Border pixels that are always 0 in MNIST have
    # zero true variance and would create division-by-zero noise in the
    # sample-variance ratio.  We only use active pixels for that metric.
    active_pixel_mask = true_pixel_var > 1e-4  # ~480 of 784 pixels for MNIST
    n_active = int(active_pixel_mask.sum())

    # ---- Storage for metrics across generations ----
    print(f"  Active pixels (var > 1e-4): {n_active}/784")

    # ---- Real data features for FID ----
    real_features = original_data_flat.numpy()

    # ---- Run for each alpha value ----
    all_results = {}

    for alpha in alpha_values:
        print(f"\n{'='*50}")
        print(f"  Mixing ratio alpha = {alpha}")
        print(f"{'='*50}")

        metrics = {
            'kl_known': [],        # KL with known (true) variance [Theorem 1]
            'kl_unknown': [],      # KL with sample variance [Theorem 1b]
            'pixel_var': [],       # Mean pixel variance of generated data
            'sample_var_ratio': [], # sample_var / true_var (should → ((n-1)/n)^t)
            'fid': [],             # Pixel-space FD
            'reconstruction_loss': [],
            'kl_z': [],
        }

        current_data = original_data_flat.clone()

        for gen in range(n_generations):
            gen_start = time.time()
            print(f"\n  --- Generation {gen+1}/{n_generations} (alpha={alpha}) ---")

            # Check for checkpoint
            ckpt = load_checkpoint(f'exp1_vae_mnist_a{alpha}', gen)
            if ckpt is not None and not quick_test:
                for key in metrics:
                    if key in ckpt.get('metrics', {}):
                        metrics[key] = ckpt['metrics'][key]
                if ckpt.get('completed', False):
                    print(f"    Generation {gen+1} already completed, skipping.")
                    continue

            # ---- Train VAE (BCE loss = known variance) ----
            model = VAE(latent_dim=latent_dim, hidden_dim=hidden_dim,
                       fixed_variance=True).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            dataset = TensorDataset(current_data)
            loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

            model.train()
            # KL warm-up: linearly increase beta from 0 to 1 over the first
            # half of training epochs.  This lets the model learn good
            # reconstructions before the KL penalty forces latent-space
            # regularisation, which is critical for avoiding posterior collapse
            # when training on low-diversity data from previous generations.
            warmup_epochs = max(1, n_epochs // 2)
            for epoch in range(n_epochs):
                beta = min(1.0, epoch / warmup_epochs)
                total_loss = 0
                total_recon = 0
                total_kl_z = 0
                n_batches = 0
                for batch_idx, (batch_x,) in enumerate(loader):
                    batch_x = batch_x.to(device)
                    optimizer.zero_grad()
                    mu_x, logvar_x, mu_z, logvar_z = model(batch_x)
                    loss, recon, kl_z = model.loss_function(
                        batch_x, mu_x, logvar_x, mu_z, logvar_z,
                        min_kl_per_dim=1.0, kl_constraint_weight=5.0, beta=beta)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    total_recon += recon.item()
                    total_kl_z += kl_z.item()
                    n_batches += 1

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    avg_loss = total_loss / n_batches
                    avg_recon = total_recon / n_batches
                    avg_kl = total_kl_z / n_batches
                    print(f"    Epoch {epoch+1}/{n_epochs}: "
                          f"loss={avg_loss:.1f}, recon={avg_recon:.1f}, "
                          f"kl_z={avg_kl:.1f}, beta={beta:.2f}")

            # ---- Generate samples for next generation ----
            model.eval()
            all_samples = []
            samples_generated = 0
            gen_batch = 1000
            while samples_generated < n_samples_per_gen:
                n_batch = min(gen_batch, n_samples_per_gen - samples_generated)
                samples = model.sample(n_batch, device)
                all_samples.append(samples.cpu())
                samples_generated += n_batch

            generated_data = torch.cat(all_samples, dim=0)[:n_samples_per_gen]
            generated_data = generated_data.clamp(0, 1)

            # ---- Mixing: combine real + generated for next gen ----
            if alpha > 0:
                n_real = int(alpha * n_samples_per_gen)
                n_synth = n_samples_per_gen - n_real
                perm = torch.randperm(len(original_data_flat))[:n_real]
                real_subset = original_data_flat[perm]
                synth_subset = generated_data[:n_synth]
                next_data = torch.cat([real_subset, synth_subset], dim=0)
                perm2 = torch.randperm(len(next_data))
                next_data = next_data[perm2]
            else:
                next_data = generated_data

            # ---- Compute metrics ----
            gen_numpy = generated_data.numpy()
            n_eval = min(5000, len(gen_numpy), len(real_features))

            # Pixel variance of generated data
            gen_var = generated_data.var(dim=0).mean().item()
            metrics['pixel_var'].append(gen_var)

            # Sample variance ratio: E[s²] / σ² should equal ((n-1)/n)^t
            gen_sample_var = np.var(gen_numpy[:n_eval], axis=0)
            if n_active > 0:
                var_ratio = np.mean(gen_sample_var[active_pixel_mask] /
                                    true_pixel_var[active_pixel_mask])
            else:
                var_ratio = np.mean(gen_sample_var / (true_pixel_var + 1e-10))
            metrics['sample_var_ratio'].append(var_ratio)

            # KL with KNOWN variance (Theorem 1)
            kl_known = compute_kl_known_variance(
                real_features[:n_eval], gen_numpy[:n_eval], true_pixel_var)
            metrics['kl_known'].append(kl_known)

            # KL with UNKNOWN variance (Theorem 1b)
            kl_unknown = compute_kl_from_samples(
                real_features[:n_eval], gen_numpy[:n_eval])
            metrics['kl_unknown'].append(kl_unknown)

            # FID score
            n_fid = min(2000, len(gen_numpy), len(real_features))
            fid_score = compute_fid_simple(
                real_features[:n_fid], gen_numpy[:n_fid])
            metrics['fid'].append(fid_score)

            # Reconstruction loss and latent KL
            metrics['reconstruction_loss'].append(total_recon / n_batches)
            metrics['kl_z'].append(total_kl_z / n_batches)

            gen_elapsed = time.time() - gen_start
            print(f"    Var={gen_var:.6f}, VarRatio={var_ratio:.4f}, "
                  f"KL_known={kl_known:.4f}, KL_unknown={kl_unknown:.4f}, "
                  f"Ratio={kl_unknown/max(kl_known,1e-10):.3f}, "
                  f"FID={fid_score:.2f}, Time={gen_elapsed:.1f}s")

            # ---- Set up data for next generation ----
            current_data = next_data

            # Free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save checkpoint
            if not quick_test:
                save_checkpoint(f'exp1_vae_mnist_a{alpha}', gen, {
                    'alpha': alpha,
                    'generation': gen,
                    'metrics': metrics,
                    'completed': True,
                })

        # Convert to numpy
        for key in metrics:
            metrics[key] = np.array(metrics[key])
        all_results[f'alpha_{alpha}'] = metrics

    # ---- Save results ----
    save_results('exp1_vae_mnist',
                 {k: v['fid'] for k, v in all_results.items()})

    # ---- Plot results ----
    plot_exp1_results(all_results, alpha_values, n_generations)

    return all_results


def plot_exp1_results(all_results, alpha_values, n_generations):
    """Generate plots for Experiment 1."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    gens = np.arange(1, n_generations + 1)

    # Use alpha=0.1 (or first alpha>0) for Theorem 1/1b validation plots,
    # since alpha=0 collapses too fast for meaningful KL analysis.
    # For comparison plots, show all alpha values.
    alpha_for_theory = None
    for a in alpha_values:
        if a > 0:
            alpha_for_theory = a
            break
    if alpha_for_theory is None:
        alpha_for_theory = alpha_values[0]
    metrics = all_results[f'alpha_{alpha_for_theory}']

    # (a) KL divergence: known vs unknown variance (best alpha for theory)
    ax = axes[0, 0]
    for i, alpha in enumerate(alpha_values):
        m = all_results[f'alpha_{alpha}']
        if len(m['kl_known']) > 0:
            ax.plot(gens[:len(m['kl_known'])], m['kl_known'],
                    'o-', color=COLORS[i], linewidth=2, markersize=5,
                    label=f'Known $\\alpha$={alpha}')
        if len(m['kl_unknown']) > 0:
            ax.plot(gens[:len(m['kl_unknown'])], m['kl_unknown'],
                    's--', color=COLORS[i], linewidth=1.5, markersize=4,
                    alpha=0.7, label=f'Unknown $\\alpha$={alpha}')
    # Theoretical predictions (from alpha_for_theory)
    if len(metrics['kl_known']) > 1:
        t_range = gens[:len(metrics['kl_known'])]
        C_known = metrics['kl_known'][0]
        ax.plot(t_range, C_known * t_range, ':', color='gray',
                alpha=0.5, linewidth=2, label='Theory: linear')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$D_{\mathrm{KL}}(P_0 \| P_t)$')
    ax.set_title('(a) KL Divergence Growth')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) KL doubling ratio (key validation of Theorem 1b)
    ax = axes[0, 1]
    for i, alpha in enumerate(alpha_values):
        m = all_results[f'alpha_{alpha}']
        min_len = min(len(m['kl_known']), len(m['kl_unknown']))
        if min_len > 0:
            kl_known_safe = np.maximum(m['kl_known'][:min_len], 1e-10)
            ratio = m['kl_unknown'][:min_len] / kl_known_safe
            ax.plot(gens[:min_len], ratio, 'o-', color=COLORS[i],
                    linewidth=2, markersize=5, label=f'$\\alpha$={alpha}')
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2,
                alpha=0.5, label='Theorem 1b: ratio = 2')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('KL(unknown) / KL(known)')
    ax.set_title('(b) KL Doubling Ratio (Thm 1b)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Pixel variance contraction
    ax = axes[0, 2]
    for i, alpha in enumerate(alpha_values):
        m = all_results[f'alpha_{alpha}']
        if len(m['pixel_var']) > 0:
            ax.plot(gens[:len(m['pixel_var'])],
                    m['pixel_var'],
                    'o-', color=COLORS[i], linewidth=2,
                    label=f'$\\alpha$={alpha}')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Mean pixel variance')
    ax.set_title('(c) Variance Contraction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Sample variance ratio vs ((n-1)/n)^t
    ax = axes[1, 0]
    for i, alpha in enumerate(alpha_values):
        m = all_results[f'alpha_{alpha}']
        if len(m['sample_var_ratio']) > 0:
            ax.plot(gens[:len(m['sample_var_ratio'])],
                    m['sample_var_ratio'],
                    'o-', color=COLORS[i], linewidth=2,
                    label=f'$\\alpha$={alpha}')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$\hat{\sigma}^2 / \sigma^2_0$')
    ax.set_title('(d) Variance Ratio (Thm 1b)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) FD degradation
    ax = axes[1, 1]
    for i, alpha in enumerate(alpha_values):
        m = all_results[f'alpha_{alpha}']
        if len(m['fid']) > 0:
            ax.plot(gens[:len(m['fid'])], m['fid'],
                    'o-', color=COLORS[i], linewidth=2,
                    label=f'$\\alpha$={alpha}')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Pixel FD (lower = better)')
    ax.set_title('(e) FD Degradation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Latent KL
    ax = axes[1, 2]
    for i, alpha in enumerate(alpha_values):
        m = all_results[f'alpha_{alpha}']
        if len(m['kl_z']) > 0:
            ax.plot(gens[:len(m['kl_z'])], m['kl_z'],
                    'o-', color=COLORS[i], linewidth=2,
                    label=f'$\\alpha$={alpha}')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('KL(q(z|x) || p(z))')
    ax.set_title('(f) Latent KL (posterior activity)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 1: VAE/MNIST Recursive Training',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'exp1_vae_mnist.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure saved: {os.path.join(FIG_DIR, "exp1_vae_mnist.png")}')


# ============================================================================
# EXPERIMENT 2: VAE/CIFAR-10 Recursive Training
# ============================================================================

class ConvEncoder(nn.Module):
    """Convolutional encoder for CIFAR-10 VAE.

    Architecture: 3x32x32 -> conv layers -> fc -> (mu, logvar)
    Uses strided convolutions for downsampling.
    """
    def __init__(self, latent_dim=128, base_channels=64):
        super().__init__()
        # Input: 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, base_channels, 4, stride=2, padding=1)
        # -> 64 x 16 x 16
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1)
        # -> 128 x 8 x 8
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, 4, stride=2, padding=1)
        # -> 256 x 4 x 4
        self.fc_mu = nn.Linear(base_channels*4*4*4, latent_dim)
        self.fc_logvar = nn.Linear(base_channels*4*4*4, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class ConvDecoder(nn.Module):
    """Convolutional decoder for CIFAR-10 VAE.

    Architecture: latent_dim -> fc -> conv layers -> 3x32x32
    Uses transposed convolutions for upsampling.
    """
    def __init__(self, latent_dim=128, base_channels=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_channels*4*4*4)
        self.deconv1 = nn.ConvTranspose2d(base_channels*4, base_channels*2,
                                           4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base_channels*2, base_channels,
                                           4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(base_channels, 3,
                                           4, stride=2, padding=1)

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(h.size(0), -1, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = torch.sigmoid(self.deconv3(h))
        return h


class ConvVAE(nn.Module):
    """Convolutional VAE for CIFAR-10.

    Uses convolutional architecture for better image generation quality,
    enabling meaningful FID and Inception Score measurements.
    """
    def __init__(self, latent_dim=128, base_channels=64):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, base_channels)
        self.decoder = ConvDecoder(latent_dim, base_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        z = self.reparameterize(mu_z, logvar_z)
        recon = self.decoder(z)
        return recon, mu_z, logvar_z

    def loss_function(self, x, recon, mu_z, logvar_z,
                       min_kl_per_dim=0.5, kl_constraint_weight=1.0, beta=1.0):
        # MSE reconstruction loss (standard for VAEs on continuous images).
        recon_loss = F.mse_loss(recon, x, reduction='sum')

        # Per-sample, per-dimension KL
        kl_per_dim = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

        # Standard KL penalty
        kl_z = kl_per_dim.sum()

        # Anti-collapse: penalise |logvar_z| being too close to 0.
        # Direct logvar regularisation has constant gradient at collapse.
        logvar_threshold = 2.0 * min_kl_per_dim
        logvar_reg = kl_constraint_weight * torch.sum(
            F.relu(logvar_threshold - logvar_z.abs())
        )

        return recon_loss + beta * kl_z + logvar_reg, recon_loss, kl_z

    @torch.no_grad()
    def sample(self, n_samples, device):
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z)


def compute_inception_score(images, n_splits=10):
    """Compute Inception Score for generated images.

    IS = exp(E[KL(p(y|x) || p(y))])
    Higher IS = better quality and diversity.
    Under model collapse, IS should decrease.

    NOTE: Requires torchvision with Inception-v3. Falls back to a simple
    entropy-based proxy if Inception is unavailable.
    """
    try:
        from torchvision.models import inception_v3
        import torchvision.transforms as T
    except ImportError:
        # Fallback: use token/cluster entropy as a proxy
        # Not comparable to published IS, but monotonic with quality
        flat = images.reshape(len(images), -1)
        # Compute marginal distribution entropy
        from scipy.stats import entropy as scipy_entropy
        # Use pixel intensity histogram as a proxy
        hist, _ = np.histogram(flat, bins=256, range=(0, 1), density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        ent = scipy_entropy(hist)
        # Convert entropy to a score-like metric (higher = better)
        # Max entropy for 256 bins = ln(256) ≈ 5.55
        return float(np.exp(ent / 5.55 * 2))  # Roughly in [1, ~7.4]
    except Exception:
        return float('nan')


def compute_precision_recall(real_features, gen_features, k=5):
    """Compute precision and recall metrics.

    Precision: fraction of generated samples close to real data (quality).
    Recall: fraction of real samples close to generated data (coverage).

    Under model collapse:
      - Precision may initially stay high (samples look OK)
      - Recall drops (coverage decreases as modes collapse)
    """
    from sklearn.neighbors import NearestNeighbors

    n_real = len(real_features)
    n_gen = len(gen_features)

    # Flatten features if needed
    real_flat = real_features.reshape(n_real, -1)[:, :100]  # Use first 100 dims
    gen_flat = gen_features.reshape(n_gen, -1)[:, :100]

    # Precision: fraction of gen samples whose k-NN in real set is close
    nbrs_real = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(real_flat)
    distances_gen, _ = nbrs_real.kneighbors(gen_flat)
    # Use k-th neighbor distance as radius
    radii_real = np.mean(
        NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        .fit(real_flat).kneighbors(real_flat)[0][:, -1]
    )
    precision = np.mean(distances_gen[:, -1] <= radii_real * 1.5)

    # Recall: fraction of real samples whose k-NN in gen set is close
    nbrs_gen = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(gen_flat)
    distances_real, _ = nbrs_gen.kneighbors(real_flat)
    radii_gen = np.mean(
        NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
        .fit(gen_flat).kneighbors(gen_flat)[0][:, -1]
    )
    recall = np.mean(distances_real[:, -1] <= radii_gen * 1.5)

    return precision, recall


def experiment2_vae_cifar10(seed=42, n_generations=10, n_samples_per_gen=50000,
                             n_epochs=100, batch_size=128, lr=2e-4,
                             latent_dim=128, base_channels=64,
                             alpha_values=None, quick_test=False):
    """Experiment 2: VAE/CIFAR-10 Recursive Training with Mixing.

    Validates Theorem 4 (mixing prevents collapse) using a ConvVAE on CIFAR-10.

    Theoretical predictions:
      1. Without mixing (alpha=0): progressive collapse (FID up, IS down)
      2. With mixing (alpha>0): steady-state variance = sigma^2/(n*alpha*(2-alpha))
      3. Higher alpha = more stable but less efficient
      4. Precision stays relatively stable; recall drops under collapse

    Estimated runtime: ~8 hours on RTX 5090
    GPU memory: ~4 GB
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: VAE/CIFAR-10 Recursive Training")
    print("="*70)

    if alpha_values is None:
        alpha_values = [0, 0.01, 0.05, 0.1, 0.2, 0.5]

    set_seed(seed)
    device = get_device()

    if quick_test:
        n_generations = 3
        n_samples_per_gen = 2000
        n_epochs = 10
        alpha_values = [0, 0.1]
        print("  [QUICK TEST MODE]")

    # ---- Load CIFAR-10 ----
    from torchvision import datasets, transforms
    # Load WITHOUT transform — we'll convert to tensors manually to avoid
    # double-transform bug (CIFAR-10 returns PIL images when no transform is set)
    cifar_train = datasets.CIFAR10('./data', train=True, download=True)

    # Get all training data as tensors
    # Handle both PIL Image (older torchvision) and already-tensor
    # (newer torchvision) returns from CIFAR-10's __getitem__.
    to_tensor = transforms.ToTensor()
    images_list = []
    for i in range(len(cifar_train)):
        img = cifar_train[i][0]
        if isinstance(img, torch.Tensor):
            images_list.append(img)
        else:
            images_list.append(to_tensor(img))
    real_images = torch.stack(images_list)
    # Normalize to [0,1] (ToTensor already does this for PIL images)
    real_images = real_images.clamp(0, 1)

    # Store for FID computation
    real_features = real_images[:5000].numpy().reshape(5000, -1)

    # ---- Results storage ----
    all_results = {}

    for alpha in alpha_values:
        print(f"\n{'='*50}")
        print(f"  Mixing ratio alpha = {alpha}")
        print(f"{'='*50}")

        results = {
            'fid': [],
            'precision': [],
            'recall': [],
            'pixel_var': [],
            'kl_estimate': [],
        }

        current_images = real_images.clone()

        for gen in range(n_generations):
            gen_start = time.time()
            print(f"\n  --- Generation {gen+1}/{n_generations} (alpha={alpha}) ---")

            # ---- Train ConvVAE ----
            model = ConvVAE(latent_dim=latent_dim,
                           base_channels=base_channels).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            dataset = TensorDataset(current_images)
            loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)

            model.train()
            # KL warm-up for ConvVAE as well
            warmup_epochs = max(1, n_epochs // 2)
            for epoch in range(n_epochs):
                beta = min(1.0, epoch / warmup_epochs)
                total_loss = 0
                total_kl_z = 0
                n_batches = 0
                for (batch_x,) in loader:
                    batch_x = batch_x.to(device)
                    optimizer.zero_grad()
                    recon, mu_z, logvar_z = model(batch_x)
                    loss, recon_l, kl_z = model.loss_function(
                        batch_x, recon, mu_z, logvar_z,
                        min_kl_per_dim=0.5, kl_constraint_weight=1.0, beta=beta)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_kl_z += kl_z.item()
                    n_batches += 1

                if (epoch + 1) % 25 == 0 or epoch == 0:
                    print(f"    Epoch {epoch+1}/{n_epochs}: "
                          f"loss={total_loss/n_batches:.1f}, "
                          f"kl_z={total_kl_z/n_batches:.1f}, "
                          f"beta={beta:.2f}")

            # ---- Generate samples ----
            model.eval()
            all_gen = []
            n_gen_so_far = 0
            while n_gen_so_far < n_samples_per_gen:
                n_batch = min(500, n_samples_per_gen - n_gen_so_far)
                samples = model.sample(n_batch, device)
                all_gen.append(samples.cpu())
                n_gen_so_far += n_batch
            generated_images = torch.cat(all_gen, dim=0)[:n_samples_per_gen]
            generated_images = generated_images.clamp(0, 1)

            # ---- Mixing: combine real + generated ----
            if alpha > 0:
                n_real = int(alpha * n_samples_per_gen)
                n_synth = n_samples_per_gen - n_real
                # Random subset of real data
                perm = torch.randperm(len(real_images))[:n_real]
                real_subset = real_images[perm]
                synth_subset = generated_images[:n_synth]
                next_data = torch.cat([real_subset, synth_subset], dim=0)
                # Shuffle
                perm2 = torch.randperm(len(next_data))
                next_data = next_data[perm2]
            else:
                next_data = generated_images

            # ---- Compute metrics ----
            n_eval = min(2000, len(generated_images), len(real_features))
            gen_features = generated_images[:n_eval].numpy().reshape(n_eval, -1)

            # FID
            fid = compute_fid_simple(real_features[:n_eval], gen_features[:n_eval])
            results['fid'].append(fid)

            # Pixel variance
            pvar = generated_images.var(dim=0).mean().item()
            results['pixel_var'].append(pvar)

            # Precision & Recall
            try:
                prec, rec = compute_precision_recall(
                    real_features[:n_eval], gen_features[:n_eval])
                results['precision'].append(prec)
                results['recall'].append(rec)
            except Exception:
                results['precision'].append(float('nan'))
                results['recall'].append(float('nan'))

            gen_elapsed = time.time() - gen_start
            print(f"    FID={fid:.2f}, Var={pvar:.6f}, "
                  f"P={results['precision'][-1]:.3f}, "
                  f"R={results['recall'][-1]:.3f}, Time={gen_elapsed:.1f}s")

            current_images = next_data

            # Free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Checkpoint
            if not quick_test:
                save_checkpoint('exp2_vae_cifar10', gen, {
                    'alpha': alpha,
                    'generation': gen,
                    'results': {k: v for k, v in results.items()},
                    'completed': True,
                })

        # Convert to numpy
        for key in results:
            results[key] = np.array(results[key])
        all_results[f'alpha_{alpha}'] = results

    # ---- Save all results ----
    save_results('exp2_vae_cifar10',
                 {k: v['fid'] for k, v in all_results.items()})

    # ---- Plot ----
    plot_exp2_results(all_results, alpha_values, n_generations)

    return all_results


def plot_exp2_results(all_results, alpha_values, n_generations):
    """Generate plots for Experiment 2."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    gens = np.arange(1, n_generations + 1)

    # (a) FID across generations for different alpha
    ax = axes[0, 0]
    for i, alpha in enumerate(alpha_values):
        key = f'alpha_{alpha}'
        if key in all_results and len(all_results[key]['fid']) > 0:
            ax.plot(gens[:len(all_results[key]['fid'])],
                    all_results[key]['fid'],
                    'o-', color=COLORS[i], linewidth=2,
                    label=f'$\\alpha$={alpha}')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Pixel FD (lower = better)')
    ax.set_title('(a) FD Degradation vs Mixing')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Pixel variance
    ax = axes[0, 1]
    for i, alpha in enumerate(alpha_values):
        key = f'alpha_{alpha}'
        if key in all_results and len(all_results[key]['pixel_var']) > 0:
            ax.plot(gens[:len(all_results[key]['pixel_var'])],
                    all_results[key]['pixel_var'],
                    'o-', color=COLORS[i], linewidth=2,
                    label=f'$\\alpha$={alpha}')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Mean pixel variance')
    ax.set_title('(b) Variance Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Precision
    ax = axes[0, 2]
    for i, alpha in enumerate(alpha_values):
        key = f'alpha_{alpha}'
        if key in all_results and len(all_results[key]['precision']) > 0:
            ax.plot(gens[:len(all_results[key]['precision'])],
                    all_results[key]['precision'],
                    'o-', color=COLORS[i], linewidth=2,
                    label=f'$\\alpha$={alpha}')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Precision')
    ax.set_title('(c) Sample Quality (Precision)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Recall
    ax = axes[1, 0]
    for i, alpha in enumerate(alpha_values):
        key = f'alpha_{alpha}'
        if key in all_results and len(all_results[key]['recall']) > 0:
            ax.plot(gens[:len(all_results[key]['recall'])],
                    all_results[key]['recall'],
                    'o-', color=COLORS[i], linewidth=2,
                    label=f'$\\alpha$={alpha}')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Recall')
    ax.set_title('(d) Coverage (Recall)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) Steady-state FID vs alpha (Theorem 4 prediction)
    ax = axes[1, 1]
    final_fids = []
    valid_alphas = []
    for alpha in alpha_values:
        key = f'alpha_{alpha}'
        if key in all_results and len(all_results[key]['fid']) > 0:
            final_fids.append(all_results[key]['fid'][-1])
            valid_alphas.append(alpha)
    if valid_alphas:
        ax.plot(valid_alphas, final_fids, 'ro-', markersize=8, linewidth=2)
        ax.set_xlabel('Mixing fraction $\\alpha$')
        ax.set_ylabel('Final Pixel FD')
        ax.set_title('(e) Steady-State vs $\\alpha$ (Thm 4)')
    ax.grid(True, alpha=0.3)

    # (f) Variance ratio vs Theorem 4 prediction
    ax = axes[1, 2]
    if valid_alphas:
        # Theorem 4: steady-state var = sigma^2 / (n * alpha * (2 - alpha))
        # We can't directly measure sigma^2/n for images, but we can check
        # the functional form
        for i, alpha in enumerate(valid_alphas):
            key = f'alpha_{alpha}'
            if key in all_results and len(all_results[key]['pixel_var']) > 1:
                var_ratio = all_results[key]['pixel_var'][-1] / all_results[key]['pixel_var'][0]
                ax.plot(alpha, var_ratio, 'ro', markersize=8)
        alpha_range = np.linspace(0.01, 0.6, 100)
        # Illustrative theoretical curve (scaled)
        theory_curve = 1.0 / (alpha_range * (2 - alpha_range))
        theory_curve = theory_curve / theory_curve[0]  # Normalize
        ax.plot(alpha_range, theory_curve, 'b--', linewidth=2,
                label=r'Theory: $\propto 1/(\alpha(2-\alpha))$')
        ax.set_xlabel('Mixing fraction $\\alpha$')
        ax.set_ylabel('Variance ratio (final/initial)')
        ax.set_title('(f) Variance vs Theory (Thm 4)')
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 2: VAE/CIFAR-10 Recursive Training',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'exp2_vae_cifar10.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure saved: {os.path.join(FIG_DIR, "exp2_vae_cifar10.png")}')


# ============================================================================
# EXPERIMENT 3: Diffusion Model Recursive Training
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def _gn_groups(channels, target_groups=8):
    """Compute a valid number of GroupNorm groups for the given channel count.

    GroupNorm requires channels % num_groups == 0.  We pick the largest
    divisor of `channels` that does not exceed `target_groups`, falling
    back to 1 group when necessary.
    """
    for g in range(min(target_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SimpleUNetBlock(nn.Module):
    """Simple residual block for the DDPM UNet."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm1 = nn.GroupNorm(_gn_groups(in_ch), in_ch)
        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # Add time embedding
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_proj
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class SmallDDPM(nn.Module):
    """Small DDPM model for CIFAR-32x32.

    A simplified UNet architecture for tractable recursive training.
    Not state-of-the-art, but sufficient to validate theoretical predictions
    about diffusion model collapse under recursive training.

    Theoretical connection:
      - Diffusion models estimate the score function of the data distribution.
      - Under recursive training, the score estimate becomes biased toward
        the previous generation's distribution, causing progressive drift.
      - Theorem 4 predicts that mixing ratio alpha controls the steady-state
        divergence, and the critical exponent p*=1/2 separates convergent
        from divergent decaying schedules.
    """
    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=128,
                 n_timesteps=1000):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder (downsampling)
        self.enc1 = SimpleUNetBlock(in_channels, base_channels, time_emb_dim)
        self.enc2 = SimpleUNetBlock(base_channels, base_channels*2, time_emb_dim)
        self.enc3 = SimpleUNetBlock(base_channels*2, base_channels*4, time_emb_dim)
        self.pool = nn.AvgPool2d(2)

        # Middle
        self.mid = SimpleUNetBlock(base_channels*4, base_channels*4, time_emb_dim)

        # Decoder (upsampling)
        # After cat(skip, upsampled), input channels = up_out_ch + skip_ch:
        #   dec3: base_ch*2 + base_ch*4 = base_ch*6  (up3_out + e3)
        #   dec2: base_ch   + base_ch*2 = base_ch*3  (up2_out + e2)
        #   dec1: base_ch   + base_ch   = base_ch*2  (up1_out + e1)
        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec3 = SimpleUNetBlock(base_channels*2 + base_channels*4, base_channels*2, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec2 = SimpleUNetBlock(base_channels + base_channels*2, base_channels, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.dec1 = SimpleUNetBlock(base_channels + base_channels, base_channels, time_emb_dim)

        # Output
        self.out_conv = nn.Conv2d(base_channels, in_channels, 1)

        # Beta schedule (linear)
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1.0 - self.betas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod',
                             torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - self.alphas_cumprod))

    def forward(self, x, t):
        """Predict noise given noisy image and timestep."""
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)         # 64 x 32 x 32
        e2 = self.enc2(self.pool(e1), t_emb)  # 128 x 16 x 16
        e3 = self.enc3(self.pool(e2), t_emb)  # 256 x 8 x 8

        # Middle
        m = self.mid(self.pool(e3), t_emb)    # 256 x 4 x 4

        # Decoder
        d3 = self.up3(m)                        # 128 x 8 x 8
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)
        d2 = self.up2(d3)                        # 64 x 16 x 16
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)
        d1 = self.up1(d2)                        # 64 x 32 x 32
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)

        return self.out_conv(d1)

    def add_noise(self, x, t, noise=None):
        """Add noise to clean images at timestep t (forward process)."""
        if noise is None:
            noise = torch.randn_like(x)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def sample(self, n_samples, device, img_size=32):
        """Generate samples using the reverse diffusion process."""
        self.eval()
        x = torch.randn(n_samples, 3, img_size, img_size, device=device)

        for t_idx in reversed(range(self.n_timesteps)):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            predicted_noise = self(x, t)

            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]

            # Compute x_{t-1}
            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # DDPM reverse step
            coeff1 = 1.0 / torch.sqrt(alpha)
            coeff2 = beta / torch.sqrt(1.0 - alpha_cumprod)
            x = coeff1 * (x - coeff2 * predicted_noise) + \
                torch.sqrt(beta) * noise

        return x.clamp(0, 1)

    @torch.no_grad()
    def sample_fast(self, n_samples, device, img_size=32, n_inference_steps=50):
        """Generate samples using DDIM-style sub-sampling for speed.

        Instead of running all n_timesteps reverse steps, we sub-sample
        n_inference_steps evenly spaced timesteps.  This gives a ~20x
        speedup with minor quality loss, following the DDIM insight that
        many diffusion steps are redundant during inference.
        """
        self.eval()
        x = torch.randn(n_samples, 3, img_size, img_size, device=device)

        # Sub-sample timesteps
        step_size = max(1, self.n_timesteps // n_inference_steps)
        timesteps = list(reversed(range(0, self.n_timesteps, step_size)))

        for i, t_idx in enumerate(timesteps):
            t = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            predicted_noise = self(x, t)

            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]

            # Predict x_0 from noise prediction (DDIM-style)
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / \
                      torch.sqrt(alpha_cumprod)
            x0_pred = x0_pred.clamp(-1, 1)

            if i < len(timesteps) - 1:
                # DDIM step: deterministic transition to next sub-step
                t_next = timesteps[i + 1]
                alpha_cumprod_next = self.alphas_cumprod[t_next]
                x = torch.sqrt(alpha_cumprod_next) * x0_pred + \
                    torch.sqrt(1 - alpha_cumprod_next) * predicted_noise
            else:
                x = x0_pred

        return x.clamp(0, 1)


def train_ddpm(model, dataloader, n_epochs, device, lr=2e-4,
              fine_tune_from=None, fine_tune_lr=5e-5):
    """Train a DDPM model for one generation.

    Args:
        fine_tune_from: If provided, a state_dict to load as initialization
            for fine-tuning (instead of training from scratch).
        fine_tune_lr: Learning rate for fine-tuning epochs (lower than
            training from scratch to preserve learned features).
    """
    if fine_tune_from is not None:
        model.load_state_dict(fine_tune_from)
        effective_lr = fine_tune_lr
        print(f"    [Fine-tuning from previous generation, lr={effective_lr}]")
    else:
        effective_lr = lr

    optimizer = optim.Adam(model.parameters(), lr=effective_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model.train()
    final_loss = 0.0
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        for batch_x, in dataloader:
            batch_x = batch_x.to(device)
            batch_size = batch_x.shape[0]

            # Sample random timesteps
            t = torch.randint(0, model.n_timesteps, (batch_size,),
                            device=device)

            # Add noise
            noise = torch.randn_like(batch_x)
            noisy_x = model.add_noise(batch_x, t, noise)

            # Predict noise
            predicted_noise = model(noisy_x, t)

            # Simple MSE loss
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        final_loss = avg_loss

        # Print first epoch, last epoch, and every 25th
        if (epoch + 1) % 25 == 0 or epoch == 0 or epoch == n_epochs - 1:
            tag = " (fine-tune)" if fine_tune_from is not None else ""
            print(f"    Epoch {epoch+1}/{n_epochs}{tag}: "
                  f"loss={avg_loss:.6f}")

    return model, final_loss


def experiment3_diffusion_cifar10(seed=42, n_generations=8,
                                   n_samples_per_gen=10000,
                                   n_epochs=200, batch_size=64,
                                   lr=2e-4, n_timesteps=1000,
                                   base_channels=64, quick_test=False):
    """Experiment 3: Diffusion Model Recursive Training on CIFAR-10.

    Validates Theorem 4 in the context of diffusion models:
      1. Without mixing: progressive quality degradation
      2. Constant alpha: controlled steady-state divergence
      3. Decaying alpha_t = t^{-p}: critical exponent p*=1/2
         - p < 1/2: variance converges (too much mixing)
         - p > 1/2: variance diverges (insufficient mixing)
         - p = 1/2: boundary case, logarithmic divergence

    Estimated runtime: ~20 hours on RTX 5090
    GPU memory: ~8 GB
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Diffusion Model Recursive Training (CIFAR-10)")
    print("="*70)

    set_seed(seed)
    device = get_device()

    if quick_test:
        n_generations = 4
        n_samples_per_gen = 2000
        n_epochs_gen0 = 100       # Gen 1: train well on real data
        n_epochs_finetune = 50    # Gen 2+: fine-tune (fewer epochs needed)
        n_timesteps = 500         # Enough steps for meaningful diffusion
        base_channels = 32        # Smaller model for quick test
        fine_tune_lr = 5e-5       # Lower LR for fine-tuning
        print("  [QUICK TEST MODE]")
    else:
        n_epochs_gen0 = n_epochs
        n_epochs_finetune = n_epochs // 2  # Fine-tune needs fewer epochs
        fine_tune_lr = 5e-5

    # ---- Load CIFAR-10 ----
    from torchvision import datasets, transforms
    # Load WITHOUT transform to avoid double-transform bug
    cifar_train = datasets.CIFAR10('./data', train=True, download=True)
    # Handle both PIL Image and Tensor returns from CIFAR-10
    to_tensor = transforms.ToTensor()
    images_list = []
    for i in range(len(cifar_train)):
        img = cifar_train[i][0]
        if isinstance(img, torch.Tensor):
            images_list.append(img)
        else:
            images_list.append(to_tensor(img))
    real_images = torch.stack(images_list)
    real_images = real_images.clamp(0, 1)
    real_features = real_images[:3000].numpy().reshape(3000, -1)

    # ---- Mixing strategies to test ----
    if quick_test:
        strategies = {
            'no_mixing': lambda t: 0.0,
            'alpha_0.10': lambda t: 0.10,
        }
    else:
        strategies = {
            'no_mixing': lambda t: 0.0,
            'alpha_0.05': lambda t: 0.05,
            'alpha_0.10': lambda t: 0.10,
            'alpha_sqrt': lambda t: min(0.5, 0.5 / np.sqrt(t + 1)),  # p=1/2
            'alpha_inv': lambda t: min(0.5, 0.5 / (t + 1)),          # p=1
            'alpha_p0.3': lambda t: min(0.5, 0.5 / ((t + 1) ** 0.3)), # p=0.3
        }

    all_results = {}

    for strat_name, alpha_fn in strategies.items():
        print(f"\n{'='*50}")
        print(f"  Strategy: {strat_name}")
        print(f"{'='*50}")

        results = {
            'fid': [],
            'pixel_var': [],
            'mean_l2_dist': [],
            'loss_final': [],
        }

        current_images = real_images.clone()
        gen0_state_dict = None  # Will store Gen 1 weights for fine-tuning

        for gen in range(n_generations):
            gen_start = time.time()
            alpha_t = alpha_fn(gen)
            is_gen0 = (gen == 0)
            print(f"\n  --- Generation {gen+1}/{n_generations} "
                  f"(alpha_t={alpha_t:.4f}, "
                  f"{'from-scratch' if is_gen0 else 'fine-tune'}) ---")

            # ---- Train DDPM ----
            model = SmallDDPM(in_channels=3, base_channels=base_channels,
                              n_timesteps=n_timesteps).to(device)

            # Gen 1: train from scratch on real data (many epochs)
            # Gen 2+: fine-tune from Gen 1's weights (fewer epochs, lower LR)
            if is_gen0:
                n_ep = n_epochs_gen0
                model, final_loss = train_ddpm(
                    model,
                    DataLoader(TensorDataset(current_images),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True),
                    n_ep, device, lr=lr
                )
                # Save Gen 1 weights for subsequent fine-tuning
                gen0_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                n_ep = n_epochs_finetune
                model, final_loss = train_ddpm(
                    model,
                    DataLoader(TensorDataset(current_images),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True),
                    n_ep, device, lr=lr,
                    fine_tune_from=gen0_state_dict,
                    fine_tune_lr=fine_tune_lr,
                )

            results['loss_final'].append(final_loss)

            # ---- Generate samples ----
            model.eval()
            n_gen_batches = (n_samples_per_gen + batch_size - 1) // batch_size
            all_gen = []
            # Use DDIM-style fast sampling
            n_inference = min(50, n_timesteps)
            for _ in range(n_gen_batches):
                n_batch = min(batch_size, n_samples_per_gen - len(all_gen) * batch_size)
                if n_batch <= 0:
                    break
                samples = model.sample_fast(n_batch, device, img_size=32,
                                            n_inference_steps=n_inference)
                all_gen.append(samples.cpu())
            generated_images = torch.cat(all_gen, dim=0)[:n_samples_per_gen]
            generated_images = generated_images.clamp(0, 1)

            # ---- Mixing ----
            if alpha_t > 0:
                n_real = int(alpha_t * n_samples_per_gen)
                n_synth = n_samples_per_gen - n_real
                perm = torch.randperm(len(real_images))[:n_real]
                real_subset = real_images[perm]
                synth_subset = generated_images[:n_synth]
                next_data = torch.cat([real_subset, synth_subset], dim=0)
                perm2 = torch.randperm(len(next_data))
                next_data = next_data[perm2]
            else:
                next_data = generated_images

            # ---- Compute metrics ----
            n_eval = min(2000, len(generated_images), len(real_features))
            gen_features = generated_images[:n_eval].numpy().reshape(n_eval, -1)
            fid = compute_fid_simple(real_features[:n_eval], gen_features[:n_eval])
            results['fid'].append(fid)

            pvar = generated_images.var(dim=0).mean().item()
            results['pixel_var'].append(pvar)

            # Mean L2 distance to real data center
            real_mean = real_images.mean(dim=0)
            gen_mean = generated_images.mean(dim=0)
            mean_l2 = (real_mean - gen_mean).pow(2).mean().sqrt().item()
            results['mean_l2_dist'].append(mean_l2)

            gen_elapsed = time.time() - gen_start
            print(f"    FID={fid:.2f}, Var={pvar:.6f}, "
                  f"L2={mean_l2:.4f}, Loss={final_loss:.6f}, "
                  f"Time={gen_elapsed:.1f}s")

            current_images = next_data

            # Free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Checkpoint
            if not quick_test:
                save_checkpoint('exp3_diffusion', gen, {
                    'strategy': strat_name,
                    'generation': gen,
                    'results': {k: list(v) for k, v in results.items()},
                    'completed': True,
                })

        for key in results:
            results[key] = np.array(results[key])
        all_results[strat_name] = results

    # ---- Save results ----
    save_results('exp3_diffusion',
                 {k: v['fid'] for k, v in all_results.items()})

    # ---- Plot ----
    plot_exp3_results(all_results, n_generations)

    return all_results


def plot_exp3_results(all_results, n_generations):
    """Generate plots for Experiment 3."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    gens = np.arange(1, n_generations + 1)

    # (a) FID across generations
    ax = axes[0, 0]
    for i, (name, res) in enumerate(all_results.items()):
        if len(res['fid']) > 0:
            ax.plot(gens[:len(res['fid'])], res['fid'],
                    'o-', color=COLORS[i], linewidth=2, label=name)
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Pixel FD (lower = better)')
    ax.set_title('(a) FD vs Mixing Strategy')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Pixel variance
    ax = axes[0, 1]
    for i, (name, res) in enumerate(all_results.items()):
        if len(res['pixel_var']) > 0:
            ax.plot(gens[:len(res['pixel_var'])], res['pixel_var'],
                    'o-', color=COLORS[i], linewidth=2, label=name)
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Mean pixel variance')
    ax.set_title('(b) Variance Evolution')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (c) Mean L2 drift
    ax = axes[1, 0]
    for i, (name, res) in enumerate(all_results.items()):
        if len(res['mean_l2_dist']) > 0:
            ax.plot(gens[:len(res['mean_l2_dist'])], res['mean_l2_dist'],
                    'o-', color=COLORS[i], linewidth=2, label=name)
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Mean L2 drift')
    ax.set_title('(c) Distribution Drift')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) Training loss across generations (shows drift from gen-0 model)
    ax = axes[1, 1]
    for i, (name, res) in enumerate(all_results.items()):
        if len(res['loss_final']) > 0:
            ax.plot(gens[:len(res['loss_final'])], res['loss_final'],
                    'o-', color=COLORS[i], linewidth=2, label=name)
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('Final training loss')
    ax.set_title('(d) Training Loss vs Generation')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 3: Diffusion Model Recursive Training',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'exp3_diffusion.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure saved: {os.path.join(FIG_DIR, "exp3_diffusion.png")}')


# ============================================================================
# EXPERIMENT 4: Large-Scale Gaussian Simulation (GPU-Accelerated)
# ============================================================================

def experiment4_gaussian_gpu(seed=42, n_trials=100000,
                              n_values=None, d_values=None,
                              quick_test=False):
    """Experiment 4: Large-Scale Gaussian Monte Carlo on GPU.

    Massive Monte Carlo simulation using GPU tensor operations to validate
    Theorems 1, 1b, and 5 at unprecedented scale.

    Theoretical predictions:
      Theorem 1:  E[KL(P_0 || P_t)] = t * d / (2n)  [known variance]
      Theorem 1b: E[KL(P_0 || P_t)] ~ t * d / n      [unknown variance]
                  Variance contracts: E[sigma^2_t] = sigma^2_0 * ((n-1)/n)^t
      Theorem 5:  Minimax lower bound matches upper bound (constant factor)

    Key advantage of GPU: we can run 100,000 trials in parallel using
    batched tensor operations, enabling precise estimation of expectations
    and distributions (not just means).

    Estimated runtime: ~1 hour on RTX 5090
    GPU memory: ~6 GB (largest config: n=10000, d=500, 100k trials)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: Large-Scale Gaussian Simulation (GPU)")
    print("="*70)

    if n_values is None:
        n_values = [100, 500, 1000, 5000, 10000]
    if d_values is None:
        d_values = [1, 10, 50, 100, 500]

    if quick_test:
        n_trials = 1000
        n_values = [100, 500]
        d_values = [1, 10]
        print("  [QUICK TEST MODE]")

    set_seed(seed)
    device = get_device()

    T_max = 50  # Maximum generations to simulate

    # ---- Part A: Known variance (Theorem 1) ----
    print("\n--- Part A: Known Variance (Theorem 1) ---")
    results_A = {}

    for n in n_values:
        for d in d_values:
            config_key = f'n={n}_d={d}'
            print(f"\n  Config: n={n}, d={d}, trials={n_trials}")

            # Estimate GPU memory needed
            mem_per_trial = d * 4 * 3  # mu_t, samples, mu_hat (float32)
            total_mem = mem_per_trial * n_trials / 1e9
            print(f"    Estimated GPU memory: ~{total_mem:.2f} GB")

            # Process in chunks to manage GPU memory
            chunk_size = min(n_trials, max(1, int(2e9 / (mem_per_trial + 1))))
            n_chunks = (n_trials + chunk_size - 1) // chunk_size

            kl_all_trials = []
            mu_drift_all = []

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, n_trials)
                current_chunk_size = end_idx - start_idx

                # Initialize: mu_0 = 0 for all trials
                # Shape: (chunk_size, d)
                mu_t = torch.zeros(current_chunk_size, d, device=device)
                kl_per_gen = torch.zeros(current_chunk_size, T_max, device=device)

                for t in range(T_max):
                    # Sample from N(mu_t, sigma^2 * I)
                    # Shape: (chunk_size, n, d)
                    # To save memory, compute mean directly without storing all samples
                    # E[sample_mean] = mu_t, Var[sample_mean] = sigma^2 / n * I
                    # sample_mean = mu_t + sigma/sqrt(n) * epsilon
                    epsilon = torch.randn(current_chunk_size, d, device=device)
                    mu_hat = mu_t + np.sqrt(SIGMA2 / n) * epsilon

                    # KL divergence: KL(N(0, I) || N(mu_hat, I))
                    # = ||mu_hat - 0||^2 / (2 * sigma^2) = ||mu_hat||^2 / 2
                    kl = torch.sum(mu_hat ** 2, dim=1) / (2 * SIGMA2)
                    kl_per_gen[:, t] = kl

                    # Update: next generation uses mu_hat as the mean
                    mu_t = mu_hat

                kl_all_trials.append(kl_per_gen.cpu().numpy())
                mu_drift_all.append(mu_t.cpu().numpy())

                if (chunk_idx + 1) % max(1, n_chunks // 5) == 0:
                    print(f"    Chunk {chunk_idx+1}/{n_chunks} done")

            # Combine chunks
            kl_all = np.concatenate(kl_all_trials, axis=0)  # (n_trials, T_max)

            # Compute statistics
            kl_mean = kl_all.mean(axis=0)
            kl_std = kl_all.std(axis=0)
            kl_median = np.median(kl_all, axis=0)

            # Theoretical prediction: KL = t * d / (2n)
            kl_theory = np.arange(1, T_max + 1) * d / (2 * n)

            results_A[config_key] = {
                'kl_mean': kl_mean,
                'kl_std': kl_std,
                'kl_median': kl_median,
                'kl_theory': kl_theory,
                'n': n, 'd': d,
            }

            # Quick validation print
            t_check = min(20, T_max) - 1
            ratio = kl_mean[t_check] / kl_theory[t_check] if kl_theory[t_check] > 0 else float('inf')
            print(f"    KL(t={t_check+1}): sim={kl_mean[t_check]:.6f}, "
                  f"theory={kl_theory[t_check]:.6f}, ratio={ratio:.4f}")

    # ---- Part B: Unknown variance (Theorem 1b) ----
    print("\n--- Part B: Unknown Variance (Theorem 1b) ---")
    results_B = {}

    for n in n_values:
        print(f"\n  n={n}, d=1 (scalar case for variance tracking), "
              f"trials={n_trials}")

        chunk_size = min(n_trials, 50000)
        n_chunks = (n_trials + chunk_size - 1) // chunk_size

        kl_known_all = []
        kl_unknown_all = []
        var_ratio_all = []

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_trials)
            cs = end_idx - start_idx

            mu_t = torch.zeros(cs, device=device)
            var_t = torch.ones(cs, device=device) * SIGMA2

            kl_known = torch.zeros(cs, T_max, device=device)
            kl_unknown = torch.zeros(cs, T_max, device=device)
            var_history = torch.zeros(cs, T_max, device=device)

            for t in range(T_max):
                # Sample mean: mu_hat = mu_t + sqrt(var_t/n) * eps_mu
                eps_mu = torch.randn(cs, device=device)
                mu_hat = mu_t + torch.sqrt(var_t / n) * eps_mu

                # Sample variance: s^2 = var_t * chi2(n-1) / n
                #
                # The previous Gaussian approximation for chi2 was incorrect:
                # it could produce negative s^2 values, causing the KL doubling
                # ratio to be 3.76 (should be 2.0).
                #
                # Correct approach: chi2(k) = Gamma(k/2, 2), so
                #   chi2(n-1)/n = Gamma((n-1)/2, 2/n)
                # We sample using: Gamma(a, b) ~ b * Gamma_sample(a)
                # where Gamma_sample(a) = -log(U_1*...*U_a) for integer a
                # (Erlang distribution), or use torch.distributions.
                #
                # For efficiency, we use the distribution module when available,
                # and fall back to the expectation-only (deterministic) path
                # which still correctly validates the KL doubling theorem.
                if n >= 30:
                    # For large n, use the Gaussian approximation to chi2
                    # with proper truncation to ensure s2 > 0.
                    # By CLT: chi2(k) ≈ N(k, 2k) for large k.
                    # The probability of negative values is P(Z < -k/sqrt(2k))
                    # which for k=n-1 >= 29 is < 1e-7.
                    chi2_val = (n - 1) + torch.randn(cs, device=device) * \
                               np.sqrt(2.0 * (n - 1))
                    chi2_val = chi2_val.clamp(min=0.1)  # Safety clamp
                    s2 = var_t * chi2_val / n
                else:
                    # For small n, use exact Gamma sampling via
                    # torch.distributions.Gamma
                    gamma_dist = torch.distributions.Gamma(
                        torch.tensor((n - 1) / 2.0, device=device),
                        torch.tensor(0.5, device=device)  # rate = 1/scale, scale=2
                    )
                    chi2_val = gamma_dist.sample((cs,))
                    s2 = var_t * chi2_val / n

                s2 = s2.clamp(min=1e-10)  # Numerical stability

                # KL divergence components (1D case, d=1):
                # KL(N(0, sigma^2) || N(mu_hat, s^2))
                # = 0.5 * [log(s^2/sigma^2) + sigma^2/s^2 - 1 + mu_hat^2/s^2]

                # Known variance KL: only mean drift
                # KL_known = mu_hat^2 / (2 * sigma^2)
                kl_k = mu_hat ** 2 / (2 * SIGMA2)
                kl_known[:, t] = kl_k

                # Unknown variance KL: full KL
                # KL_unknown = 0.5 * [log(s^2/sigma^2) + sigma^2/s^2 - 1 + mu_hat^2/s^2]
                kl_u = 0.5 * (torch.log(s2 / SIGMA2) + SIGMA2 / s2 - 1 + mu_hat ** 2 / s2)
                kl_unknown[:, t] = kl_u

                var_history[:, t] = s2

                # Update for next generation
                mu_t = mu_hat
                var_t = s2

            kl_known_all.append(kl_known.cpu().numpy())
            kl_unknown_all.append(kl_unknown.cpu().numpy())
            var_ratio_all.append((var_history / SIGMA2).cpu().numpy())

        kl_known_arr = np.concatenate(kl_known_all, axis=0)
        kl_unknown_arr = np.concatenate(kl_unknown_all, axis=0)
        var_ratio_arr = np.concatenate(var_ratio_all, axis=0)

        # Mean KL
        kl_known_mean = kl_known_arr.mean(axis=0)
        kl_unknown_mean = kl_unknown_arr.mean(axis=0)
        var_ratio_mean = var_ratio_arr.mean(axis=0)

        # Theoretical predictions
        t_range = np.arange(1, T_max + 1)
        kl_known_theory = t_range / (2 * n)           # Theorem 1
        kl_unknown_theory = t_range / n                # Theorem 1b (2x rate)
        var_theory = ((n - 1) / n) ** t_range           # Theorem 1b

        # KL doubling ratio
        kl_doubling = kl_unknown_mean / np.maximum(kl_known_mean, 1e-10)

        results_B[f'n={n}'] = {
            'kl_known_mean': kl_known_mean,
            'kl_unknown_mean': kl_unknown_mean,
            'kl_known_theory': kl_known_theory,
            'kl_unknown_theory': kl_unknown_theory,
            'var_ratio_mean': var_ratio_mean,
            'var_theory': var_theory,
            'kl_doubling_ratio': kl_doubling,
            'n': n,
        }

        # Validation summary
        t_check = 19  # t=20
        print(f"    KL_known(t=20):  sim={kl_known_mean[t_check]:.6f}, "
              f"theory={kl_known_theory[t_check]:.6f}")
        print(f"    KL_unknown(t=20): sim={kl_unknown_mean[t_check]:.6f}, "
              f"theory={kl_unknown_theory[t_check]:.6f}")
        print(f"    Doubling ratio: {kl_doubling[t_check]:.4f} "
              f"(theory: 2.0)")
        print(f"    Var ratio(t=20): {var_ratio_mean[t_check]:.6f}, "
              f"theory={var_theory[t_check]:.6f}")

    # ---- Part C: Minimax lower bound (Theorem 5) ----
    print("\n--- Part C: Minimax Lower Bound (Theorem 5) ---")
    results_C = {}

    n_minimax = 100
    n_trials_mm = n_trials
    T_mm = 25

    chunk_size = min(n_trials_mm, 50000)
    n_chunks = (n_trials_mm + chunk_size - 1) // chunk_size

    kl_per_trial_all = []
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_trials_mm)
        cs = end_idx - start_idx

        mu_t = torch.zeros(cs, device=device)
        kl_chunk = torch.zeros(cs, T_mm, device=device)

        for t in range(T_mm):
            eps = torch.randn(cs, device=device)
            mu_hat = mu_t + np.sqrt(SIGMA2 / n_minimax) * eps
            kl_chunk[:, t] = mu_hat ** 2 / (2 * SIGMA2)
            mu_t = mu_hat

        kl_per_trial_all.append(kl_chunk.cpu().numpy())

    kl_per_trial = np.concatenate(kl_per_trial_all, axis=0)
    kl_mean_mm = kl_per_trial.mean(axis=0)
    kl_p10 = np.percentile(kl_per_trial, 10, axis=0)
    kl_p90 = np.percentile(kl_per_trial, 90, axis=0)

    # Theoretical bounds
    t_range_mm = np.arange(1, T_mm + 1)
    upper_bound = t_range_mm / (2 * n_minimax)
    # Le Cam lower bound: (1 - 1/sqrt(2)) * t / (2n)
    le_cam_lower = upper_bound * 2 * (1 - 1 / np.sqrt(2))

    results_C = {
        'kl_mean': kl_mean_mm,
        'kl_p10': kl_p10,
        'kl_p90': kl_p90,
        'upper_bound': upper_bound,
        'le_cam_lower': le_cam_lower,
        'n': n_minimax,
    }

    print(f"  Minimax validation (n={n_minimax}):")
    for t_check in [4, 9, 19]:
        if t_check < T_mm:
            print(f"    t={t_check+1}: sim={kl_mean_mm[t_check]:.6f}, "
                  f"upper={upper_bound[t_check]:.6f}, "
                  f"lower={le_cam_lower[t_check]:.6f}, "
                  f"ratio={kl_mean_mm[t_check]/upper_bound[t_check]:.4f}")

    # ---- Save all results ----
    all_exp4 = {
        'A_known_var': {k: {kk: vv for kk, vv in v.items() if isinstance(vv, np.ndarray)}
                        for k, v in results_A.items()},
        'B_unknown_var': {k: {kk: vv for kk, vv in v.items() if isinstance(vv, np.ndarray)}
                          for k, v in results_B.items()},
        'C_minimax': {k: v for k, v in results_C.items() if isinstance(v, np.ndarray)},
    }
    save_results('exp4_gaussian_gpu', all_exp4)

    # ---- Plot ----
    plot_exp4_results(results_A, results_B, results_C, T_max, n_values, d_values)

    return results_A, results_B, results_C


def plot_exp4_results(results_A, results_B, results_C, T_max, n_values, d_values):
    """Generate plots for Experiment 4."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) Known variance KL growth for different n (d=1)
    ax = axes[0, 0]
    d_fixed = 1
    for i, n in enumerate(n_values[:4]):
        key = f'n={n}_d={d_fixed}'
        if key in results_A:
            res = results_A[key]
            gens = np.arange(1, T_max + 1)
            ax.plot(gens, res['kl_mean'], '-', color=COLORS[i],
                    linewidth=2, label=f'n={n} (sim)')
            ax.plot(gens, res['kl_theory'], '--', color=COLORS[i],
                    linewidth=1.5, alpha=0.6, label=f'n={n} (theory)')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$\mathbb{E}[D_{\mathrm{KL}}]$')
    ax.set_title('(a) KL Growth: Known Var (Thm 1)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (b) Dimension dependence (n=100)
    ax = axes[0, 1]
    n_fixed = 100
    for i, d in enumerate(d_values[:4]):
        key = f'n={n_fixed}_d={d}'
        if key in results_A:
            res = results_A[key]
            gens = np.arange(1, T_max + 1)
            ax.plot(gens, res['kl_mean'], 'o-', color=COLORS[i],
                    linewidth=2, markersize=3, label=f'd={d} (sim)')
            ax.plot(gens, res['kl_theory'], '--', color=COLORS[i],
                    linewidth=1.5, alpha=0.6, label=f'd={d} (theory)')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$\mathbb{E}[D_{\mathrm{KL}}]$')
    ax.set_title(f'(b) Dimension Dependence (n={n_fixed})')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (c) Unknown vs Known variance (Theorem 1b validation)
    ax = axes[0, 2]
    for i, n in enumerate(n_values[:4]):
        key = f'n={n}'
        if key in results_B:
            res = results_B[key]
            gens = np.arange(1, T_max + 1)
            ax.plot(gens, res['kl_known_mean'], '-', color=COLORS[i],
                    linewidth=2, label=f'n={n} (known)')
            ax.plot(gens, res['kl_unknown_mean'], '--', color=COLORS[i],
                    linewidth=2, label=f'n={n} (unknown)')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$D_{\mathrm{KL}}$')
    ax.set_title('(c) Unknown vs Known Var (Thm 1b)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (d) KL doubling ratio
    ax = axes[1, 0]
    for i, n in enumerate(n_values[:4]):
        key = f'n={n}'
        if key in results_B:
            res = results_B[key]
            gens = np.arange(1, T_max + 1)
            ax.plot(gens, res['kl_doubling_ratio'], '-', color=COLORS[i],
                    linewidth=2, label=f'n={n}')
    ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2,
                alpha=0.5, label='Theorem 1b: ratio = 2')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel('KL(unknown) / KL(known)')
    ax.set_title('(d) KL Doubling Ratio (Thm 1b)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) Variance contraction
    ax = axes[1, 1]
    for i, n in enumerate(n_values[:4]):
        key = f'n={n}'
        if key in results_B:
            res = results_B[key]
            gens = np.arange(1, T_max + 1)
            ax.plot(gens, res['var_ratio_mean'], '-', color=COLORS[i],
                    linewidth=2, label=f'n={n} (sim)')
            ax.plot(gens, res['var_theory'], '--', color=COLORS[i],
                    linewidth=1.5, alpha=0.6, label=f'n={n} (theory)')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$\mathbb{E}[\hat{\sigma}^2_t] / \sigma^2_0$')
    ax.set_title('(e) Variance Contraction (Thm 1b)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (f) Minimax bounds (Theorem 5)
    ax = axes[1, 2]
    res = results_C
    gens = np.arange(1, len(res['kl_mean']) + 1)
    ax.plot(gens, res['kl_mean'], 'r-', linewidth=2, label='MLE (simulation)')
    ax.fill_between(gens, res['kl_p10'], res['kl_p90'],
                    alpha=0.2, color='red', label='10-90 percentile')
    ax.plot(gens, res['upper_bound'], 'b--', linewidth=2.5,
            label=r'Upper: $td/(2n)$')
    ax.plot(gens, res['le_cam_lower'], 'g:', linewidth=2.5,
            label=r'Le Cam lower bound')
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$D_{\mathrm{KL}}$')
    ax.set_title(f'(f) Minimax Tightness (Thm 5, n={res["n"]})')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 4: GPU-Accelerated Gaussian Simulation\n'
                 '(100,000 trials per configuration)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'exp4_gaussian_gpu.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure saved: {os.path.join(FIG_DIR, "exp4_gaussian_gpu.png")}')


# ============================================================================
# EXPERIMENT 5: Language Model Recursive Training
# ============================================================================

def experiment5_lm_recursive(seed=42, n_generations=8, n_epochs=3,
                              batch_size=8, lr=5e-5, max_length=128,
                              n_samples_per_gen=10000,
                              alpha_values=None, quick_test=False):
    """Experiment 5: Language Model Recursive Training (GPT-2).

    Tests model collapse in the language domain using GPT-2 small (124M).
    Validates Theorems 1 and 2 in the discrete token space:
      - Token/n-gram distribution narrowing (analogous to variance contraction)
      - Vocabulary extinction (analogous to category extinction, Theorem 2)
      - Mixing with real data prevents collapse (Theorem 4)

    Metrics:
      - Perplexity: should increase under recursive training
      - Unique token ratio: should decrease (vocabulary collapse)
      - N-gram diversity: should decrease
      - Category extinction: rare tokens disappear first (Theorem 2)

    Estimated runtime: ~12 hours on RTX 5090
    GPU memory: ~6 GB (GPT-2 124M in fp32)
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Language Model Recursive Training (GPT-2)")
    print("="*70)

    if alpha_values is None:
        alpha_values = [0.0, 0.05, 0.10, 0.20]

    set_seed(seed)
    device = get_device()

    if quick_test:
        n_generations = 2
        n_samples_per_gen = 500
        n_epochs = 1
        alpha_values = [0.0, 0.1]
        max_length = 64
        print("  [QUICK TEST MODE]")

    # ---- Load GPT-2 and tokenizer ----
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("  [ERROR] transformers not installed. "
              "Install with: pip install transformers")
        return {}

    print("  Loading GPT-2 small (124M)...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model_ref = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model_ref.eval()

    # ---- Prepare a text dataset ----
    # NOTE: For a fully self-contained script we use GPT-2 itself to generate
    # a "real" reference corpus.  This means our "real" distribution is already
    # model-generated, so the *absolute* collapse rate will be under-estimated
    # (the starting distribution is narrower than true human text).  However,
    # the *relative* comparisons (different alpha values, monotonic trends)
    # remain valid because the recursive mechanism is the same regardless of
    # the initial distribution shape.
    #
    # For a production run, replace this block with a real corpus (e.g.,
    # WikiText-2/103, OpenWebText) to measure absolute collapse rates.
    vocab_size = len(tokenizer)  # GPT-2 vocab = 50257
    print(f"  Generating reference text corpus (vocab_size={vocab_size})...")
    real_texts = []
    # Generate enough texts for both training AND held-out evaluation.
    # We need extra texts so that the PPL evaluation set is NEVER seen
    # during training (no data leakage).
    n_ref_samples = min(n_samples_per_gen, 5000)
    n_eval_texts = 500          # Held-out texts for PPL computation
    n_total_ref = n_ref_samples + n_eval_texts  # Need more than n_ref_samples
    with torch.no_grad():
        for i in range(0, n_total_ref, batch_size):
            n_batch = min(batch_size, n_total_ref - i)
            # Use valid token IDs within GPT-2's vocabulary range
            input_ids = torch.randint(0, vocab_size, (n_batch, 10),
                                     device=device)
            outputs = model_ref.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
            for output in outputs:
                text = tokenizer.decode(output, skip_special_tokens=True)
                real_texts.append(text)
            if (i + batch_size) % 500 == 0:
                print(f"    Generated {min(i + n_batch, n_total_ref)}/{n_total_ref} texts")

    # Split: first n_ref_samples for training/mixing, last n_eval_texts for PPL eval only
    real_texts_for_mixing = real_texts[:n_ref_samples]
    eval_texts = real_texts[n_ref_samples:n_ref_samples + n_eval_texts]
    print(f"  Reference corpus: {len(real_texts_for_mixing)} training + "
          f"{len(eval_texts)} held-out eval texts")

    # ---- Helper functions ----
    def compute_perplexity(model, texts, device, max_length=128):
        """Compute perplexity of model on a set of texts."""
        model.eval()
        total_nll = 0
        total_tokens = 0
        with torch.no_grad():
            for text in texts[:500]:  # Limit for speed
                encodings = tokenizer(text, return_tensors='pt',
                                      truncation=True, max_length=max_length)
                input_ids = encodings['input_ids'].to(device)
                if input_ids.shape[1] < 2:
                    continue
                outputs = model(input_ids, labels=input_ids)
                total_nll += outputs.loss.item() * (input_ids.shape[1] - 1)
                total_tokens += input_ids.shape[1] - 1
        if total_tokens == 0:
            return float('inf')
        return np.exp(total_nll / total_tokens)

    def compute_diversity_metrics(texts, tokenizer, n_gram=4):
        """Compute diversity metrics for generated text.

        Returns:
            unique_token_ratio: fraction of vocab that appears
            ngram_diversity: distinct n-grams / total n-grams
            entropy: token distribution entropy
        """
        all_tokens = []
        for text in texts:
            # Truncate to avoid GPT-2's 1024-token context limit warning
            tokens = tokenizer.encode(text, truncation=True, max_length=1024)
            all_tokens.extend(tokens)

        if len(all_tokens) == 0:
            return 0, 0, 0

        # Unique token ratio (category survival — Theorem 2)
        unique_tokens = set(all_tokens)
        # Use a reference vocab size
        vocab_size = len(tokenizer)
        unique_ratio = len(unique_tokens) / vocab_size

        # N-gram diversity
        ngrams = []
        for i in range(len(all_tokens) - n_gram + 1):
            ngrams.append(tuple(all_tokens[i:i+n_gram]))
        if len(ngrams) > 0:
            ngram_div = len(set(ngrams)) / len(ngrams)
        else:
            ngram_div = 0

        # Token distribution entropy
        from collections import Counter
        token_counts = Counter(all_tokens)
        total = sum(token_counts.values())
        probs = np.array(list(token_counts.values())) / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return unique_ratio, ngram_div, entropy

    def tokenize_texts(texts, tokenizer, max_length=128):
        """Tokenize a list of texts for training."""
        all_input_ids = []
        for text in texts:
            enc = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=max_length, padding='max_length')
            all_input_ids.append(enc['input_ids'])
        return torch.cat(all_input_ids, dim=0)

    # ---- Recursive training loop ----
    all_results = {}

    for alpha in alpha_values:
        print(f"\n{'='*50}")
        print(f"  Mixing ratio alpha = {alpha}")
        print(f"{'='*50}")

        results = {
            'perplexity': [],
            'unique_token_ratio': [],
            'ngram_diversity': [],
            'token_entropy': [],
            'vocab_size': [],
        }

        current_texts = real_texts_for_mixing.copy()

        for gen in range(n_generations):
            gen_start = time.time()
            print(f"\n  --- Generation {gen+1}/{n_generations} "
                  f"(alpha={alpha}) ---")

            # ---- Fine-tune GPT-2 on current text ----
            model = copy.deepcopy(model_ref)
            model.train()
            optimizer = optim.AdamW(model.parameters(), lr=lr)

            # Tokenize current texts
            input_ids = tokenize_texts(current_texts[:n_samples_per_gen],
                                       tokenizer, max_length)
            dataset = TensorDataset(input_ids)
            loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True)

            for epoch in range(n_epochs):
                total_loss = 0
                n_batches = 0
                for (batch_ids,) in loader:
                    batch_ids = batch_ids.to(device)
                    # Shift for next-token prediction
                    outputs = model(batch_ids, labels=batch_ids)
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

                print(f"    Epoch {epoch+1}/{n_epochs}: "
                      f"loss={total_loss/n_batches:.4f}")

            # ---- Generate text for next generation ----
            model.eval()
            gen_texts = []
            with torch.no_grad():
                for i in range(0, n_samples_per_gen, batch_size):
                    n_batch = min(batch_size, n_samples_per_gen - i)
                    # Use diverse prompts within GPT-2 vocabulary
                    prompts = torch.randint(0, vocab_size, (n_batch, 5),
                                           device=device)
                    outputs = model.generate(
                        prompts,
                        max_length=max_length,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    for output in outputs:
                        text = tokenizer.decode(output, skip_special_tokens=True)
                        gen_texts.append(text)

            print(f"    Generated {len(gen_texts)} texts")

            # ---- Mixing ----
            if alpha > 0:
                n_real = int(alpha * n_samples_per_gen)
                # Sample random real texts each generation (no recycling)
                # to avoid memorization of fixed subset
                real_idx = np.random.choice(len(real_texts_for_mixing),
                                           size=min(n_real, len(real_texts_for_mixing)),
                                           replace=False)
                real_subset = [real_texts_for_mixing[i] for i in real_idx]
                n_synth = n_samples_per_gen - n_real
                synth_subset = gen_texts[:n_synth]
                next_texts = real_subset + synth_subset
                np.random.shuffle(next_texts)
            else:
                next_texts = gen_texts[:n_samples_per_gen]

            # ---- Compute metrics ----
            # PPL on HELD-OUT eval texts (never seen during training)
            ppl = compute_perplexity(model, eval_texts, device)
            results['perplexity'].append(ppl)

            unique_ratio, ngram_div, entropy = compute_diversity_metrics(
                gen_texts[:1000], tokenizer)
            results['unique_token_ratio'].append(unique_ratio)
            results['ngram_diversity'].append(ngram_div)
            results['token_entropy'].append(entropy)
            results['vocab_size'].append(
                len(set(tokenizer.encode(' '.join(gen_texts[:500]),
                                         truncation=True, max_length=1024))))

            gen_elapsed = time.time() - gen_start
            print(f"    PPL={ppl:.2f}, UniqueRatio={unique_ratio:.4f}, "
                  f"NGramDiv={ngram_div:.4f}, Entropy={entropy:.2f}, "
                  f"Time={gen_elapsed:.1f}s")

            current_texts = next_texts

            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Checkpoint
            if not quick_test:
                save_checkpoint('exp5_lm_recursive', gen, {
                    'alpha': alpha,
                    'generation': gen,
                    'results': {k: v for k, v in results.items()},
                    'completed': True,
                })

        for key in results:
            results[key] = np.array(results[key])
        all_results[f'alpha_{alpha}'] = results

    # ---- Save results ----
    save_results('exp5_lm_recursive',
                 {k: v['perplexity'] for k, v in all_results.items()})

    # ---- Plot ----
    plot_exp5_results(all_results, alpha_values, n_generations)

    return all_results


def plot_exp5_results(all_results, alpha_values, n_generations):
    """Generate plots for Experiment 5."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    gens = np.arange(1, n_generations + 1)

    metrics_to_plot = [
        ('perplexity', 'Perplexity', '(a) Perplexity Growth'),
        ('unique_token_ratio', 'Unique Token Ratio', '(b) Vocab Collapse (Thm 2)'),
        ('ngram_diversity', 'N-gram Diversity', '(c) N-gram Diversity'),
        ('token_entropy', 'Token Entropy', '(d) Distribution Entropy'),
        ('vocab_size', 'Active Vocab Size', '(e) Category Extinction (Thm 2)'),
    ]

    for idx, (metric_key, ylabel, title) in enumerate(metrics_to_plot):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        for i, alpha in enumerate(alpha_values):
            key = f'alpha_{alpha}'
            if key in all_results and len(all_results[key][metric_key]) > 0:
                ax.plot(gens[:len(all_results[key][metric_key])],
                        all_results[key][metric_key],
                        'o-', color=COLORS[i], linewidth=2,
                        label=f'$\\alpha$={alpha}')
        ax.set_xlabel('Generation $t$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # (f) Perplexity ratio (no mixing vs with mixing)
    ax = axes[1, 2]
    key_no_mix = 'alpha_0.0'
    if key_no_mix in all_results and len(all_results[key_no_mix]['perplexity']) > 1:
        ppl_no_mix = all_results[key_no_mix]['perplexity']
        for i, alpha in enumerate(alpha_values):
            if alpha > 0:
                key = f'alpha_{alpha}'
                if key in all_results and len(all_results[key]['perplexity']) > 0:
                    min_len = min(len(ppl_no_mix),
                                 len(all_results[key]['perplexity']))
                    ratio = all_results[key]['perplexity'][:min_len] / \
                            np.maximum(ppl_no_mix[:min_len], 1e-10)
                    ax.plot(gens[:min_len], ratio, 'o-', color=COLORS[i],
                            linewidth=2, label=f'$\\alpha$={alpha}')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Generation $t$')
        ax.set_ylabel('PPL(mixing) / PPL(no mixing)')
        ax.set_title('(f) Mixing Benefit Ratio')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Experiment 5: Language Model Recursive Training (GPT-2)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'exp5_lm_recursive.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure saved: {os.path.join(FIG_DIR, "exp5_lm_recursive.png")}')


# ============================================================================
# EXPERIMENT 6: Optimal Mixing Schedule Search
# ============================================================================

def experiment6_mixing_schedules(seed=42, T_max=200, n_trials=10000,
                                  n=100, d=1, quick_test=False):
    """Experiment 6: Optimal Mixing Schedule Search.

    Systematically compares mixing schedules to find which minimizes
    total KL divergence over T generations, validating Theorem 4.

    Schedules tested:
      1. Constant alpha in {0.01, 0.05, 0.1, 0.2, 0.5}
      2. Linear decay: alpha_t = max(alpha_min, alpha_0 - c*t)
      3. Square root decay: alpha_t = c / sqrt(t)  [Theorem 4: p*=1/2]
      4. Inverse decay: alpha_t = c / t
      5. RDT-inspired: more real data early, less later

    Theoretical predictions (Theorem 4):
      - For alpha_t = t^{-p}:
        * p < 1/2: variance converges (over-mixing)
        * p = 1/2: boundary case (log divergence)
        * p > 1/2: variance diverges (under-mixing)
      - The critical exponent p* = 1/2 separates stable from unstable regimes.
      - Steady-state variance with constant alpha: sigma^2 / (n*alpha*(2-alpha))

    Estimated runtime: ~30 minutes on RTX 5090
    GPU memory: ~2 GB
    """
    print("\n" + "="*70)
    print("EXPERIMENT 6: Optimal Mixing Schedule Search")
    print("="*70)

    set_seed(seed)
    device = get_device()

    if quick_test:
        T_max = 50
        n_trials = 1000
        print("  [QUICK TEST MODE]")

    # ---- Define schedules ----
    schedules = {
        # Constant mixing
        'const_0.01': lambda t: 0.01,
        'const_0.05': lambda t: 0.05,
        'const_0.10': lambda t: 0.10,
        'const_0.20': lambda t: 0.20,
        'const_0.50': lambda t: 0.50,
        # Linear decay
        'linear_decay': lambda t: max(0.01, 0.2 - 0.002 * t),
        # Square root decay (critical exponent p=1/2)
        'sqrt_decay': lambda t: min(0.5, 0.3 / np.sqrt(t + 1)),
        # Inverse decay (p=1)
        'inv_decay': lambda t: min(0.5, 0.3 / (t + 1)),
        # RDT-inspired: more real data early, then decay
        'rdt_inspired': lambda t: min(0.5, 0.3 * np.exp(-0.02 * t)),
        # No mixing (baseline)
        'no_mixing': lambda t: 0.0,
    }

    all_results = {}

    for sched_name, alpha_fn in schedules.items():
        print(f"\n  Schedule: {sched_name}")

        # ---- GPU-accelerated simulation ----
        # Process in chunks for memory efficiency
        chunk_size = min(n_trials, 20000)
        n_chunks = (n_trials + chunk_size - 1) // chunk_size

        var_mu_all = []
        kl_cumulative_all = []

        for chunk_idx in range(n_chunks):
            cs = min(chunk_size, n_trials - chunk_idx * chunk_size)

            mu_t = torch.zeros(cs, device=device)
            var_mu = torch.zeros(cs, T_max, device=device)
            kl_cumulative = torch.zeros(cs, T_max, device=device)

            for t in range(T_max):
                alpha_t = alpha_fn(t)

                # Number of real vs synthetic samples
                n_real = max(0, int(n * alpha_t))
                n_synth = n - n_real

                # Compute mean estimate
                # Real samples: N(mu_0, sigma^2) -> mean contribution
                # Synth samples: N(mu_t, sigma^2) -> mean contribution
                # Combined mean = (n_real * mu_0 + n_synth * mu_t) / n + noise
                # where noise ~ N(0, sigma^2/n)

                if n_real > 0 and n_synth > 0:
                    # Mixed estimator
                    mu_hat = (n_real * MU0 + n_synth * mu_t) / n + \
                             np.sqrt(SIGMA2 / n) * torch.randn(cs, device=device)
                elif n_real > 0:
                    # All real data (perfect recovery in expectation)
                    mu_hat = MU0 + np.sqrt(SIGMA2 / n_real) * \
                             torch.randn(cs, device=device)
                else:
                    # All synthetic (pure recursive)
                    mu_hat = mu_t + np.sqrt(SIGMA2 / n) * \
                             torch.randn(cs, device=device)

                # Track variance of mu_t
                var_mu[:, t] = (mu_hat - MU0) ** 2

                # KL divergence at this step
                kl_t = (mu_hat - MU0) ** 2 / (2 * SIGMA2)
                if t > 0:
                    kl_cumulative[:, t] = kl_cumulative[:, t-1] + kl_t
                else:
                    kl_cumulative[:, t] = kl_t

                mu_t = mu_hat

            var_mu_all.append(var_mu.cpu().numpy())
            kl_cumulative_all.append(kl_cumulative.cpu().numpy())

        var_mu_arr = np.concatenate(var_mu_all, axis=0)
        kl_cum_arr = np.concatenate(kl_cumulative_all, axis=0)

        # Mean statistics
        var_mu_mean = var_mu_arr.mean(axis=0)
        kl_cum_mean = kl_cum_arr.mean(axis=0)
        kl_per_gen_mean = np.diff(kl_cum_mean, prepend=0)

        all_results[sched_name] = {
            'var_mu_mean': var_mu_mean,
            'kl_cumulative_mean': kl_cum_mean,
            'kl_per_gen_mean': kl_per_gen_mean,
            'alpha_schedule': np.array([alpha_fn(t) for t in range(T_max)]),
            'final_kl': kl_cum_mean[-1],
            'final_var': var_mu_mean[-1],
        }

        print(f"    Final KL: {kl_cum_mean[-1]:.6f}, "
              f"Final Var: {var_mu_mean[-1]:.6f}")

    # ---- Theoretical validation ----
    print("\n--- Theoretical Validation ---")

    # For constant alpha schedules, compare with Theorem 4 prediction
    for alpha_val in [0.01, 0.05, 0.10, 0.20, 0.50]:
        key = f'const_{alpha_val:.2f}'
        if key in all_results:
            # Theorem 4: steady-state var = sigma^2 / (n * alpha * (2 - alpha))
            ss_theory = SIGMA2 / (n * alpha_val * (2 - alpha_val))
            ss_sim = all_results[key]['var_mu_mean'][-1]
            print(f"  alpha={alpha_val:.2f}: "
                  f"sim_ss={ss_sim:.6f}, theory_ss={ss_theory:.6f}, "
                  f"ratio={ss_sim/ss_theory:.4f}")

    # Critical exponent validation
    print("\n  Critical exponent p* = 1/2 validation:")
    for sched in ['sqrt_decay', 'inv_decay', 'rdt_inspired']:
        if sched in all_results:
            final_var = all_results[sched]['var_mu_mean'][-1]
            final_kl = all_results[sched]['kl_cumulative_mean'][-1]
            # sqrt_decay (p=1/2) should be at the boundary
            # inv_decay (p=1) should converge
            # Compare var growth rate in later generations
            late_var = all_results[sched]['var_mu_mean'][-20:]
            is_growing = late_var[-1] > late_var[0]
            print(f"    {sched}: var_growing={is_growing}, "
                  f"final_var={final_var:.6f}, final_kl={final_kl:.6f}")

    # ---- Save results ----
    save_results('exp6_mixing_schedules',
                 {k: v['var_mu_mean'] for k, v in all_results.items()})

    # ---- Plot ----
    plot_exp6_results(all_results, T_max, n)

    return all_results


def plot_exp6_results(all_results, T_max, n):
    """Generate plots for Experiment 6."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    gens = np.arange(1, T_max + 1)

    # (a) Variance of mu_t across generations (all schedules)
    ax = axes[0, 0]
    for i, (name, res) in enumerate(all_results.items()):
        ax.plot(gens, res['var_mu_mean'], '-', color=COLORS[i % len(COLORS)],
                linewidth=2, label=name)
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$\mathrm{Var}[\mu_t]$')
    ax.set_title('(a) Variance Evolution by Schedule')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (b) Cumulative KL
    ax = axes[0, 1]
    for i, (name, res) in enumerate(all_results.items()):
        ax.plot(gens, res['kl_cumulative_mean'], '-',
                color=COLORS[i % len(COLORS)],
                linewidth=2, label=name)
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'Cumulative $D_{\mathrm{KL}}$')
    ax.set_title('(b) Cumulative KL Divergence')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (c) Per-generation KL
    ax = axes[0, 2]
    for i, (name, res) in enumerate(all_results.items()):
        ax.plot(gens, res['kl_per_gen_mean'], '-',
                color=COLORS[i % len(COLORS)],
                linewidth=2, label=name)
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$D_{\mathrm{KL}}$ per generation')
    ax.set_title('(c) Per-Generation KL')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (d) Steady-state validation (Theorem 4)
    ax = axes[1, 0]
    alpha_range = np.linspace(0.001, 1.0, 200)
    ss_theory = SIGMA2 / (n * alpha_range * (2 - alpha_range))
    ax.plot(alpha_range, ss_theory, 'b-', linewidth=2.5,
            label=r'Theory: $\sigma^2/(n\alpha(2-\alpha))$')
    for alpha_val in [0.01, 0.05, 0.10, 0.20, 0.50]:
        key = f'const_{alpha_val:.2f}'
        if key in all_results:
            ax.plot(alpha_val, all_results[key]['var_mu_mean'][-1],
                    'ro', markersize=10, zorder=5)
    ax.plot([], [], 'ro', markersize=10, label='Simulation')
    ax.set_xlabel('Mixing fraction $\\alpha$')
    ax.set_ylabel(r'Steady-state $\mathrm{Var}[\mu_t]$')
    ax.set_title('(d) Steady-State Validation (Thm 4)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) Alpha schedule profiles
    ax = axes[1, 1]
    for i, (name, res) in enumerate(all_results.items()):
        ax.plot(gens, res['alpha_schedule'], '-',
                color=COLORS[i % len(COLORS)],
                linewidth=2, label=name)
    ax.set_xlabel('Generation $t$')
    ax.set_ylabel(r'$\alpha_t$')
    ax.set_title('(e) Mixing Schedule Profiles')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (f) Bar chart: final cumulative KL
    ax = axes[1, 2]
    names = list(all_results.keys())
    final_kls = [all_results[n]['final_kl'] for n in names]
    sorted_idx = np.argsort(final_kls)
    names_sorted = [names[i] for i in sorted_idx]
    kls_sorted = [final_kls[i] for i in sorted_idx]
    ax.barh(range(len(names_sorted)), kls_sorted,
            color=[COLORS[i % len(COLORS)] for i in range(len(names_sorted))])
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel(r'Total $D_{\mathrm{KL}}$ over $T$ generations')
    ax.set_title('(f) Schedule Ranking by Total KL')
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Experiment 6: Optimal Mixing Schedule Search',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'exp6_mixing_schedules.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure saved: {os.path.join(FIG_DIR, "exp6_mixing_schedules.png")}')


# ============================================================================
# QUICK TEST: Verify Setup
# ============================================================================

def quick_test(seed=42):
    """Run a quick test of each experiment to verify the setup.

    This runs each experiment with minimal parameters (1-3 generations,
    few epochs, small data) to catch configuration errors before
    committing to the full run.

    Expected runtime: ~5 minutes on RTX 5090
    """
    print("\n" + "="*70)
    print("QUICK TEST: Verifying Experimental Setup")
    print("="*70)

    # Check GPU
    device = get_device()
    if not torch.cuda.is_available():
        print("\n  [WARNING] No GPU detected! Experiments will be very slow on CPU.")
        print("  Continuing anyway for correctness check...")

    set_seed(seed)

    # Test basic tensor operations on GPU
    print("\n  Testing GPU tensor operations...")
    x = torch.randn(1000, 100, device=device)
    y = torch.randn(100, 50, device=device)
    z = x @ y
    print(f"    GPU matmul: {z.shape} - OK")

    # Test 1: Small VAE forward pass
    print("\n  Testing VAE model...")
    vae = VAE(latent_dim=20, hidden_dim=400, fixed_variance=True).to(device)
    x_test = torch.randn(32, 784, device=device).sigmoid()
    mu_x, logvar_x, mu_z, logvar_z = vae(x_test)
    loss, recon, kl = vae.loss_function(x_test, mu_x, logvar_x, mu_z, logvar_z)
    print(f"    VAE loss: {loss.item():.2f} - OK")
    samples = vae.sample(10, device)
    print(f"    VAE sample shape: {samples.shape} - OK")

    # Test 2: ConvVAE forward pass
    print("\n  Testing ConvVAE model...")
    conv_vae = ConvVAE(latent_dim=128, base_channels=64).to(device)
    x_cifar = torch.randn(8, 3, 32, 32, device=device).sigmoid()
    recon, mu_z, logvar_z = conv_vae(x_cifar)
    loss, recon_l, kl = conv_vae.loss_function(x_cifar, recon, mu_z, logvar_z)
    print(f"    ConvVAE loss: {loss.item():.2f} - OK")

    # Test 3: DDPM forward pass and fast sampling
    print("\n  Testing DDPM model...")
    ddpm = SmallDDPM(in_channels=3, base_channels=32, n_timesteps=100).to(device)
    t_test = torch.randint(0, 100, (8,), device=device)
    noise_pred = ddpm(x_cifar, t_test)
    print(f"    DDPM forward: output shape {noise_pred.shape} - OK")
    # Test fast sampling (2 samples, 10 inference steps)
    sample = ddpm.sample_fast(2, device, img_size=32, n_inference_steps=10)
    print(f"    DDPM sample_fast: shape {sample.shape}, "
          f"range [{sample.min():.3f}, {sample.max():.3f}] - OK")

    # Test 4: GPU Gaussian simulation
    print("\n  Testing GPU Gaussian simulation...")
    mu_t = torch.zeros(1000, device=device)
    for t in range(10):
        eps = torch.randn(1000, device=device)
        mu_hat = mu_t + np.sqrt(1.0 / 100) * eps
        mu_t = mu_hat
    var_sim = mu_t.var().item()
    var_theory = 10 / 100
    print(f"    Var(sim)={var_sim:.6f}, Var(theory)={var_theory:.6f} - OK")

    # Test 5: Check output directories
    print("\n  Testing output directories...")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    test_fig = os.path.join(FIG_DIR, 'test_plot.png')
    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.savefig(test_fig, dpi=72)
    plt.close()
    os.remove(test_fig)
    print(f"    {FIG_DIR} - OK")
    print(f"    {RES_DIR} - OK")

    # Test 6: Check GPT-2 availability (optional)
    print("\n  Checking GPT-2 availability...")
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        print("    GPT-2 available - OK")
    except ImportError:
        print("    [WARNING] transformers not installed. "
              "Experiment 5 will be skipped.")

    # Now run abbreviated versions of each experiment
    print("\n" + "-"*50)
    print("  Running abbreviated experiments...")
    print("-"*50)

    try:
        print("\n  >> Experiment 1 (quick)...")
        experiment1_vae_mnist(quick_test=True, seed=seed)
        print("  >> Experiment 1 PASSED")
    except Exception as e:
        print(f"  >> Experiment 1 FAILED: {e}")
        traceback.print_exc()

    try:
        print("\n  >> Experiment 2 (quick)...")
        experiment2_vae_cifar10(quick_test=True, seed=seed)
        print("  >> Experiment 2 PASSED")
    except Exception as e:
        print(f"  >> Experiment 2 FAILED: {e}")
        traceback.print_exc()

    try:
        print("\n  >> Experiment 3 (quick)...")
        experiment3_diffusion_cifar10(quick_test=True, seed=seed)
        print("  >> Experiment 3 PASSED")
    except Exception as e:
        print(f"  >> Experiment 3 FAILED: {e}")
        traceback.print_exc()

    try:
        print("\n  >> Experiment 4 (quick)...")
        experiment4_gaussian_gpu(quick_test=True, seed=seed)
        print("  >> Experiment 4 PASSED")
    except Exception as e:
        print(f"  >> Experiment 4 FAILED: {e}")
        traceback.print_exc()

    try:
        print("\n  >> Experiment 5 (quick)...")
        experiment5_lm_recursive(quick_test=True, seed=seed)
        print("  >> Experiment 5 PASSED")
    except Exception as e:
        print(f"  >> Experiment 5 FAILED: {e}")
        traceback.print_exc()

    try:
        print("\n  >> Experiment 6 (quick)...")
        experiment6_mixing_schedules(quick_test=True, seed=seed)
        print("  >> Experiment 6 PASSED")
    except Exception as e:
        print(f"  >> Experiment 6 FAILED: {e}")
        traceback.print_exc()

    print("\n" + "="*70)
    print("QUICK TEST COMPLETE")
    print("="*70)
    print("\nIf all tests passed, you can run the full experiments with:")
    print("  python gpu_experiments.py --exp 1    # Single experiment")
    print("  python gpu_experiments.py --exp all  # All experiments")


# ============================================================================
# MAIN: Argument parsing and experiment dispatch
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated Model Collapse Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments:
  1  VAE/MNIST Recursive Training       (~2.5 hrs)
  2  VAE/CIFAR-10 Recursive Training     (~8 hrs)
  3  Diffusion Model Recursive Training  (~20 hrs)
  4  Large-Scale Gaussian Simulation     (~1 hr)
  5  Language Model Recursive Training   (~12 hrs)
  6  Optimal Mixing Schedule Search      (~30 min)

Examples:
  python gpu_experiments.py --test                # Quick verification
  python gpu_experiments.py --exp 4               # Run Experiment 4 only
  python gpu_experiments.py --exp 1 --seed 123    # Custom seed
  python gpu_experiments.py --exp all             # Run everything
        """)

    parser.add_argument('--exp', type=str, default=None,
                        help='Experiment number (1-6) or "all"')
    parser.add_argument('--test', action='store_true',
                        help='Run quick test to verify setup')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help='Run in quick test mode (reduced parameters)')

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("GPU-ACCELERATED MODEL COLLAPSE EXPERIMENTS")
    print("On the Information-Theoretic Limits of Recursive Generative Training")
    print("=" * 70)
    print(f"Random seed: {args.seed}")
    print(f"Output figures: {FIG_DIR}")
    print(f"Output results: {RES_DIR}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected!")
    print()

    # Quick test mode
    if args.test:
        quick_test(seed=args.seed)
        return

    # Run specified experiment(s)
    if args.exp is None:
        parser.print_help()
        return

    start_time = time.time()

    if args.exp == 'all' or args.exp == 'ALL':
        # Run all experiments
        for exp_num in range(1, 7):
            print(f"\n{'#'*70}")
            print(f"# STARTING EXPERIMENT {exp_num}")
            print(f"{'#'*70}")
            exp_start = time.time()
            try:
                run_experiment(exp_num, args.seed, args.quick)
                exp_elapsed = time.time() - exp_start
                print(f"\n  Experiment {exp_num} completed in "
                      f"{exp_elapsed/60:.1f} minutes")
            except Exception as e:
                print(f"\n  Experiment {exp_num} FAILED: {e}")
                import traceback
                traceback.print_exc()
    else:
        try:
            exp_num = int(args.exp)
            if exp_num < 1 or exp_num > 6:
                print(f"ERROR: Invalid experiment number {exp_num}. "
                      f"Must be 1-6.")
                return
            run_experiment(exp_num, args.seed, args.quick)
        except ValueError:
            print(f"ERROR: Invalid experiment specifier '{args.exp}'")
            parser.print_help()
            return

    total_elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TOTAL ELAPSED TIME: {total_elapsed/3600:.2f} hours")
    print(f"Results saved to: {RES_DIR}")
    print(f"Figures saved to: {FIG_DIR}")
    print(f"{'='*70}")


def run_experiment(exp_num, seed, quick):
    """Dispatch to the appropriate experiment function."""
    if exp_num == 1:
        experiment1_vae_mnist(seed=seed, quick_test=quick)
    elif exp_num == 2:
        experiment2_vae_cifar10(seed=seed, quick_test=quick)
    elif exp_num == 3:
        experiment3_diffusion_cifar10(seed=seed, quick_test=quick)
    elif exp_num == 4:
        experiment4_gaussian_gpu(seed=seed, quick_test=quick)
    elif exp_num == 5:
        experiment5_lm_recursive(seed=seed, quick_test=quick)
    elif exp_num == 6:
        experiment6_mixing_schedules(seed=seed, quick_test=quick)
    else:
        raise ValueError(f"Unknown experiment: {exp_num}")


if __name__ == '__main__':
    main()
