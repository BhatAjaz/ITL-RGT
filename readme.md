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
