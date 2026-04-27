# Mamba-Decoupled-EKF Track: Project Context & AI Guidelines

## 1. Project Background & Goals
**Objective:** Refactor the baseline `MCTrack` project to develop a Fast, Efficient, and Accurate 3D Multi-Object Tracking (3DMOT) framework named **Mamba-Decoupled-EKF Track**.
**Target Publication:** IEEE TITS / IEEE TVT.

**Core Innovations:**
1. **Zero Transformer/GNN:** Strictly linear complexity `O(N)` to achieve extremely high FPS by avoiding spatial cross-attention.
2. **MA-DKF (Mamba-driven Adaptive Decoupled KF):** Use State Space Models (Mamba) as a "Soft-Coupler". It processes joint historical trajectories to dynamically predict independent process noise (Q) and measurement noise (R) for three decoupled filters, handling non-linear maneuvers while maintaining dimensional isolation.
3. **Physics-Infused Adaptive Filtering:** Combine the stability of independent physical constraints with Mamba's long-term semantic and uncertainty estimation.

## 2. Core Architecture Specifications
When refactoring, adhere to this decoupled modular structure:

### Module A: Decoupled Adaptive KFs (Kinematics)
* **State Isolation:** Maintain three independent filters to prevent "dimension confusion" and error propagation:
    1. **Position Filter:** `[x, y, z, vx, vy, vz, ax, ay]` (Constant Acceleration model).
    2. **Size Filter:** `[l, w, h]` (Constant model).
    3. **Orientation Filter:** `[theta, omega]` (Constant Velocity model).
* **Key Feature:** Each filter must support batched operations and accept independent, dynamic `Q` and `R` matrices from Module B.
* **Time-Step Awareness:** Prediction equations must explicitly integrate `delta_t` for asynchronous sensor inputs and missing detections.

### Module B: Mamba Soft-Coupler & Memory
* **Input:** Joint historical states and features of each tracklet over the past `T` frames, capturing latent correlations between orientation and velocity.
* **Backbone:** Use `mamba-ssm` (Triton-based implementation).
* **Adaptive Outputs:** 1. `Temporal Embedding (E_m)`: For downstream semantic data association.
    2. `Multi-Head Noise Prediction`: Shallow MLP heads predicting independent Cholesky factors (L_matrix) for the $Q$ and $R$ matrices of the Position, Size, and Orientation filters respectively.

### Module C: Uncertainty-Aware Association
* **Cost Matrix:** Combine Kinematic Affinity (Ro_GDIoU or Mahalanobis distance) and Semantic Affinity (Cosine similarity from Mamba Embeddings).
* **Uncertainty Penalty:** Incorporate the trace or determinant of the predicted covariance `P` from the Position and Orientation filters as a dynamic penalty to increase search tolerance during high-uncertainty phases.

### Module D: Tracker Management
* **Lifecycle:** Standard 3DMOT logic (Birth, Death, Active/Coasted states).

## 3. Strict Development Rules (CRITICAL)
1. **NO GNN or Transformer:** Strictly maintain `O(1)` per-tracklet inference.
2. **Vectorization (PyTorch):** Use batched tensor operations (e.g., `torch.bmm`) for all three KFs.
3. **Type Hinting:** Explicit hints for all functions.
4. **Math Annotations:** Write raw formulas in comments above code blocks.
5. **PSD Constraints (Safety Lock):** For ALL predicted $Q$ and $R$ matrices, apply `Softplus` activation plus a small `epsilon` (1e-5) on the Cholesky diagonals to prevent inversion crashes and guarantee positive definiteness.

## 4. Refactoring Roadmap
* **Step 1 (Clean-up):** Remove the original unified KF and spatial attention modules from MCTrack.
* **Step 2 (Infra):** Build the three Decoupled EKF classes in PyTorch, ensuring they are time-aware and differentiable.
* **Step 3 (Mamba):** Integrate `mamba-ssm` with multiple heads for independent Q/R estimation.
* **Step 4 (Pipeline):** Rewrite matching logic to utilize Mamba embeddings and filter uncertainty.
* **Step 5 (Training):** Construct joint loss (MSE for state prediction + Contrastive Loss for embeddings).

## 5. Special AI Instructions for Claude
* **Dimension Tracing:** Document tensor shapes, especially when bridging joint Mamba inputs with decoupled KF updates.
* **Stability Check:** Monitor for NaN in any of the three filters during training.