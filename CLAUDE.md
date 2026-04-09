# Mamba-EKF Track: Project Context & AI Guidelines

## 1. Project Background & Goals
**Objective:** Refactor the baseline `MCTrack` project to develop a Fast, Efficient, and Accurate 3D Multi-Object Tracking (3DMOT) framework named **Mamba-EKF Track**.
**Target Publication:** IEEE TITS / IEEE TVT.

**Core Innovations:**
1. **Zero Transformer/GNN:** Strictly linear complexity `O(N)` to achieve extremely high FPS.
2. **MA-EKF (Mamba-driven Adaptive EKF):** Use State Space Models (Mamba) to process historical trajectories. It dynamically predicts dense, fully-coupled process noise (Q) and measurement noise (R) to handle complex vehicle maneuvers.
3. **Physics + Deep Learning:** Combine short-term kinematics (Differentiable EKF) with long-term semantic embeddings (Mamba).

## 2. Core Architecture Specifications
When refactoring or writing new code, adhere to these four modules:

### Module A: Differentiable EKF (Kinematics)
*   **State Definition:** `[x, y, z, theta, v, w, a]` based on the CTRA (Constant Turn Rate and Acceleration) model.
*   **Key Feature:** Must support batched tensor operations and accept dynamic `Q` and `R` matrices as inputs from Module B. 
*   **Time-Step Awareness:** The EKF equations must explicitly accept irregular time steps (delta_t) as inputs to handle variable sensor frame rates and missing detections robustly.
*   **Gradient Flow:** The EKF must be differentiable to allow end-to-end BPTT (Backpropagation Through Time).

### Module B: Mamba Trajectory Memory
*   **Input:** Historical states and basic features of each active tracklet over the past `T` frames, including their corresponding `delta_t`.
*   **Backbone:** Use `mamba-ssm` (Triton-based implementation).
*   **Dual Outputs:** 
    1. `Temporal Embedding (E_m)`: Used for downstream data association.
    2. `Adaptive Parameters`: A parameterization layer predicting Cholesky factors (L_matrix) to dynamically reconstruct fully dense, symmetric positive semi-definite (PSD) Q and R matrices.

### Module C: Multi-Modal Association
*   **Cost Matrix:** Combine Kinematic Affinity (3D IoU or Mahalanobis distance from EKF) and Semantic Affinity (Cosine similarity from Mamba Embeddings).
*   **Matching:** Implement Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`) or a fast Greedy Matcher.

### Module D: Tracker Management
*   **Lifecycle:** Standard 3DMOT logic (Birth for unmatched detections, Death for long-lost tracks, Active/Coasted states).

## 3. Strict Development Rules (CRITICAL)
As an AI coding assistant, you MUST follow these constraints:

1. **NO GNN or Transformer:** DO NOT introduce Graph Neural Networks, Self-Attention, or any `O(N^2)` modules.
2. **Vectorization (PyTorch):** Avoid `for` loops when processing multiple targets. Use batched tensor operations (e.g., `(B, N, State_dim)`).
3. **Type Hinting:** All Python functions must have explicit Type Hints.
4. **Math Annotations:** When writing EKF Jacobians or Mamba state equations, write the raw mathematical formula in the comments right above the code block.
5. **PSD Constraints (Safety Lock):** When predicting covariance matrices (Q and R), the network MUST apply a `Softplus` activation plus a small `epsilon` (e.g., 1e-5) on the diagonal elements of the predicted Cholesky factor `L` to prevent EKF inversion crashes and guarantee strict positive definiteness (`Q = L @ L^T + eps * I`).

## 4. Refactoring Roadmap (Step-by-Step)
Please assist the user by following this logical sequence:
*   **Step 1 (Clean-up):** Strip out all GNN/Transformer modules and heavy spatial-attention codes from the original MCTrack repository.
*   **Step 2 (Infra):** Build the pure PyTorch, batched, time-aware CTRA-EKF class.
*   **Step 3 (Mamba):** Integrate `mamba-ssm` to build the Trajectory Encoder with the Cholesky parameterization head.
*   **Step 4 (Pipeline):** Rewrite the association logic and tracker manager.
*   **Step 5 (Training):** Construct the end-to-end loss functions (MSE for EKF prediction + Contrastive Loss for Mamba embeddings).

## 5. Special AI Instructions for Claude
*   **Dimension Tracing:** Before writing complex tensor reshaping/permuting code, output a dimension tracing thought process in the code comments.
*   **Memory Check:** Keep an eye on VRAM usage. Ensure the computation graph is not broken between the Differentiable EKF and the Mamba network.
