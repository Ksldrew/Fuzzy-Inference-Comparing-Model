# TSK-FIS Model (Traditional Version)

---

# Examiner Quick-Read

**Traditional TSK-FIS (K-Means-Based Fuzzy Inference System)**

---

## Model Type (Key Statement)

This work implements a **Traditional (Monolithic) Takagi–Sugeno–Kang Fuzzy Inference System (TSK-FIS)** based on **machine learning**, not deep learning.

* **Not Deep Learning**: no neural layers, no hidden representations, no end-to-end backpropagation
* **Machine Learning (Soft Computing)**: data-driven fuzzy partitions + supervised parameter estimation
* **Purpose**: transparent baseline for comparison with Modular TSK-FIS

---

## Dataset (Auto-Loaded)

The script automatically reads:

* `train.csv`
* `val.csv`
* `test.csv`

from:

```
C:\Users\asus\Desktop\FYP Improvement\FYP2\Edited_Dataset\
```

**Target label**: `cardio` (0 / 1)

**Input features (4)**:

* `age_years`
* `ap_hi` (Systolic BP)
* `ap_lo` (Diastolic BP)
* `cholesterol`

---

## Output Location

All results are saved to:

```
C:\Users\asus\Desktop\FYP Improvement\FYP2\feature_4\tsk-fis_model\
└── tskfis_run_YYYYMMDD_HHMMSS\
```

---

## Architecture Summary (Matches Chapter 3)

### 1) Input Layer

Four numerical inputs are passed directly into the fuzzy system.
Z-score normalization is **optional** and **not required** because clustering is performed **per feature (1-D)**.

---

### 2) Membership Function Layer

Each input variable is represented by **three Gaussian membership functions** (Low / Normal / High):

[
\mu(x)=\exp\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)
]

* ( \mu ): learned using **K-Means clustering**
* ( \sigma ): cluster standard deviation
* MF parameters are saved in JSON format

---

### 3) Fuzzy Rule Layer

Rules are generated using **grid partitioning** (truth-table style).

* Inputs = 4
* MFs per input = 3
* **Total rules**:
  [
  3^4 = 81
  ]

Rules are **not manually written**; all combinations are enumerated automatically.

---

### 4) Rule Firing & Normalization

* Firing strength:
  [
  w_i = \prod_{j=1}^{4} \mu_{ij}(x_j)
  ]

* Normalization:
  [
  \tilde{w}*i = \frac{w_i}{\sum*{j=1}^{81} w_j}
  ]

---

### 5) TSK Consequent Layer

Each rule produces a linear output:

[
f_i(x)=p_i x_1 + q_i x_2 + r_i x_3 + s_i x_4 + t_i
]

* Coefficients are estimated **once** using **Ridge Regression**
* No epochs, no loss function, no iterative updates

---

### 6) Final Output

The final output is computed as:

[
y=\sum_{i=1}^{81} \tilde{w}_i f_i(x)
]

The continuous score is optionally mapped to risk levels:

* Low, Moderate, High

---

## Visualization (Explainability Evidence)

Generated outputs include:

* Membership Functions (original units)
* Histogram + MF overlay (model sanity check)
* **2D response curves** (single-feature effect)
* **2D slice curves** (interaction shown via multiple lines)
* **3D response surfaces** (pairwise feature interaction)

All plots are for **interpretability and validation**, not training.

---

## Learning Characteristics (Examiner Checklist)

| Aspect               | Traditional TSK-FIS    |
| -------------------- | ---------------------- |
| MF generation        | Unsupervised (K-Means) |
| Rule structure       | Fixed (3⁴ = 81)        |
| Parameter learning   | One-shot (Ridge)       |
| Epoch-based training | No                     |
| Loss function        | No                     |
| End-to-end learning  | No                     |
| Deep learning        | No                     |

---

## Key Clarification

> This traditional TSK-FIS is a **constructed model**, not a continuously trained model.
> Learning occurs only during membership initialization (unsupervised) and consequent estimation (supervised, one-time).

---

## Purpose in This Study

* Serves as a **transparent, interpretable baseline**
* Enables **fair comparison** with the proposed **Modular TSK-FIS**
* Fully aligned with **Chapter 3 methodology**

---

### Final Consistency Check

✔ Architecture matches Chapter 3
✔ Backend computation follows TSK-FIS theory
✔ Correctly classified as **Machine Learning (Fuzzy Inference System)**
✔ No deep learning components involved

---


