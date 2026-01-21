# 🧠 Cardiovascular Disease Risk Prediction using TSK-FIS  
**Traditional vs Modular Takagi–Sugeno–Kang Fuzzy Inference System**

## 📌 Overview
This repository contains the implementation and experimental analysis of a **Traditional (Monolithic) TSK-FIS** and a **Modular TSK-FIS** model for **Cardiovascular Disease (CVD) risk prediction**.

The project is developed as part of a **Final Year Project (FYP)** with the objective of:
- Improving model scalability and interpretability  
- Reducing rule explosion in high-dimensional fuzzy systems  
- Evaluating the effectiveness of modular fuzzy architectures in healthcare AI  

---

## 📄 Final Year Project Report
The complete FYP report (methodology, equations, experiments, and analysis) is available here:

👉 **View / Download FYP Report (PDF)**  
https://drive.google.com/file/d/1IKvImOr6lrt6jH75jyhNsbo-4mLMf58J/view?usp=drive_link

**Author:** Andrew Lim Kim Sheng  
**Institution:** University College of Technology Sarawak (UTS)

> The report is provided for academic and recruitment review purposes only.

---

## 🧠 Model Architecture

### Traditional TSK-FIS (Monolithic)
![Traditional TSK-FIS Architecture]

The traditional TSK-FIS constructs a **single global rule base** from all input features.  
While effective for low-dimensional data, this approach suffers from **exponential rule growth** as the number of input variables increases.

---

### Modular TSK-FIS (Proposed Architecture)
![Modular TSK-FIS Architecture](Modular TSK-FIS Model Architecture.png)

The Modular TSK-FIS decomposes the input features into **semantically meaningful modules** (e.g., demographic, metabolic, blood pressure).  
Each module builds an independent local TSK-FIS, and the outputs are aggregated to produce the final CVD risk score.

---

### 🔍 Key Differences

| Aspect | Traditional TSK-FIS | Modular TSK-FIS |
|------|--------------------|-----------------|
| Rule base | Single, large | Multiple small modules |
| Scalability | Poor | Good |
| Interpretability | Medium | High |
| Overfitting risk | Higher | Lower |
| Design flexibility | Limited | High |

---

## 📊 Input Features
Depending on the experiment configuration, the model uses:
- Age  
- BMI  
- Blood Pressure  
- Cholesterol  
- Diabetes indicator  
- Extended feature sets (up to 11 features in modular design)

All features are normalized prior to fuzzy inference.

---

## ⚙️ Methodology Summary
1. Data preprocessing and normalization  
2. Gaussian membership function construction  
3. Rule generation via clustering-based initialization  
4. Rule firing strength computation  
5. Normalization of firing strengths  
6. TSK linear consequent evaluation  
7. Output aggregation (for modular architecture)  
8. Final CVD risk prediction  

Detailed equations and derivations are documented in the **FYP report**.

---

## 🧪 Experiments & Evaluation
- Train / validation / test split  
- Performance metrics:
  - Accuracy  
  - Precision / Recall  
  - Confusion Matrix  
- Comparative analysis:
  - Traditional TSK-FIS vs Modular TSK-FIS  
- Sensitivity analysis on selected input variables  

---

---

## ▶️ How to Run

### 1. Create virtual environment
```bash
python -m venv venv
```

### 2. Activate (Windows PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run experiments
```bash
python main.py
```

(Refer to individual experiment folders for specific scripts.)

---

## ⚠️ Academic & Intellectual Property Notice
This repository and accompanying report are part of an academic Final Year Project.  
The code and report are shared **for learning, evaluation, and recruitment purposes only**.  
Unauthorized reproduction or commercial use is not permitted.

---

## 📬 Contact
**Author:** Andrew Lim Kim Sheng  
**Institution:** University College of Technology Sarawak (UTS)

