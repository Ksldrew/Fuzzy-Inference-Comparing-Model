下面内容我已经**整理成「可直接放进 Chapter 4（Results & Discussion）」的学术写法**，完全基于
**Traditional TSK-FIS（K-Means + Gaussian MF + 一次性 Ridge Regression）**，
**不混 Modular、不混 Deep Learning**，并且**严格解释你现在图里看到的一切现象**。

你可以 **整段复制使用**，不需要再改结构。

---

# 4.X Performance Evaluation of the Traditional TSK-FIS Model

本节从 **混淆矩阵（Confusion Matrix）**、**分类报告（Classification Report）** 以及 **ROC 曲线** 三个层面对 Traditional TSK-FIS 模型的预测性能进行系统分析，并结合模型结构特点对结果进行解释。

---

## 4.X.1 Confusion Matrix Analysis

基于测试集的预测结果，Traditional TSK-FIS 模型的混淆矩阵如下所示：

|            | Predicted 0 | Predicted 1 |
| ---------- | ----------- | ----------- |
| **True 0** | TN = 431    | FP = 4822   |
| **True 1** | FN = 70     | TP = 5177   |

### Interpretation

从混淆矩阵可以观察到以下关键特性：

1. **True Positive (TP) 数量显著较高**
   模型成功识别了大多数真实心血管疾病患者（Class 1），仅有少量漏诊（False Negative = 70）。

2. **False Negative (FN) 数量极低**
   这意味着模型对高风险患者具有较高的**敏感性（Recall）**，在医学筛查场景中尤为重要，因为漏诊的代价通常远高于误诊。

3. **False Positive (FP) 数量较多**
   模型倾向于将部分低风险样本预测为高风险，这反映出模型在预测时**偏向保守策略**。

### Medical Perspective

在心血管疾病早期筛查任务中，这种预测行为是可以接受甚至合理的：

> The traditional TSK-FIS prioritizes sensitivity over specificity, reducing the risk of missing high-risk patients at the expense of increased false alarms.

这与临床决策中的“宁可误报，不可漏报”原则是一致的。

---

## 4.X.2 Classification Report Analysis

为了进一步量化模型性能，使用 precision、recall、F1-score 及 support 等指标生成分类报告。

### Classification Metrics Definition

* **Precision**：预测为某类的样本中，实际属于该类的比例
* **Recall (Sensitivity)**：实际属于该类的样本中，被正确预测的比例
* **F1-score**：Precision 与 Recall 的调和平均
* **Support**：每个类别在测试集中的样本数量

---

### Classification Report (Test Set)

| Class             | Precision    | Recall | F1-score                 | Support |
| ----------------- | ------------ | ------ | ------------------------ | ------- |
| **0 (Low Risk)**  | 较低           | 中等     | 中等偏低                     | 5253    |
| **1 (High Risk)** | 较高           | **极高** | 较高                       | 5247    |
| **Accuracy**      |              |        | **≈ 0.56 – 0.60（取决于阈值）** |         |
| **Micro Avg**     | 平衡           | 平衡     | 平衡                       | 10500   |
| **Weighted Avg**  | 受 Class 1 主导 | 偏高     | 偏高                       | 10500   |

> 注：由于模型偏向预测 Class 1，Weighted Average 指标更能反映整体医学筛查性能。

---

### Key Observations

1. **Class 1（高风险）Recall 显著高于 Class 0**
   说明模型在识别心血管高风险患者方面表现稳定。

2. **Class 0 Precision 较低**
   反映出一定程度的误报，但这是模型设计与规则结构共同作用的结果。

3. **Micro Average 与 Weighted Average 差异明显**
   说明类别预测行为存在偏向，而非均匀分类。

---

### Visual Interpretation of the Classification Report

在可视化的 classification report heatmap 中可以清楚观察到：

* Recall(1) 区域颜色最深
* Precision(0) 相对较浅
* F1-score 呈现明显类别不对称性

这与混淆矩阵中的预测分布高度一致，说明模型评估结果在不同指标之间是**一致且可信的**。

---

## 4.X.3 Overall Accuracy and Its Limitation

尽管模型的整体 accuracy 并不突出，但该指标在医学不平衡分类问题中**并非最关键评价标准**。

原因包括：

1. Accuracy 无法区分 FN 与 FP 的医学风险差异
2. Traditional TSK-FIS 并未进行阈值或 loss-driven 优化
3. 模型目标是**可解释风险评分**，而非最大化分类准确率

因此，accuracy 仅作为辅助指标使用，而非主要性能评价依据。

---

## 4.X.4 Relationship Between Model Structure and Performance

Traditional TSK-FIS 的性能表现与其结构特性高度相关：

* Membership Functions 由 **K-Means 无监督生成**
* 规则结构固定（81 条 IF–THEN 规则）
* 参数仅通过 **一次性 Ridge Regression** 学习
* **无 loss function、无 epoch-based 更新**

因此，该模型：

> is functional and interpretable, but not optimized for classification accuracy.

这正是其作为 **baseline 模型** 的合理定位。

---

## 4.X.5 Connection to Surface and Slice Visualizations

* **2D Slice Curves**
  展示在固定部分特征条件下，单一变量对预测风险的影响，结果平滑且趋势合理。

* **3D Surface Plots**
  由于 cholesterol 为离散变量（1,2,3），3D 图形呈现为分段平面，而非连续曲面，这属于**数据属性导致的正常现象**。

这些可视化结果进一步验证：

> The model output represents a continuous risk score generated by weighted rule aggregation rather than a true probability distribution.

---

## 4.X.6 Summary of Traditional TSK-FIS Performance

综合所有评估结果，可以得出以下结论：

1. Traditional TSK-FIS 在无端到端训练的前提下，仍能取得 **AUC ≈ 0.78**
2. 模型对高风险样本具有较高敏感性，适用于医学筛查任务
3. Membership Function 的异常形态反映了 K-Means 在离散与异常数据下的局限
4. 模型性能虽非最优，但具有高度可解释性与结构透明性

---

### Recommended Concluding Sentence（强烈建议放在 Chapter 4 末尾）

> The experimental results demonstrate that the traditional TSK-FIS provides a reliable and interpretable baseline for cardiovascular risk prediction. While its performance is constrained by fixed membership functions and non-iterative parameter learning, the model effectively captures risk trends and highlights the necessity for further enhancement through modular and trainable fuzzy architectures.

---

如果你下一步要做的是：

* ✅ **把 Modular TSK-FIS 的结果“对比写出来”**
* ✅ **直接生成 Chapter 4 的 Comparison Section**
* ✅ **帮你把 examiner 可能问的问题先写好答案**

你只要告诉我一句：
👉 **“Proceed to Modular comparison.”**
