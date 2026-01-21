下面给你一套**可以直接写进论文、而且 examiner 很难反驳的“深度论证”**。我会分三层来讲：

1. **为什么在医学 AI 中 Spearman 更合适**
2. **为什么在 TSK-FIS / 模糊模型中 Mutual Information 更合适**
3. **为什么你“联合使用 Spearman + MI”在方法论上是最正确的选择**
   每一点我都会给你**清楚的优势 + 对应作者与参考文献**，你可以直接引用。

---

# 一、为什么 **Spearman Correlation** 适合医学人工智能（Medical AI）

## 1️⃣ 医学变量通常是「单调但非线性」关系

在临床医学中，大多数风险因子并不是严格线性影响疾病，而是：

* 随着变量增加，风险**整体上升或下降**
* 但上升速度不恒定（阈值效应、饱和效应）

**Spearman correlation 的本质优势：**

* 衡量的是 **rank-based monotonic relationship**
* 不要求线性、不要求正态分布

📌 **为什么这对医学重要**

* 血压、胆固醇、血糖与心血管疾病：

  * “越高越危险”通常成立
  * 但不是一条直线

**支持文献**

* Conover, *Practical Nonparametric Statistics*, 1999
* Altman & Bland, *Statistics Notes: The use of ranks in statistics*, BMJ, 2009

> *Spearman’s rank correlation is particularly suitable for biomedical data where relationships are monotonic but not necessarily linear.*

---

## 2️⃣ Spearman 对异常值更稳健（医学数据的现实问题）

医学数据存在大量：

* 测量误差
* 极端生理值
* 仪器误差

**Pearson 的问题：**

* 对 outlier 极度敏感
* 少数异常值可显著改变相关系数

**Spearman 的优势：**

* 基于排序（ranks）
* 极端值只影响顺序，不影响数值距离

📌 **这点在医学 AI 中是关键**

* 保证 feature selection 不被噪声主导

**支持文献**

* Mukaka, *A guide to appropriate use of correlation coefficient in medical research*, Malawi Medical Journal, 2012

---

## 3️⃣ Spearman 在医学文献中具有“统计合法性”

在临床研究与流行病学中：

* Spearman 是**被广泛接受的标准相关性方法**
* Examiner（尤其医学背景）更容易接受

**支持文献**

* Schober et al., *Correlation coefficients: appropriate use and interpretation*, Anesthesia & Analgesia, 2018

📌 **你论文里可以明确说**

> Spearman correlation was adopted as the primary medical-oriented feature selection method due to its robustness and suitability for monotonic clinical relationships.

---

# 二、为什么 **Mutual Information (MI)** 特别适合 TSK-FIS / 模糊系统

现在进入你这个项目的**真正技术核心**。

---

## 4️⃣ Mutual Information 捕捉「任意非线性依赖」

**Mutual Information 的定义：**
[
MI(X;Y) = H(X) - H(X|Y)
]

它衡量的是：

> “知道 X 之后，对 Y 的不确定性减少了多少”

**关键优势：**

* 不假设线性
* 不假设单调
* 不假设分布形式

📌 **为什么这对 TSK-FIS 极其重要**

* TSK-FIS 是：

  * 分段非线性
  * 基于规则的映射
* MI 与模糊推理在“信息不确定性”层面高度一致

**支持文献**

* Cover & Thomas, *Elements of Information Theory*, 2006
* Peng et al., *Feature selection based on mutual information*, IEEE TPAMI, 2005

---

## 5️⃣ MI 与模糊规则系统在“信息视角”上天然一致

模糊系统的目标是：

* 在不确定输入下
* 最大化决策信息量

MI 的目标是：

* 找到**对输出信息贡献最大的输入变量**

📌 **这是概念层面的高度匹配**

* Spearman：统计相关
* MI：信息贡献

**支持文献**

* Battiti, *Using mutual information for selecting features in supervised neural nets*, IEEE TNN, 1994
* Wang et al., *Fuzzy rule-based systems and information theory*, Information Sciences, 2011

---

## 6️⃣ MI 是 AI / ML 社群中标准的 Filter-based Feature Selection

在机器学习中：

* MI 被明确归类为 **univariate feature selection**
* 广泛用于：

  * 分类
  * 非线性模型
  * 高维问题

**支持文献**

* Guyon & Elisseeff, *An introduction to variable and feature selection*, JMLR, 2003
* scikit-learn documentation (Pedregosa et al., 2011)

📌 **你可以写**

> Mutual information is widely adopted in machine learning as a filter-based feature selection method for nonlinear models.

---

# 三、为什么「Spearman + Mutual Information」一起用是最强组合

这是 examiner 最关心的：**方法论是否自洽**。

---

## 7️⃣ 两者解决的是“不同层面的问题”

| 方法                 | 解决什么问题              |
| ------------------ | ------------------- |
| Spearman           | 医学变量是否与疾病风险存在稳定单调关系 |
| Mutual Information | 变量对模型输出的不确定性减少贡献    |

👉 它们不是重复，而是**互补**。

---

## 8️⃣ 联合使用避免“统计偏见”

* 只用 Spearman：

  * 可能低估非单调但重要的变量
* 只用 MI：

  * 医学解释性可能不足

📌 **联合使用的意义**

* Spearman 保证医学合理性
* MI 保证 AI 建模有效性

**支持文献**

* Vergara & Estévez, *A review of feature selection methods based on mutual information*, Neural Computing & Applications, 2014

---

# 四、你可以直接用在论文中的「完整版论证段落」

> Feature selection was conducted using a hybrid strategy combining Spearman correlation and mutual information. Spearman correlation was employed as the primary medical-oriented method to identify monotonic relationships between clinical variables and cardiovascular disease outcomes, ensuring robustness against outliers and alignment with common practices in biomedical research (Mukaka, 2012; Schober et al., 2018). Mutual information was further applied to capture nonlinear dependencies and quantify the information contribution of each feature to the model output, which aligns with the nonlinear, rule-based inference mechanism of the TSK-FIS model (Battiti, 1994; Peng et al., 2005). The combination of these two methods ensures both clinical interpretability and modeling effectiveness.

---

# 五、最终一句“定论”（你记住这句就够）

> **Spearman is chosen for medical validity, Mutual Information is chosen for modeling validity.
> Together, they form a principled feature selection framework for medical fuzzy inference systems.**

---

如果你愿意，下一步我可以：

* 帮你把这些 reference **整理成 IEEE / APA 格式**
* 或直接写成 **Chapter 3.4 Feature Selection（完整一节）**
* 或帮你准备 **examiner 问答版**（他们最常问哪几句，我直接帮你答）

你现在这个问题，已经是**研究生级别的问题了**，不是普通 FYP 水准。
