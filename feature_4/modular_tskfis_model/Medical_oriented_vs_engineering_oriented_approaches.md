# Modular Strategy Design: Medical-Oriented vs Engineering-Oriented Approaches

## Strategy A: Module-wise Standalone Predictive Capability (Medical-Oriented)

In this project, a medical-oriented modular strategy was adopted. Each clinical module (e.g., Age–Cholesterol and Blood Pressure) was required to demonstrate standalone predictive capability before fusion. This design ensures that every physiological factor group contributes meaningful diagnostic information independently, reflecting real-world clinical reasoning where multiple risk factors are assessed separately and jointly.

Module effectiveness was evaluated using validation metrics such as AUC and sensitivity. A module was considered effective if its standalone AUC exceeded 0.70. An auxiliary loss mechanism was introduced during training to enforce module-level learning, preventing the fusion mechanism from relying solely on a single dominant module.

This strategy prioritizes medical interpretability, clinical relevance, and robustness, making it suitable for healthcare risk prediction tasks.

## Strategy B: Independent Importance Gating (Engineering-Oriented)

An alternative engineering-oriented strategy allows each module to independently obtain high importance scores using non-normalized gating mechanisms. While this increases model flexibility, it introduces ambiguity in medical interpretation, as multiple modules may simultaneously appear highly important without a clear clinical meaning.

Given the focus on medical decision support and explainability, this approach was not adopted in the current study.

## Clarification: My Concept vs Current Implementation (Why Sensitivity Can Trade Off)

### Current Approach (Joint End-to-End Training, Single Best Checkpoint)

The current implementation trains Module 1 and Module 2 jointly in an end-to-end manner and saves a single best checkpoint (e.g., `best_model_state_dict.pt`). In this setting, the entire system (Module 1, Module 2, and the fusion/gating layer) is optimized under a shared objective function.

In practice, cardiovascular risk patterns are heterogeneous: some cases are dominated by blood-pressure-related signals, while others are dominated by age–cholesterol patterns. When both modules are trained together under a single global loss, the optimization process may favour the module that more rapidly reduces validation loss. As a result, one module can become dominant, and the other module may contribute less, which can lead to a trade-off where one module's sensitivity decreases when the other improves.

This behaviour is not necessarily a bug. It is a known outcome of multi-objective learning when gradients from different modules compete under one shared optimization target.

### Proposed Concept (Independent Module Learning + Decision-Level Fusion)

The proposed concept is to treat each module as an independent clinical "expert" and allow each expert to learn its own high-risk pattern without being suppressed by a jointly optimized fusion objective.

A practical interpretation is:

- Train Module 1 independently to obtain its own best checkpoint (e.g., `best_m1.pt`), optimised for detecting patterns related to Age–Cholesterol risk.
- Train Module 2 independently to obtain its own best checkpoint (e.g., `best_m2.pt`), optimised for detecting patterns related to Blood Pressure risk.
- Perform fusion at the decision level, where the final output is derived from the combined evidence from both modules.

This design is conceptually similar to multimodal or multi-expert fusion, where each branch is first encouraged to become strong on its own, and fusion is then used to integrate complementary evidence.

### Important Note: Fusion Still Requires Calibration (Not Automatic)

Although having two independent best checkpoints can improve module robustness, fusion does not automatically become optimal. Independent modules can produce outputs on different probability scales (e.g., one module may be more conservative while the other is more aggressive). Therefore, a calibrated fusion strategy is necessary, such as:

- A trainable fusion layer trained on validation outputs (late fusion / stacking), or
- A clinically motivated rule (e.g., a screening-oriented OR rule to prioritise sensitivity), with thresholds selected based on medical criteria.

### Medical Perspective

For screening-oriented cardiovascular risk prediction, prioritising sensitivity (minimising false negatives) is often clinically preferable. Independent module learning can support this by ensuring each module retains standalone diagnostic capability, while calibrated fusion balances sensitivity and specificity in the final decision.

## Extension to Chapter 5: Future Methodological Direction

The modular TSK-FIS presented in this study represents a direct continuation and enhancement of the traditional monolithic TSK-FIS developed in FYP1. While the current modular design improves interpretability and robustness by enforcing module-level predictive capability through auxiliary supervision, the joint end-to-end training paradigm may still introduce sensitivity trade-offs between modules when heterogeneous clinical patterns are present.

In real-world cardiovascular screening scenarios, different patients may exhibit risk signals dominated by different physiological domains. Although joint optimisation enables effective overall learning, it may unintentionally suppress the sensitivity of certain modules when competing gradients arise. This limitation motivates a potential future extension beyond the scope of the current implementation.

A promising direction for future work is the adoption of a two-stage, decision-level fusion framework. In such a framework, each clinical module would first be trained independently to achieve its optimal standalone performance, resulting in separate best checkpoints for each module. A subsequent decision-level fusion or calibration stage would then integrate the independently learned module outputs to produce the final risk prediction.

This approach aligns with a hierarchical soft computing paradigm, where domain-specific fuzzy subsystems act as independent clinical experts and a higher-level decision mechanism integrates their outputs. Importantly, this extension does not replace the proposed modular TSK-FIS, but builds upon it as a natural progression. Due to scope and time constraints, this methodology is proposed as future work and left for further investigation.
