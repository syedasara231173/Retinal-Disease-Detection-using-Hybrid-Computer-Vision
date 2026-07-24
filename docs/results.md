# Experimental Results

## Evaluation Metrics

| Metric | Value |
|---------|------:|
| Accuracy | 80.55% |
| Precision (Macro Avg) | 66.70% |
| Recall (Macro Avg) | 68.90% |
| F1-Score (Macro Avg) | 67.30% |
| AUC (Mean, One-vs-Rest) | 0.9407 |
| QWK (Quadratic Weighted Kappa) | 0.9034 |

## Key Findings

* **Exceptional Ordinal Agreement:** The Quadratic Weighted Kappa (QWK) reached **0.9034**, demonstrating outstanding overall agreement with clinical grading despite a standard accuracy of **80.55%**.
* **Class-Specific Performance:** Performance varied across Diabetic Retinopathy (DR) severity grades. The model excelled at identifying healthy cases (**No DR F1-score: 0.968**), while minority classes like Severe DR presented greater challenges (**Severe F1-score: 0.432**).
* **Clinical Safety Validation:** High-risk evaluations for Grade 3 (Severe) and Grade 4 (Proliferative DR) confirmed that **0 dangerous misclassifications** (missed as Grade 0) occurred.
* **Error Margin Analysis:** When the model made incorrect predictions, **83.2% of errors** were off by only a single grade, indicating that misclassifications stayed close to the correct clinical severity level.