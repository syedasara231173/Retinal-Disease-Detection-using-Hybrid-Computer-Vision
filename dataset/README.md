
**Dataset Provider**

EyePACS (Eye Picture Archive Communication System)

**Competition Host**

Kaggle

**Dataset Link**

https://www.kaggle.com/competitions/diabetic-retinopathy-detection

---

# Dataset Statistics

| Property | Value |
|-----------|-------|
| Total Images | 88,702 |
| Training Images | 35,126 |
| Test Images | 53,576 |
| Number of Classes | 5 |
| Image Type | RGB Retinal Fundus Images |
| Resolution | Approximately 433 × 289 up to 5184 × 3456 pixels |
| Labels | Expert Ophthalmologist Annotations |
| Grading System | International Clinical Diabetic Retinopathy (ICDR) |

The original dataset download is approximately **80–90 GB** because it contains tens of thousands of high-resolution retinal images. Smaller preprocessed mirrors exist, but the official competition data is substantially larger due to the original image resolutions.


# Diabetic Retinopathy Severity Grades

Each retinal image is assigned one of five severity grades based on the International Clinical Diabetic Retinopathy (ICDR) grading protocol.

## Grade 0 — No Diabetic Retinopathy

Healthy retina with no visible signs of diabetic retinopathy.

Typical characteristics include:

- Healthy retinal blood vessels
- No hemorrhages
- No microaneurysms
- No hard exudates
- No cotton wool spots

Clinical interpretation:

The patient currently shows no detectable retinal damage caused by diabetes.

## Grade 1 — Mild Non-Proliferative Diabetic Retinopathy (Mild NPDR)

The earliest clinically detectable stage of diabetic retinopathy.

Visible findings include:

- Few microaneurysms
- Small localized vascular abnormalities
- Minimal retinal damage

Clinical interpretation:

Vision is often unaffected, but early pathological changes have begun. Regular monitoring is recommended to prevent disease progression.

---

## Grade 2 — Moderate Non-Proliferative Diabetic Retinopathy (Moderate NPDR)

Retinal damage becomes more pronounced.

Common observations include:

- Multiple microaneurysms
- Retinal hemorrhages
- Hard exudates
- Cotton wool spots
- Increased vascular abnormalities

Clinical interpretation:

Blood vessels begin leaking larger amounts of blood and fluid, increasing the risk of vision impairment.

---

## Grade 3 — Severe Non-Proliferative Diabetic Retinopathy (Severe NPDR)

Extensive retinal vascular damage is present.

Typical findings include:

- Numerous retinal hemorrhages
- Extensive venous abnormalities
- Intraretinal Microvascular Abnormalities (IRMA)
- Significant retinal ischemia

Clinical interpretation:

The retina receives insufficient oxygen, placing the patient at high risk of developing proliferative diabetic retinopathy.

---

## Grade 4 — Proliferative Diabetic Retinopathy (PDR)

The most advanced and vision-threatening stage.

Typical findings include:

- Neovascularization (growth of abnormal blood vessels)
- Vitreous hemorrhage
- Fibrous tissue formation
- Retinal detachment risk
- Severe vision loss

Clinical interpretation:

Immediate ophthalmic intervention is generally required. Without treatment, this stage may lead to permanent blindness.

---

# Class Distribution

One of the major challenges of the EyePACS dataset is its severe class imbalance.

Approximately three-quarters of the images belong to the **No Diabetic Retinopathy (Grade 0)** category, while the advanced disease stages (Grades 3 and 4) represent only a small fraction of the dataset. This imbalance can bias deep learning models toward predicting the majority class unless appropriate techniques such as class weighting, oversampling, focal loss, or data augmentation are employed. :contentReference[oaicite:2]{index=2}

---

# Dataset Challenges

The EyePACS dataset presents several real-world challenges:

- High variation in illumination conditions.
- Different camera manufacturers and acquisition devices.
- Variable image resolutions.
- Blurred and out-of-focus retinal images.
- Underexposed and overexposed photographs.
- Imaging artifacts and noise.
- Large inter-patient variability.
- Significant class imbalance.
- Presence of low-quality and partially visible retinal images.

These characteristics make EyePACS considerably more difficult than many curated academic datasets while also making it a stronger benchmark for developing clinically applicable models. :contentReference[oaicite:3]{index=3}

---

# Preprocessing Performed in This Project

To improve image quality and facilitate feature extraction, the following preprocessing pipeline was applied:

- Image resizing
- Green channel extraction
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Noise reduction
- Blood vessel enhancement
- Lesion localization
- Normalization

These preprocessing steps improve the visibility of retinal structures and pathological lesions before feature extraction and classification.
