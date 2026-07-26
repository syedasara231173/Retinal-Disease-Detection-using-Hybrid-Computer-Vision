# Technique 1
# Green Channel Extraction
Green channel extraction is a widely adopted preprocessing technique in retinal fundus image analysis. Among the three RGB color channels, the green channel provides the highest contrast between retinal blood vessels and the surrounding retinal tissue. 
## Why Green Channel?
Retinal blood vessels absorb green light more effectively than red or blue wavelengths. As a result, the green channel captures vascular structures with greater clarity while suppressing unnecessary color variations present in the other channels. This property makes it particularly suitable for retinal image enhancement and has become a standard practice in computer-aided diagnosis of diabetic retinopathy.
## Implementation
For each retinal fundus image:

1. The RGB image is separated into its individual color channels.
2. The green channel is extracted and visualized alongside the original retinal image.
3. Multiple validation samples are displayed to demonstrate the consistency of this preprocessing step across different disease severity levels.

## Contribution to the Proposed Framework

Green channel extraction improves the visibility of retinal blood vessels and subtle pathological structures, providing a cleaner representation for subsequent preprocessing operations such as contrast enhancement, vessel segmentation, and lesion localization. By emphasizing clinically relevant features while reducing redundant color information, this step helps improve feature quality before the images are processed by the DINO Vision Transformer.

Overall, green channel extraction serves as the first stage of the hybrid computer vision pipeline, establishing a stronger visual foundation for accurate and explainable diabetic retinopathy classification.

# Techique 2
## Contrast Limited Adaptive Histogram Equalization (CLAHE)
CLAHE was applied to improve the local contrast of retinal fundus images while preserving fine anatomical structures. Unlike standard histogram equalization, CLAHE enhances contrast within small image regions and limits excessive amplification of noise, making it particularly suitable for medical imaging applications.

This preprocessing step enhances the visibility of retinal blood vessels, microaneurysms, hemorrhages, and exudates, resulting in clearer feature representations for subsequent vessel segmentation, lesion detection, and DINO Vision Transformer-based classification. By producing higher-quality input images, CLAHE improves the robustness of the proposed diabetic retinopathy detection framework.