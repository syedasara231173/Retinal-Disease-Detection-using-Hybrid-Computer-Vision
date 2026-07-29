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


# Technique 3
## Top-Hat Transformation

Top-Hat Transformation is a morphological image processing technique used to enhance small, bright retinal structures while suppressing uneven background illumination. By subtracting the morphologically opened image from the original image, it improves the visibility of fine anatomical details such as blood vessels and hard exudates.

In our preprocessing pipeline, this step enhances local contrast and highlights subtle pathological features, providing cleaner and more informative inputs for vessel segmentation, lesion detection, and the DINO Vision Transformer. As a result, it improves feature representation and supports more accurate diabetic retinopathy classification.

# Technique 4
## Black-Hat Transformation

Black-Hat Transformation is a morphological image processing technique used to enhance **small, dark structures** in retinal fundus images. It is computed by subtracting the original image from its morphologically closed version, making dark regions such as blood vessels, microaneurysms, and certain hemorrhages more prominent against the retinal background.

In our preprocessing pipeline, Black-Hat Transformation complements other enhancement techniques by improving the visibility of subtle dark pathological features. This enriched representation supports more accurate vessel segmentation, lesion analysis, and feature extraction, ultimately providing higher-quality inputs for the DINO Vision Transformer.

