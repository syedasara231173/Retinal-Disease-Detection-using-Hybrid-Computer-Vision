def extract_green_channel(img_path, img_size=224):
    """
    Green channel isolates retinal vasculature best.
    Standard in fundus image analysis — blood vessels
    absorb green light most, giving maximum contrast.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    samples = val_df.sample(4, random_state=10)

    for i, (_, row) in enumerate(samples.iterrows()):
        p = f"{IMG_DIR}/{row['id_code']}.png"
        im = cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (img_size, img_size))
        green = im[:,:,1]

        axes[0][i].imshow(im)
        axes[0][i].set_title(f"Original\n{class_names[row['diagnosis']]}", fontsize=9)
        axes[0][i].axis('off')

        axes[1][i].imshow(green, cmap='gray')
        axes[1][i].set_title(f"Green Channel\n(vessel contrast)", fontsize=9)
        axes[1][i].axis('off')

    plt.suptitle("DIP Step 1 — Green Channel Extraction for Vessel Enhancement", fontsize=13)
    plt.tight_layout()
    plt.show()

extract_green_channel(f"{IMG_DIR}/{val_df.iloc[0]['id_code']}.png")

#step 2: CLAHE on green channel
def morphological_vessel_segmentation(img_path, img_size=224):
    """
    Classical DIP pipeline for vessel segmentation:
    Green channel → CLAHE → Top-hat transform → Threshold → Skeletonize
    Top-hat = difference between image and its morphological opening
    It reveals fine structures (vessels) against uneven background.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    green = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:,:,1]

    # Step 1: CLAHE on green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)

    # Step 2: Top-hat transform (reveals vessels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

    # Step 3: Blackhat (reveals dark lesions like hemorrhages)
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

    # Step 4: Combine and threshold
    combined = cv2.add(enhanced, tophat)
    combined = cv2.subtract(combined, blackhat)
    _, vessel_mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 5: Clean up noise
    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, clean_kernel)

    return green, enhanced, tophat, blackhat, vessel_mask


# Visualize full pipeline on multiple grades
fig, axes = plt.subplots(5, 5, figsize=(20, 20))
stages = ['Green Channel', 'CLAHE Enhanced', 'Top-Hat\n(vessels)',
          'Black-Hat\n(hemorrhages)', 'Vessel Mask\n(Otsu Threshold)']

# Try to get one sample per grade
grade_samples = []
for g in range(5):
    subset = val_df[val_df['diagnosis'] == g]
    if len(subset) > 0:
        grade_samples.append(subset.sample(1).iloc[0])

for row_idx, row in enumerate(grade_samples):
    path = f"{IMG_DIR}/{row['id_code']}.png"
    green, enhanced, tophat, blackhat, vessels = morphological_vessel_segmentation(path)

    stages_imgs = [green, enhanced, tophat, blackhat, vessels]
    for col_idx, (stage_img, stage_name) in enumerate(zip(stages_imgs, stages)):
        axes[row_idx][col_idx].imshow(stage_img, cmap='gray')
        if row_idx == 0:
            axes[row_idx][col_idx].set_title(stage_name, fontsize=10, fontweight='bold')
        axes[row_idx][col_idx].set_ylabel(f"Grade {row['diagnosis']}\n{class_names[row['diagnosis']]}",
                                           fontsize=8)
        axes[row_idx][col_idx].set_xticks([])
        axes[row_idx][col_idx].set_yticks([])

plt.suptitle("DIP Step 2 — Morphological Vessel Segmentation Pipeline (All 5 DR Grades)",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

#step 3: Classical lesion candidate detection
def detect_lesion_candidates(img_path, img_size=224):
    """
    Classical detection of two key DR lesion types:

    Microaneurysms: tiny dark red dots, earliest DR sign
    → detect via top-hat on inverted green + blob detection

    Hemorrhages: larger dark blotches
    → detect via connected components on thresholded image
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    green = img_rgb[:,:,1]

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)

    # Invert (microaneurysms are dark → become bright after invert)
    inverted = cv2.bitwise_not(enhanced)

    # --- Microaneurysm candidates: small bright blobs on inverted ---
    kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    tophat_ma = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel_ma)
    _, ma_thresh = cv2.threshold(tophat_ma, 20, 255, cv2.THRESH_BINARY)

    # Blob detector for microaneurysms
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 80
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = False
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(cv2.bitwise_not(ma_thresh))

    # --- Hemorrhage candidates: larger dark regions ---
    kernel_hm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    tophat_hm = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel_hm)
    _, hm_thresh = cv2.threshold(tophat_hm, 30, 255, cv2.THRESH_BINARY)

    # Remove small blobs — keep only hemorrhage-sized regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hm_thresh)
    hemorrhage_mask = np.zeros_like(hm_thresh)
    for j in range(1, num_labels):
        area = stats[j, cv2.CC_STAT_AREA]
        if 80 < area < 3000:
            hemorrhage_mask[labels == j] = 255

    # Draw on original
    annotated = img_rgb.copy()

    # Draw microaneurysm blobs (red circles)
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        r = max(int(kp.size), 4)
        cv2.circle(annotated, (x,y), r, (255, 50, 50), 1)

    # Draw hemorrhage regions (yellow contours)
    contours, _ = cv2.findContours(hemorrhage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (255, 220, 0), 1)

    return img_rgb, ma_thresh, hemorrhage_mask, annotated, len(keypoints), len(contours)


# Run on samples across grades
fig, axes = plt.subplots(len(grade_samples), 4, figsize=(18, len(grade_samples)*4))
col_titles = ['Original', 'Microaneurysm\nCandidates',
              'Hemorrhage\nCandidates', 'Annotated\n(Red=MA, Yellow=HM)']

for i, row in enumerate(grade_samples):
    path = f"{IMG_DIR}/{row['id_code']}.png"
    orig, ma, hm, annotated, n_ma, n_hm = detect_lesion_candidates(path)

    for j, (im, title) in enumerate(zip([orig, ma, hm, annotated], col_titles)):
        axes[i][j].imshow(im, cmap='gray' if j in [1,2] else None)
        if i == 0:
            axes[i][j].set_title(title, fontsize=11, fontweight='bold')
        if j == 0:
            axes[i][j].set_ylabel(f"Grade {row['diagnosis']}\n{class_names[row['diagnosis']]}", fontsize=9)
        if j == 3:
            axes[i][j].set_xlabel(f"MA candidates: {n_ma} | HM candidates: {n_hm}", fontsize=8)
        axes[i][j].set_xticks([]); axes[i][j].set_yticks([])

plt.suptitle("DIP Step 3 — Classical Lesion Candidate Detection\n(Microaneurysms & Hemorrhages)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

#Gabor And Fourier Transform for Vessel Enhancement
def fourier_gabor_analysis(img_path, img_size=224):
    """
    Frequency domain DIP:
    - Fourier: reveals periodic structures (vessel patterns, drusen)
    - Gabor filters: oriented texture — detects vessels at multiple angles
      (Gabor is what early visual cortex does — nice story for judges)
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    green = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:,:,1].astype(np.float32)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green.astype(np.uint8)).astype(np.float32)

    # --- Fourier Transform ---
    f_transform = np.fft.fft2(enhanced)
    f_shifted   = np.fft.fftshift(f_transform)
    magnitude   = np.log1p(np.abs(f_shifted))
    magnitude_n = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

    # --- Gabor Filters at 4 orientations ---
    gabor_responses = []
    orientations = [0, 45, 90, 135]
    for theta in orientations:
        kernel = cv2.getGaborKernel(
            ksize=(21, 21),
            sigma=4.0,
            theta=np.deg2rad(theta),
            lambd=10.0,
            gamma=0.5,
            psi=0
        )
        response = cv2.filter2D(enhanced, cv2.CV_32F, kernel)
        gabor_responses.append(np.abs(response))

    gabor_combined = np.max(gabor_responses, axis=0)

    return enhanced, magnitude_n, gabor_responses, gabor_combined, orientations


# Visualize on one sample per grade
fig, axes = plt.subplots(len(grade_samples), 7, figsize=(24, len(grade_samples)*3.5))
col_labels = ['Green+CLAHE', 'FFT Magnitude',
              'Gabor 0°', 'Gabor 45°', 'Gabor 90°', 'Gabor 135°',
              'Gabor Max\n(all angles)']

for i, row in enumerate(grade_samples):
    path = f"{IMG_DIR}/{row['id_code']}.png"
    enhanced, fft_mag, gabors, gabor_comb, angles = fourier_gabor_analysis(path)

    all_imgs = [enhanced, fft_mag] + gabors + [gabor_comb]

    for j, (im, title) in enumerate(zip(all_imgs, col_labels)):
        axes[i][j].imshow(im, cmap='gray')
        if i == 0:
            axes[i][j].set_title(title, fontsize=9, fontweight='bold')
        if j == 0:
            axes[i][j].set_ylabel(f"Grade {row['diagnosis']}\n{class_names[row['diagnosis']]}", fontsize=8)
        axes[i][j].set_xticks([]); axes[i][j].set_yticks([])

plt.suptitle("DIP Step 6 — Frequency Domain Analysis: Fourier Transform + Gabor Filters\n"
             "(Gabor mimics V1 cortex orientation selectivity — biologically inspired DIP)",
             fontsize=12, y=1.01)
plt.tight_layout()
plt.show()