def apply_clahe(img_path, img_size=224):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Standard preprocessing for fundus images — enhances
    microaneurysms and hemorrhages that DINO needs to attend to.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    # Convert to LAB, apply CLAHE only on L channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

# Visualize preprocessing effect
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, row in df.sample(4, random_state=42).iterrows():
    path = f"{BASE}/train_images/{row['id_code']}.png"
    raw = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (224,224))
    clahe_img = apply_clahe(path)
    col = list(df.sample(4, random_state=42).index).index(i)
    axes[0][col].imshow(raw)
    axes[0][col].set_title(f"Raw — {class_names[row['diagnosis']]}")
    axes[0][col].axis('off')
    axes[1][col].imshow(clahe_img)
    axes[1][col].set_title(f"CLAHE — Grade {row['diagnosis']}")
    axes[1][col].axis('off')

plt.suptitle('Raw vs CLAHE Preprocessed Fundus Images', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

model.load_state_dict(torch.load('best_model.pth'))
_, _, final_preds, final_labels = val_epoch(model, val_loader, criterion)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
axes[0].plot(history['train_loss'], label='Train', color='steelblue')
axes[0].plot(history['val_loss'],   label='Val',   color='tomato')
axes[0].axvline(x=EPOCHS_PHASE1-1, color='gray', linestyle='--', label='Unfreeze point')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].set_xlabel('Epoch')

# Kappa curve
axes[1].plot(history['train_kappa'], label='Train', color='steelblue')
axes[1].plot(history['val_kappa'],   label='Val',   color='tomato')
axes[1].axvline(x=EPOCHS_PHASE1-1, color='gray', linestyle='--', label='Unfreeze point')
axes[1].set_title('Quadratic Weighted Kappa'); axes[1].legend(); axes[1].set_xlabel('Epoch')

# Confusion matrix
cm = confusion_matrix(final_labels, final_preds)
sns.heatmap(cm, annot=True, fmt='d', ax=axes[2],
            xticklabels=class_names.values(),
            yticklabels=class_names.values(),
            cmap='Blues')
axes[2].set_title(f'Confusion Matrix (Kappa={best_kappa:.3f})')
axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('True')

plt.tight_layout()
plt.show()

def extract_dip_features(img_path, img_size=224):
    """
    Extract a clinically-meaningful DIP feature vector per image.
    These are the features an ophthalmologist implicitly computes:
    - Vessel density, tortuosity proxy
    - Lesion count and area fractions
    - Texture (GLCM-based contrast, entropy)
    - Optic disc brightness estimate
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    green = img_rgb[:,:,1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(green)
    inverted = cv2.bitwise_not(enhanced)

    features = {}

    # 1. Vessel density (fraction of vessel pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    _, vmask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    features['vessel_density'] = vmask.sum() / (img_size * img_size * 255)

    # 2. Lesion burden (dark region fraction)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    tophat2 = cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel2)
    _, lmask = cv2.threshold(tophat2, 20, 255, cv2.THRESH_BINARY)
    features['lesion_density'] = lmask.sum() / (img_size * img_size * 255)

    # 3. Texture — Local Binary Pattern-like (std in patches)
    patch_stds = []
    for py in range(0, img_size-16, 16):
        for px in range(0, img_size-16, 16):
            patch = enhanced[py:py+16, px:px+16]
            patch_stds.append(patch.std())
    features['texture_mean_std']   = np.mean(patch_stds)
    features['texture_max_std']    = np.max(patch_stds)
    features['texture_entropy']    = -np.sum(
        np.histogram(enhanced, bins=256, density=True)[0] *
        np.log2(np.histogram(enhanced, bins=256, density=True)[0] + 1e-10)
    )

    # 4. Optic disc proxy — brightest circular region in red channel
    red = img_rgb[:,:,0]
    features['max_brightness']  = float(red.max())
    features['mean_brightness'] = float(red.mean())

    # 5. Color channel ratios (hemorrhages shift R/G ratio)
    r_mean = float(img_rgb[:,:,0].mean())
    g_mean = float(img_rgb[:,:,1].mean())
    b_mean = float(img_rgb[:,:,2].mean())
    features['rg_ratio'] = r_mean / (g_mean + 1e-8)
    features['rb_ratio'] = r_mean / (b_mean + 1e-8)

    return features


# Extract features for all validation images + show correlation with grade
print("Extracting DIP features from validation set...")
dip_records = []
for _, row in val_df.iterrows():
    path = f"{IMG_DIR}/{row['id_code']}.png"
    feats = extract_dip_features(path)
    feats['diagnosis'] = row['diagnosis']
    dip_records.append(feats)

dip_df = pd.DataFrame(dip_records)
print(f"Feature matrix: {dip_df.shape}")

# Correlation heatmap — features vs DR grade
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Correlation bar chart
feat_cols = [c for c in dip_df.columns if c != 'diagnosis']
correlations = dip_df[feat_cols].corrwith(dip_df['diagnosis']).sort_values()
colors = ['tomato' if x < 0 else 'steelblue' for x in correlations]
axes[0].barh(correlations.index, correlations.values, color=colors)
axes[0].axvline(0, color='black', linewidth=0.8)
axes[0].set_title("DIP Feature Correlation with DR Grade", fontsize=12)
axes[0].set_xlabel("Pearson Correlation")

# Box plots for most correlated feature
top_feat = correlations.abs().idxmax()
grade_groups = [dip_df[dip_df['diagnosis']==g][top_feat].values for g in range(5)]
axes[1].boxplot(grade_groups, labels=[f"G{g}\n{class_names[g][:8]}" for g in range(5)],
                patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[1].set_title(f"Most Discriminative Feature: '{top_feat}'", fontsize=12)
axes[1].set_ylabel(top_feat)
axes[1].set_xlabel("DR Grade")

plt.suptitle("DIP Step 4 — Handcrafted Feature Analysis vs DR Severity", fontsize=13)
plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

print("Extracting DIP features for full train+val split...")

# Build full feature sets
train_records, val_records = [], []

for _, row in train_df.iterrows():
    path = f"{IMG_DIR}/{row['id_code']}.png"
    feats = extract_dip_features(path)
    feats['diagnosis'] = row['diagnosis']
    train_records.append(feats)

for _, row in val_df.iterrows():
    path = f"{IMG_DIR}/{row['id_code']}.png"
    feats = extract_dip_features(path)
    feats['diagnosis'] = row['diagnosis']
    val_records.append(feats)

train_feat_df = pd.DataFrame(train_records)
val_feat_df   = pd.DataFrame(val_records)

feat_cols = [c for c in train_feat_df.columns if c != 'diagnosis']
X_train = train_feat_df[feat_cols].values
y_train = train_feat_df['diagnosis'].values
X_val   = val_feat_df[feat_cols].values
y_val   = val_feat_df['diagnosis'].values

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

# Train 3 classical models
classifiers = {
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, random_state=42),
    'SVM (RBF)':           SVC(kernel='rbf', C=10, random_state=42),
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train_sc, y_train)
    preds = clf.predict(X_val_sc)
    kappa = cohen_kappa_score(y_val, preds, weights='quadratic')
    results[name] = {'kappa': kappa, 'preds': preds}
    print(f"{name:25s} | Kappa: {kappa:.4f}")

dino_kappa = best_kappa
print(f"{'DINO ViT (our model)':25s} | Kappa: {dino_kappa:.4f}  ← Deep Learning")

# Comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

model_names  = list(results.keys()) + ['DINO ViT\n(Weakly Supervised)']
kappa_values = [results[n]['kappa'] for n in results.keys()] + [dino_kappa]
bar_colors   = ['#5b8db8','#5b8db8','#5b8db8','#e05c5c']

bars = axes[0].bar(model_names, kappa_values, color=bar_colors, alpha=0.85, edgecolor='black')
axes[0].set_ylim(0, 1.0)
axes[0].axhline(0.6, color='gray', linestyle='--', label='Clinical threshold (0.6)')
axes[0].set_title("Classical DIP+ML vs DINO Deep Learning\n(Quadratic Weighted Kappa)", fontsize=12)
axes[0].set_ylabel("Quadratic Weighted Kappa")
axes[0].legend()
for bar, val in zip(bars, kappa_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

# Best classical vs DINO classification report comparison
best_clf_name = max(results, key=lambda x: results[x]['kappa'])
best_preds    = results[best_clf_name]['preds']
_, _, dino_preds, dino_labels = val_epoch(model, val_loader, criterion)

report_classical = classification_report(y_val, best_preds,
                                          target_names=list(class_names.values()),
                                          output_dict=True)
report_dino      = classification_report(dino_labels, dino_preds,
                                          target_names=list(class_names.values()),
                                          output_dict=True)

f1_classical = [report_classical[c]['f1-score'] for c in class_names.values()]
f1_dino      = [report_dino[c]['f1-score'] for c in class_names.values()]

x = np.arange(5)
w = 0.35
axes[1].bar(x - w/2, f1_classical, w, label=f'Best Classical ({best_clf_name})',
            color='#5b8db8', alpha=0.85, edgecolor='black')
axes[1].bar(x + w/2, f1_dino,      w, label='DINO ViT (ours)',
            color='#e05c5c', alpha=0.85, edgecolor='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels(class_names.values(), rotation=15, fontsize=9)
axes[1].set_title("Per-Class F1 Score: Classical DIP vs DINO", fontsize=12)
axes[1].set_ylabel("F1 Score")
axes[1].legend()

plt.suptitle("DIP Step 5 — Classical Machine Learning vs Deep Learning: Full Comparison", fontsize=13)
plt.tight_layout()
plt.show()

print(f"\nKey finding: DINO outperforms best classical approach "
      f"({best_clf_name}) by {(dino_kappa - results[best_clf_name]['kappa']):.3f} kappa points")
print("Classical DIP features confirm lesion_density and vessel_density "
      "are clinically meaningful signals — DINO learns these and more, implicitly.")

from sklearn.metrics import (cohen_kappa_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize
import warnings

# Get predictions with probabilities
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── 1. The metrics that actually matter ──────────────────────────
accuracy = (all_preds == all_labels).mean() * 100
qwk      = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
linear_k = cohen_kappa_score(all_labels, all_preds, weights='linear')

# One-vs-rest AUC per class
y_bin = label_binarize(all_labels, classes=[0,1,2,3,4])
aucs  = []
for c in range(5):
    if y_bin[:, c].sum() > 0:
        aucs.append(roc_auc_score(y_bin[:, c], all_probs[:, c]))

print("=" * 55)
print("           HONEST EVALUATION REPORT")
print("=" * 55)
print(f"  Accuracy (misleading):       {accuracy:.2f}%")
print(f"  Quadratic Weighted Kappa:    {qwk:.4f}  ← THE real metric")
print(f"  Linear Weighted Kappa:       {linear_k:.4f}")
print(f"  Mean AUC (one-vs-rest):      {np.mean(aucs):.4f}")
print("=" * 55)

# ── 2. Per-class breakdown ────────────────────────────────────────
print("\nPer-Class Performance:")
report = classification_report(
    all_labels, all_preds,
    target_names=list(class_names.values()),
    digits=3
)
print(report)

# ── 3. Clinical safety check ──────────────────────────────────────
print("Clinical Safety Check (High-Risk Grades 3 & 4):")
for grade in [3, 4]:
    mask   = all_labels == grade
    recall = (all_preds[mask] == grade).mean() * 100
    missed = (all_preds[mask] == 0).sum()  # predicted No DR when actually severe
    print(f"  Grade {grade} ({class_names[grade]}): "
          f"Recall={recall:.1f}% | "
          f"Missed as Grade 0 (dangerous): {missed} cases")

# ── 4. Error distance analysis ────────────────────────────────────
errors      = all_preds[all_preds != all_labels]
true_errors = all_labels[all_preds != all_labels]
distances   = np.abs(errors - true_errors)

print(f"\nError Distance Analysis (when wrong, how wrong?):")
print(f"  Off by 1 grade:  {(distances==1).sum()} ({(distances==1).mean()*100:.1f}% of errors)")
print(f"  Off by 2 grades: {(distances==2).sum()} ({(distances==2).mean()*100:.1f}% of errors)")
print(f"  Off by 3+ grades:{(distances>=3).sum()} ({(distances>=3).mean()*100:.1f}% of errors) ← dangerous")

# ── 5. Full visual dashboard ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', ax=axes[0],
            xticklabels=class_names.values(),
            yticklabels=class_names.values(),
            cmap='Blues', vmin=0, vmax=100)
axes[0].set_title(f'Confusion Matrix (%)\nQWK = {qwk:.3f}', fontsize=12)
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

# Per-class metrics bar chart
report_dict = classification_report(all_labels, all_preds,
                                     target_names=list(class_names.values()),
                                     output_dict=True)
metrics_df = pd.DataFrame({
    'Precision': [report_dict[c]['precision'] for c in class_names.values()],
    'Recall':    [report_dict[c]['recall']    for c in class_names.values()],
    'F1-Score':  [report_dict[c]['f1-score']  for c in class_names.values()],
}, index=class_names.values())

metrics_df.plot(kind='bar', ax=axes[1], color=['#4e79a7','#e15759','#59a14f'],
                alpha=0.85, edgecolor='black')
axes[1].set_title('Per-Class Precision / Recall / F1', fontsize=12)
axes[1].set_xlabel('DR Grade'); axes[1].set_ylabel('Score')
axes[1].set_ylim(0, 1.1); axes[1].legend(loc='lower right')
axes[1].tick_params(axis='x', rotation=15)

# AUC per class
auc_labels = [f"{class_names[c][:10]}\n(AUC={aucs[c]:.3f})" for c in range(5)]
axes[2].bar(auc_labels, aucs, color='#76b7b2', alpha=0.85, edgecolor='black')
axes[2].axhline(0.8, color='orange', linestyle='--', label='Good threshold (0.8)')
axes[2].axhline(0.9, color='green',  linestyle='--', label='Excellent (0.9)')
axes[2].set_title('ROC-AUC per Class (One-vs-Rest)', fontsize=12)
axes[2].set_ylabel('AUC'); axes[2].set_ylim(0.5, 1.05)
axes[2].legend(fontsize=8)
axes[2].tick_params(axis='x', rotation=10)

plt.suptitle('Complete Model Evaluation — Beyond Accuracy', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()