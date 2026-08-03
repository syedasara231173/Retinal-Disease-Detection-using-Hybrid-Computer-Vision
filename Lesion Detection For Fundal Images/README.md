# DIP Step 3 — Classical Lesion Candidate Detection

**Diabetic Retinopathy (DR) Severity Grading Project**
**Module:** Classical Digital Image Processing (pre-deep-learning stage)

---

## Overview

Before the deep learning classifier ever sees a retinal image, this step runs two classical (non-learned) image processing pipelines to flag *where* early disease signs might be hiding: **microaneurysms** (the earliest visible sign of DR) and **hemorrhages** (a marker of more advanced disease). Think of it as giving the model a pair of reading glasses tuned specifically to spot the two kinds of "red flags" an ophthalmologist looks for first.

Both techniques operate only on the **green color channel** of the fundus image — a deliberate and clinically-grounded choice, since red lesions on a red-toned retina show maximum contrast in green light (this is standard practice in retinal image analysis).

---

## Part 1 — Microaneurysm (MA) Detection

| Aspect | Description |
|---|---|
| **Technique** | CLAHE contrast enhancement → image inversion → morphological **top-hat transform** → binary threshold → **blob detection** (filtered by area, circularity, inertia) |
| **What it does** | Isolates tiny, round, dark-red dots (5–80 px in area) scattered across the retina — the classical signature of a microaneurysm — and marks each candidate with a small red circle |
| **Why it's helpful here** | Microaneurysms are the **earliest detectable sign of DR**, often present in Grade 1 (Mild) images before any other lesion appears. A model that can be shown *where* these dots are (rather than just a global "sick/not sick" label) is diagnosing the same way a clinician does — lesion by lesion |
| **Why it's impressive** | This is a fully **unsupervised, zero-training** pipeline — no labeled microaneurysm data was needed. It reproduces, using only classical operators, the same localization step that dedicated MA-detection papers use as a baseline. It demonstrates image-processing fundamentals (CLAHE, top-hat morphology, blob geometry) doing real clinical work without a single neural network parameter |

**How the technique works, in plain terms:**
- **CLAHE** boosts local contrast in patches of the image, so faint dots that would otherwise blend into the background become visible — like adjusting brightness *locally* instead of over the whole photo.
- **Inversion** flips the image so dark red dots become bright spots, since blob detectors are built to find *bright* regions.
- **Top-hat transform** subtracts a "blurred version of itself" from the image, which cancels out large smooth structures (like the general retina background) and keeps only small, sharp features — exactly the size of a microaneurysm.
- **Blob detection** then filters what's left by size, roundness, and shape, throwing away noise and keeping only dot-like candidates.

Part 2 — Hemorrhage (HM) Detection
Aspect	Description
Technique	Same top-hat pipeline as above, but with a larger structuring element (11×11 vs 5×5) → binary threshold → connected component analysis filtered by pixel area (80–3000 px)
What it does	Identifies larger, irregular dark blotches on the retina and outlines each one with a yellow contour
Why it's helpful here	Hemorrhages indicate more advanced/severe DR than microaneurysms. Distinguishing "many small dots" (MA-heavy, milder) from "large dark blotches" (HM-heavy, more severe) gives the pipeline a proxy for disease severity, not just presence — directly relevant to a grading task rather than a simple detection task
Why it's impressive	Reusing the same top-hat foundation but tuning only the kernel size and area thresholds to catch a structurally different lesion type shows deliberate, principled parameter design rather than two unrelated ad-hoc scripts — one core technique, two clinically distinct outputs

How the technique works, in plain terms:

The larger structuring element (11×11 instead of 5×5) changes what counts as "small enough to keep" during the top-hat step, so this pass captures blotches instead of dots.
Connected component analysis groups adjacent dark pixels into single blobs and measures each blob's area, so tiny noise specks (too small) and large normal shadows/vessels (too big) are automatically excluded, keeping only hemorrhage-sized regions.
Combined Documentation
How the two pipelines work together

Both detectors run on the same enhanced green channel and are overlaid on a single annotated output:

🔴 Red circles = microaneurysm candidates
🟡 Yellow contours = hemorrhage candidates

This produces one image per sample that visually summarizes both early and advanced lesion signs side-by-side, alongside a count of each (MA candidates: n | HM candidates: n) that can later be correlated against the ground-truth DR grade.

Why this matters for the overall project
Value	Explanation
Interpretability	Deep learning DR classifiers are often criticized as "black boxes." This step gives a lesion-level, human-checkable explanation before any deep model is involved — a clinician (or grader) can see exactly what the pipeline is reacting to
No training data required	Fully classical — useful even with the limited/imbalanced labeled data typical of medical imaging datasets
Feature engineering potential	The MA/HM counts and locations can be fed as auxiliary features into the CNN pipeline later, or used to sanity-check that the classifier's attention aligns with real lesions
Bridges DIP theory and applied CV	Demonstrates CLAHE, morphological top-hat transforms, blob detection, and connected component analysis — core Digital Image Processing techniques — applied end-to-end to a real medical imaging problem
Assumptions & limitations (flagged explicitly)

These are important to state plainly, since they affect how the results should be read:

All numeric thresholds (clipLimit=3.0, tophat thresholds of 20/30, area ranges 5–80 px and 80–3000 px, minCircularity=0.5, minInertiaRatio=0.3) are heuristic starting values, not values tuned or validated against pixel-level ground-truth lesion annotations. They were chosen to be reasonable for a 224×224 resized image, not empirically optimized.
Blood vessels, eyelashes/artifacts, and imaging noise can also produce small dark blobs, so candidate counts should be read as "possible lesion regions," not a validated diagnostic count.
No dataset-level evaluation (precision/recall against labeled lesions) was performed in this step — it is a candidate-generation and visualization stage, not a validated detector.

This step corresponds to Step 3 of the classical DIP pipeline, preceding the deep learning classification stage of the project.