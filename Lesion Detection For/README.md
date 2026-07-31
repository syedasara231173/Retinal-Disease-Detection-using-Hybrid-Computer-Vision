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
