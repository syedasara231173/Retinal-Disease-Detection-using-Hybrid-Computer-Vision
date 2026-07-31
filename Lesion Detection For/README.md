# DIP Step 3 — Classical Lesion Candidate Detection

**Diabetic Retinopathy (DR) Severity Grading Project**
**Module:** Classical Digital Image Processing (pre-deep-learning stage)

---

## Overview

Before the deep learning classifier ever sees a retinal image, this step runs two classical (non-learned) image processing pipelines to flag *where* early disease signs might be hiding: **microaneurysms** (the earliest visible sign of DR) and **hemorrhages** (a marker of more advanced disease). Think of it as giving the model a pair of reading glasses tuned specifically to spot the two kinds of "red flags" an ophthalmologist looks for first.

Both techniques operate only on the **green color channel** of the fundus image — a deliberate and clinically-grounded choice, since red lesions on a red-toned retina show maximum contrast in green light (this is standard practice in retinal image analysis).
