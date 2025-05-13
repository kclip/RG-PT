
# Multi-Objective Hyperparameter Selection via Hypothesis Testing on Reliability Graphs

This repository contains the implementation and experiments for the paper **"Multi-Objective Hyperparameter Selection via Hypothesis Testing on Reliability Graphs"**. The codebase includes algorithms and experiments showcasing the proposed method (RG-PT) and its applications in various domains.

---

## Table of Contents

- [Overview](#overview)
- [Files and Structure](#files-and-structure)

---

## Overview

In the paper, we propose **RG-PT**, a novel method for multi-objective hyperparameter selection based on hypothesis testing on reliability graphs. The approach is demonstrated through experiments in the following domains:

- Sequence-to-Sequence Language Translation
- SVM Image Classification
- Radio Access Scheduling
- Object Detection

---

## Files and Structure

- **`dagger.py`**: Implementation of the **DAGGER** algorithm based on the paper *"A sequential algorithm for false discovery rate control on directed acyclic graphs"*.
- **`RG-PT.py`**: Implementation of our proposed method, **RG-PT**.
- **`SVM.py`**: Code for the **SVM image classification experiment**.
- **`seq2seq.py`**: Code for the **Sequence-to-Sequence Language Translation experiment**.
- **`Wireless.py`**: Code for the **radio access scheduling experiment**.
- **`Detection/`**: Contains the implementation of the **object detection algorithm** based on the paper *"Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control"*. You need to run /experiments/detection/experiment.py to reproduce our results.

---
