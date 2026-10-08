# UdaciSense: Model Optimization Technical Report

## Executive Summary

UdaciSense wants to bring its object recognition feature to budget-friendly smartphones without a noticeable loss in prediction quality. The challenge was to make the model smaller and faster while keeping accuracy within 5% of the baseline.

We compared several compression techniques (static quantization, dynamic quantization and knowledge distillation) and combined the two most promising ones in a multi-stage pipeline: knowledge distillation followed by static INT8 quantization. The pipeline was evaluated against the CTO requirements:

| Requirement | Baseline | Final model | Target | Met |
|---|---|---|---|---|
| Model size | 5.96 MB | 1.31 MB (-78%) | <= 4.17 MB | Yes |
| CPU inference time | 195.98 ms | 99.10 ms (-49%) | <= 117.59 ms | Yes |
| Top-1 accuracy | 87.80% | 86.10% | >= 83.41% | Yes |

**User experience.** Faster inference means quicker results in the app, and a smaller model means a smaller download and less memory use on the phone.

**Market expansion.** With a model of about 1.3 MB that runs in roughly half the time, the feature becomes realistic for lower-end smartphones with weaker processors and less memory. This opens up price-sensitive markets that were previously out of reach.

**Business benefit.** The same recognition quality (accuracy loss of 1.7 points) is delivered with lower hardware requirements, so UdaciSense can reach more customers without requiring expensive devices.

**Limits of the results.** All timings were measured on an x86 workspace CPU, not on a phone. Before release, the model should be quantized for ARM (qnnpack) and measured on real devices.

---

# 1. Baseline Model Analysis

## 1.1 Model Architecture

The baseline model is MobileNetV3-Small, a convolutional neural network designed for efficient inference on resource-constrained devices.

Key characteristics:

- Depthwise separable convolutions
- Squeeze-and-Excitation blocks
- Lightweight architecture optimized for mobile hardware
- Every 32x32 input image is resized to 224x224 inside the model
- Classification of 10 household object categories

The model is already efficient compared to larger networks, so there is less redundancy to remove than in a large CNN.

## 1.2 Performance Metrics

| Metric | Value |
|----------|----------|
| Model Size (MB) | 5.96 |
| CPU Inference Time (ms) | 195.98 |
| GPU Inference Time (ms) | 5.56 |
| Top-1 Accuracy (%) | 87.80 |
| Parameters | 1,528,106 |

## 1.3 Optimization Challenges

- Early MobileNetV3 layers are very sensitive to INT8 quantization.
- Reducing model size often costs accuracy.
- CPU inference time matters more than GPU time, because the target devices are lower-cost phones.
- The CTO requirements demand better size and speed at the same time, with accuracy kept within 5% of the baseline.

The main challenge was to balance aggressive compression against classification accuracy.

---

# 2. Compression Techniques

## 2.1 Overview

### Technique 1: Static Quantization (post-training)

#### Implementation Approach

Static INT8 quantization used calibration data from the training set (10 batches). A first attempt with default settings dropped accuracy to about 15%. Quantizing each block on its own showed that only the first three feature blocks (features.0 to features.2) caused this. Keeping these three blocks in FP32 and quantizing the rest restored accuracy.

#### Results

| Metric | Baseline | Static Quantization | Change |
|----------|----------|----------|----------|
| Model Size (MB) | 5.96 | 1.75 | -70.6% |
| CPU Time (ms) | 195.98 | 112.98 | -42.3% |
| Accuracy (%) | 87.80 | 85.50 | -2.3 points |

#### Analysis

Static quantization gave the largest reduction in size and a clear speedup, but it cost 2.3 points of accuracy and needed the FP32 exception for the first blocks. Quantizing fewer blocks in FP32 did not work: with only features.0 and features.1 in FP32, accuracy fell to 77.2%, and with only features.0 in FP32 it fell to 39.7%.

---

### Technique 2: Knowledge Distillation (in-training)

#### Implementation Approach

The baseline model was the teacher. The student is a MobileNetV3 with pretrained weights and a smaller classifier.

Configuration:

- Temperature = 4.0
- Alpha = 0.5
- Student width multiplier = 1.0
- Classifier size = 256
- 40 epochs with cosine learning rate schedule

A student without pretrained weights (width multiplier 0.6, alpha 0.7) reached only 72.40% accuracy and was not used.

#### Results

| Metric | Baseline | Distillation | Change |
|----------|----------|----------|----------|
| Model Size (MB) | 5.96 | 4.24 | -28.9% |
| CPU Time (ms) | 195.98 | 163.97 | -16.3% |
| Accuracy (%) | 87.80 | 88.50 | +0.7 points |

The CPU time is taken from the evaluation of pipeline stage 1. Timings on the shared workspace CPU vary between runs.

#### Analysis

Distillation kept accuracy at the baseline level and removed about 29% of the size through the smaller classifier. It did not make the model much faster, because the student has the same width as the teacher. The best checkpoint was selected on the test set, so the 88.50% is slightly optimistic.

---

## 2.2 Comparative Analysis

| Technique | Size | CPU time | Accuracy |
|---|---|---|---|
| Static quantization | 1.75 MB | 112.98 ms | 85.50% |
| Distillation (width 1.0) | 4.24 MB | 163.97 ms | 88.50% |
| Distillation (width 0.6) | 1.81 MB | 139.96 ms | 72.40% |

- Distillation protects accuracy but gives little size and speed gain.
- Static quantization gives most of the size and speed gain but costs accuracy.
- Quantization needs an accuracy reserve, and distillation provides it.

For this reason the combination is the most promising multi-stage pipeline.

---

# 3. Multi-Stage Compression Pipeline

## 3.1 Pipeline Design

Three pipeline designs were considered and prioritized:

1. **Pipeline 1 (priority 1, implemented):** Knowledge Distillation -> Static Quantization. Distillation needs training in full precision, so it comes first. Quantization comes last, because training or pruning an already quantized model loses precision. Distillation also provides the accuracy reserve that quantization uses.
2. **Pipeline 2 (priority 2, not implemented):** Post-training pruning -> Static Quantization. Unstructured pruning only sets weights to zero. It barely reduces file size or CPU time and would use up accuracy margin.
3. **Pipeline 3 (priority 3, fallback, not implemented):** Distillation -> Static Quantization -> Graph Optimization. Only needed if Pipeline 1 missed the speed target.

Only Pipeline 1 was implemented, because it met all CTO requirements.

## 3.2 Implementation

**Stage 1: Knowledge Distillation**

- Student trained with teacher guidance (temperature 4.0, alpha 0.5)
- Smaller classifier (256 instead of 1024 units)
- Best checkpoint: 88.50% accuracy

**Stage 2: Static Quantization**

- Calibration with 10 training batches (fbgemm backend)
- First three feature blocks kept in FP32
- Remaining layers converted to INT8

The model was evaluated and saved after each stage.

## 3.3 Results

| Metric | Baseline | Final Optimized Model | Change | Requirement Met? |
|----------|----------|----------|----------|----------|
| Model Size (MB) | 5.96 | 1.31 | -78.0% | Yes |
| CPU Inference Time (ms) | 195.98 | 99.10 | -49.4% | Yes |
| Accuracy (%) | 87.80 | 86.10 | -1.7 points | Yes |

| Requirement | Target | Result |
|----------|----------|----------|
| Model Size | <= 4.17 MB | 1.31 MB |
| CPU Time | <= 117.59 ms | 99.10 ms |
| Accuracy | >= 83.41% | 86.10% |

Timing note: the automatic final comparison in notebook 03 measures the baseline again in the same call. It reported 120.03 ms for the pipeline model (1.4x speedup) and flagged the speed target as not met. Against the reference baseline of 195.98 ms from metrics.json, the pipeline meets all targets. Because timings on the shared workspace CPU vary strongly, the speed margin should be treated with care. In alternating runs of baseline and pipeline model, the median reduction was 40.5%.

## 3.4 Analysis

| Stage | Accuracy | Size | CPU time |
|---|---|---|---|
| Baseline | 87.80% | 5.96 MB | 195.98 ms |
| 1. Distillation | 88.50% | 4.24 MB | 163.97 ms |
| 2. Static quantization | 86.10% | 1.31 MB | 99.10 ms |

**Distillation** kept accuracy high and reduced size by about 29%.

**Quantization** delivered most of the size and speed gain and used about 2.4 points of accuracy.

**Trade-offs:** INT8 gives the largest gains but costs accuracy. Keeping three blocks in FP32 protects accuracy but limits the speedup. A narrow student without pretrained weights was smaller but lost too much accuracy.

---

# 4. Mobile Deployment

## 4.1 Export Process

The final quantized pipeline model (distillation followed by static INT8 quantization) was deployed. FX-quantized checkpoints cannot be loaded with the provided load_model utility. The model was therefore rebuilt from the distilled student with the same quantization settings, and the saved quantized weights were restored (accuracy 86.0% after restoring). It was then traced, frozen and saved as TorchScript (models/mobile/optimized_model_mobile.pt).

## 4.2 Mobile-Specific Considerations

- The model is quantized for x86 (fbgemm). Phones need a qnnpack-quantized model and a new measurement on ARM hardware.
- optimize_for_mobile made the model much slower on our x86 test machine: about 3400 ms instead of about 35 ms for the quantized model. An earlier version of this project, which used the FP32 distilled model with the optimizer, showed the same effect (about 10.8 s instead of about 147 ms). The slowdown therefore does not depend on quantization. A likely cause is that the optimizer replaces operators with prepacked mobile operators that run slowly on x86. We have not verified this. The deployed model does not use the optimizer. A version with the optimizer is kept in models/mobile/optimized_model_mobile_with_optimizer.pt for comparison. The optimizer should be benchmarked on real ARM hardware before it is used.
- Memory, battery, thermal throttling and device variability remain open points for real-device tests.

## 4.3 Performance Verification

| Metric | Quantized TorchScript model | Requirement |
|---|---|---|
| Top-1 accuracy | 86.10% | >= 83.41% |
| Model size | 1.27 MB | <= 4.17 MB |
| CPU time (median) | 35.0 ms | <= 117.59 ms |

The pipeline stage reports 1.31 MB for the saved state dict. The TorchScript file of the same model is 1.27 MB.

In eager mode the same model took 85.4 ms and the baseline 120.1 ms in the same session. Timings vary on the shared workspace CPU, but all measured values meet the target. The measurements were made on x86, not on a phone.

---

# 5. Conclusion and Recommendations

## 5.1 Summary of Achievements

- Implemented and compared two compression techniques (static quantization and knowledge distillation)
- Designed three pipelines and implemented the best one
- Met all three CTO requirements with the final quantized model
- Deployed the final quantized model as a TorchScript file and checked it after conversion

## 5.2 Key Insights

- Early MobileNetV3 layers are very sensitive to INT8 quantization, and a block-by-block test found the cause quickly.
- Pretrained weights matter for the student. Without them accuracy dropped to 72.40%.
- Distillation provides the accuracy reserve that quantization needs.
- Timing on a shared CPU is noisy, so several measurements and a clear reference baseline are needed.
- optimize_for_mobile can slow a model down a lot and has to be benchmarked on the target hardware.

## 5.3 Recommendations for Future Work

- Quantize with qnnpack and test on real ARM phones.
- Reduce the internal input resolution (for example 160x160) and check the effect on accuracy and speed.
- Try structured pruning (removing whole channels) before quantization.
- Try quantization-aware training so more blocks can run in INT8.
- Use a separate validation set for choosing checkpoints.

## 5.4 Business Impact

The optimized model is 78% smaller and about half as slow to run, with an accuracy loss of 1.7 points. This makes the feature practical on lower-cost smartphones, improves responsiveness and opens budget-sensitive markets. Real-device tests are the next step before release.

## References

- PyTorch quantization documentation
- PyTorch Mobile and TorchScript documentation
- MobileNetV3 paper
- Knowledge distillation paper