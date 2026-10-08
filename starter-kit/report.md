# UdaciSense: Model Optimization Technical Report

## Executive Summary

UdaciSense aims to expand its object recognition capability to budget-friendly smartphones without significantly reducing prediction quality. The primary business challenge was to decrease model size and inference latency while maintaining acceptable classification accuracy.

To achieve this objective, multiple model compression techniques were evaluated, including static quantization, dynamic quantization, and knowledge distillation. Based on the experimental results, a multi-stage optimization pipeline combining knowledge distillation and static quantization was selected and implemented.

The final optimized model successfully exceeded all CTO requirements. Model size was reduced from 5.96 MB to 1.31 MB, CPU inference latency decreased from 195.98 ms to 99.10 ms, and classification accuracy remained at 86.10%, well above the minimum acceptable threshold of 83.41%.

These improvements enable deployment on lower-cost smartphones, improve user responsiveness, reduce computational requirements, and expand the potential customer base without requiring expensive hardware upgrades.

---

# 1. Baseline Model Analysis

## 1.1 Model Architecture

The baseline model is based on MobileNetV3, a convolutional neural network specifically designed for efficient deployment on resource-constrained devices.

Key characteristics include:

- Depthwise separable convolutions
- Squeeze-and-Excitation blocks
- Lightweight architecture optimized for mobile hardware
- Input images resized from CIFAR resolution to MobileNetV3 input resolution
- Classification of 10 household object categories

The model already provides good efficiency compared to larger convolutional architectures, making it a strong starting point for additional compression.

## 1.2 Performance Metrics

| Metric | Value |
|----------|----------|
| Model Size (MB) | 5.96 |
| CPU Inference Time (ms) | 195.98 |
| GPU Inference Time (ms) | 5.56 |
| Top-1 Accuracy (%) | 87.80 |
| Parameters | 1,528,106 |

## 1.3 Optimization Challenges

Several challenges influenced optimization potential:

- Early MobileNetV3 layers were highly sensitive to quantization.
- Reducing model size often caused accuracy degradation.
- CPU inference performance was more critical than GPU performance because the target platform consists of lower-cost mobile devices.
- The CTO requirements demanded simultaneous improvements in size and speed while preserving accuracy.

The main challenge was balancing aggressive compression against classification performance.

---

# 2. Compression Techniques

## 2.1 Overview

### Technique 1: Static Quantization

#### Implementation Approach

Static INT8 quantization was applied using calibration data from the training dataset. Quantization sensitivity analysis showed that the first three feature blocks were highly sensitive, so they were retained in FP32 while later layers were quantized.

#### Results

| Metric | Baseline | Static Quantization | Change (%) |
|----------|----------|----------|----------|
| Model Size (MB) | 5.96 | 1.75 | -70.6 |
| CPU Time (ms) | 195.98 | 99.10 | -49.4 |
| Accuracy (%) | 87.80 | 85.50 | -2.3 points |

#### Analysis

Static quantization achieved the largest reduction in both model size and latency. However, it introduced some accuracy degradation. Quantization alone appeared promising but required additional accuracy preservation techniques.

---

### Technique 2: Knowledge Distillation

#### Implementation Approach

Knowledge distillation was performed using the baseline model as teacher and a MobileNetV3 student model with a reduced classifier size.

Configuration:

- Temperature = 4.0
- Alpha = 0.5
- Width Multiplier = 1.0
- Linear Layer Size = 256

#### Results

| Metric | Baseline | Distillation | Change (%) |
|----------|----------|----------|----------|
| Model Size (MB) | 5.96 | 4.24 | -28.9 |
| CPU Time (ms) | 195.98 | 163.97 | -16.3 |
| Accuracy (%) | 87.80 | 88.50 | +0.7 |

#### Analysis

Knowledge distillation preserved and slightly improved accuracy while reducing model size. Although speed improvements were moderate, distillation provided a strong foundation for a subsequent quantization stage.

---

## 2.2 Comparative Analysis

Knowledge distillation and static quantization addressed different optimization goals.

- Distillation provided excellent accuracy preservation.
- Static quantization delivered the largest improvements in size and inference speed.
- Quantization required an accuracy buffer to remain above business requirements.
- Distillation generated that accuracy buffer.

As a result, combining both techniques was the most promising strategy for a multi-stage optimization pipeline.

---

# 3. Multi-Stage Compression Pipeline

## 3.1 Pipeline Design

Three pipeline concepts were considered during the planning stage.

### Pipeline 1 (Selected)

Knowledge Distillation → Static Quantization

### Pipeline 2

Post-Training Pruning → Quantization

### Pipeline 3

Knowledge Distillation → Quantization → Graph Optimization

Pipeline 1 received the highest priority because the experimental results showed the strongest balance between size reduction, speed improvement, and accuracy preservation.

Only Pipeline 1 was implemented and evaluated. Pipelines 2 and 3 were retained as alternative design options and were not executed because Pipeline 1 successfully met all CTO requirements.

## 3.2 Implementation

The selected pipeline executed the following stages:

### Stage 1

Knowledge Distillation

- Student model trained using teacher guidance.
- Reduced classifier dimension.
- Preserved classification quality.

### Stage 2

Static Quantization

- Calibration using training data.
- Sensitive early layers retained in FP32.
- Remaining layers converted to INT8.

Intermediate checkpoints were stored and evaluated after each stage.

## 3.3 Results

| Metric | Baseline | Final Optimized Model | Change (%) | Requirement Met? |
|----------|----------|----------|----------|----------|
| Model Size (MB) | 5.96 | 1.31 | -78.0 | Yes |
| CPU Inference Time (ms) | 195.98 | 99.10 | -49.4 | Yes |
| Accuracy (%) | 87.80 | 86.10 | -1.7 points | Yes |

### CTO Requirements

| Requirement | Target | Result |
|----------|----------|----------|
| Model Size | <= 4.17 MB | 1.31 MB |
| CPU Time | <= 117.59 ms | 99.10 ms |
| Accuracy | >= 83.41 % | 86.10 % |

All requirements were successfully achieved.

## 3.4 Analysis

The pipeline demonstrated that combining complementary techniques is significantly more effective than applying either technique independently.

### Contribution of Distillation

- Preserved accuracy
- Reduced model size
- Created performance margin

### Contribution of Quantization

- Delivered most of the compression
- Produced the majority of latency reduction
- Consumed part of the accuracy reserve

The final model achieved all optimization objectives while maintaining practical usability.

---

# 4. Mobile Deployment

## 4.1 Export Process

The final quantized pipeline model (distillation followed by static INT8 quantization) was deployed. FX-quantized checkpoints cannot be loaded with the provided load_model utility, so the model was rebuilt from the distilled student with the same quantization settings, the saved quantized weights were restored (accuracy 86.0% after restoring), and the model was traced, frozen and saved as TorchScript (models/mobile/optimized_model_mobile.pt).

## 4.2 Mobile-Specific Considerations

- The model is quantized for x86 (fbgemm). Phones need a qnnpack-quantized model and a new measurement on ARM hardware.
- optimize_for_mobile made the model about 100 times slower on our x86 test machine (about 3400 ms instead of about 35 ms). The likely cause is its rewrite of operators for ARM kernels (not verified). The deployed model does not use it. A version with the optimizer is kept in models/mobile/optimized_model_mobile_with_optimizer.pt for comparison.
- Memory, battery, thermal throttling and device variability remain open points for real-device tests.

## 4.3 Performance Verification

| Metric | Quantized TorchScript model | Requirement |
|---|---|---|
| Top-1 accuracy | 86.10% | >= 83.41% |
| Model size | 1.27 MB | <= 4.17 MB |
| CPU time (median) | 35.0 ms | <= 117.59 ms |

In eager mode the same model took 85.4 ms and the baseline 120.1 ms in the same session. Timings vary on the shared workspace CPU, but all measured values meet the target. The measurements were made on x86, not on a phone.
---

# 5. Conclusion and Recommendations

## 5.1 Summary of Achievements

The project successfully:

- Implemented multiple compression techniques
- Designed and deployed a multi-stage optimization pipeline
- Met all CTO performance requirements
- Produced a mobile-compatible TorchScript deployment artifact

## 5.2 Key Insights

Important findings include:

- Early MobileNetV3 layers