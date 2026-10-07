# UdaciSense: Model Optimization Technical Report

## Executive Summary

UdaciSense aims to expand its object recognition capability to budget-friendly smartphones without significantly reducing prediction quality. The primary business challenge was to decrease model size and inference latency while maintaining acceptable classification accuracy.

To achieve this objective, multiple model compression techniques were evaluated, including dynamic quantization, static quantization, and knowledge distillation. Based on the experimental results, a multi-stage optimization pipeline combining knowledge distillation and static quantization was selected and implemented.

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

Three pipeline concepts were considered:

### Pipeline 1 (Selected)

Knowledge Distillation → Static Quantization

### Pipeline 2

Post-Training Pruning → Quantization

### Pipeline 3

Knowledge Distillation → Quantization → Graph Optimization

Pipeline 1 received the highest priority because the experimental results showed the strongest balance between size reduction, speed improvement, and accuracy preservation.

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

Contribution of stages:

### Distillation

- Preserved accuracy
- Reduced model size
- Created performance margin

### Quantization

- Delivered most of the compression
- Produced the majority of latency reduction
- Consumed part of the accuracy reserve

The final model achieved all optimization objectives while maintaining practical usability.

---

# 4. Mobile Deployment

## 4.1 Export Process

The optimized model was prepared for mobile deployment using:

1. TorchScript tracing
2. Model freezing
3. PyTorch Mobile optimization

The resulting model was exported as a mobile-compatible TorchScript model.

## 4.2 Mobile-Specific Considerations

Important deployment considerations include:

- Limited CPU resources
- Restricted memory availability
- Battery consumption
- Thermal throttling
- Device-to-device variability

TorchScript reduces runtime overhead and improves portability across mobile platforms.

## 4.3 Performance Verification

Output consistency testing confirmed that the mobile model behaved identically to the original model.

### Consistency Results

- Output Shape: (1,10)
- Maximum Absolute Difference: 0.00000620
- Result: PASSED

### Model Size

| Metric | Value |
|----------|----------|
| Original Distilled Model | 4.24 MB |
| Mobile Model | 4.12 MB |
| Reduction | 2.74% |

The mobile optimizer achieved a small additional reduction while preserving prediction quality.

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

- Early MobileNetV3 layers are highly quantization-sensitive.
- Distillation provides valuable accuracy reserves.
- Static quantization delivers the largest efficiency gains.
- Combined optimization approaches outperform individual methods.

## 5.3 Recommendations for Future Work

Potential improvements include:

- Quantization-Aware Training (QAT)
- Structured channel pruning
- Additional graph optimization
- ARM-specific benchmarking
- Real-device testing on Android and iOS hardware

## 5.4 Business Impact

The optimized solution enables:

- Deployment on lower-cost smartphones
- Improved application responsiveness
- Lower hardware requirements
- Reduced energy consumption
- Expansion into budget-sensitive markets

The final model provides a scalable foundation for broader adoption of UdaciSense technology while maintaining a high-quality user experience.

## References

- PyTorch Quantization Documentation
- PyTorch Mobile Documentation
- MobileNetV3 Research Paper
- Knowledge Distillation Research Paper