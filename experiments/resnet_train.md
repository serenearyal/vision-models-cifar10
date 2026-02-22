# ResNet Training Results

- 10 epochs: ~86% accuracy
- 30 epochs: ~92% accuracy

---

## Architecture Diagram

### Complete Network Flow

```
Input: [batch=128, channels=3, height=32, width=32]
   │
   ▼
┌──────────────────────────────────────────────┐
│  Initial Conv Block                          │
│  Conv2d(3→32, kernel=3x3, padding=1)        │
│  + BatchNorm2d(32) + ReLU                   │
└──────────────────────────────────────────────┘
   │  Shape: [128, 32, 32, 32]
   ▼
┌──────────────────────────────────────────────┐
│  LAYER 1 (stride=1, NO downsampling)        │
│  ┌────────────────────────────────────────┐ │
│  │ ResidualBlock #1 (32→32, stride=1)     │ │
│  │  ┌──────────────────┐                  │ │
│  │  │ Main Path:       │    Shortcut      │ │
│  │  │ Conv 3x3 (s=1)   │    (Identity)    │ │
│  │  │ BN + ReLU        │        │         │ │
│  │  │ Conv 3x3 (s=1)   │        │         │ │
│  │  │ BN               │        │         │ │
│  │  └────────┬─────────┘        │         │ │
│  │           └────────(+)───────┘         │ │
│  │                    │                   │ │
│  │                  ReLU                  │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │ ResidualBlock #2 (32→32, stride=1)     │ │
│  │  (Same structure as Block #1)          │ │
│  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
   │  Shape: [128, 32, 32, 32] ← NO size change
   ▼
┌──────────────────────────────────────────────┐
│  LAYER 2 (stride=2, DOWNSAMPLE)             │
│  ┌────────────────────────────────────────┐ │
│  │ ResidualBlock #1 (32→64, stride=2)     │ │
│  │  ┌──────────────────┐                  │ │
│  │  │ Main Path:       │    Shortcut      │ │
│  │  │ Conv 3x3 (s=2)   │  Conv 1x1 (s=2)  │ │
│  │  │ BN + ReLU        │  + BN            │ │
│  │  │ Conv 3x3 (s=1)   │        │         │ │
│  │  │ BN               │        │         │ │
│  │  └────────┬─────────┘        │         │ │
│  │           └────────(+)───────┘         │ │
│  │                    │                   │ │
│  │                  ReLU                  │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │ ResidualBlock #2 (64→64, stride=1)     │ │
│  │  Shortcut: Identity (no dim change)    │ │
│  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
   │  Shape: [128, 64, 16, 16] ← Halved spatial dims
   ▼
┌──────────────────────────────────────────────┐
│  LAYER 3 (stride=2, DOWNSAMPLE)             │
│  ┌────────────────────────────────────────┐ │
│  │ ResidualBlock #1 (64→128, stride=2)    │ │
│  │  Main: Conv(s=2) increases channels    │ │
│  │  Shortcut: 1x1 Conv(s=2) to match dims │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │ ResidualBlock #2 (128→128, stride=1)   │ │
│  │  Shortcut: Identity                    │ │
│  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
   │  Shape: [128, 128, 8, 8] ← Halved again
   ▼
┌──────────────────────────────────────────────┐
│  Global Average Pooling                      │
│  AdaptiveAvgPool2d(output_size=1)           │
│  Averages each 8x8 feature map → 1x1        │
└──────────────────────────────────────────────┘
   │  Shape: [128, 128, 1, 1]
   ▼
┌──────────────────────────────────────────────┐
│  Flatten                                     │
│  view(batch_size, -1)                       │
└──────────────────────────────────────────────┘
   │  Shape: [128, 128]
   ▼
┌──────────────────────────────────────────────┐
│  Fully Connected Layer                       │
│  Linear(128 → 10)                           │
└──────────────────────────────────────────────┘
   │  Shape: [128, 10]
   ▼
Output: Class Logits (10 CIFAR-10 classes)
```

---

## ResidualBlock Internals

### Key Concept: H(x) = F(x) + x

```
                Input x
                   │
        ┌──────────┴──────────┐
        │                     │
   Main Path F(x)        Shortcut x
        │                     │
    ┌───▼────┐                │
    │ Conv1  │                │
    │ 3x3    │                │
    │stride=s│           (Identity OR
    └───┬────┘            1x1 Conv
        │                 if needed)
    ┌───▼────┐                │
    │   BN   │                │
    └───┬────┘                │
        │                     │
    ┌───▼────┐                │
    │  ReLU  │                │
    └───┬────┘                │
        │                     │
    ┌───▼────┐                │
    │ Conv2  │                │
    │ 3x3    │                │
    │stride=1│                │
    └───┬────┘                │
        │                     │
    ┌───▼────┐                │
    │   BN   │                │
    └───┬────┘                │
        │                     │
        └──────────┬──────────┘
                   │
              Element-wise
                 ADD (+)
                   │
              ┌────▼────┐
              │  ReLU   │
              └────┬────┘
                   │
                Output
```

### When Does Shortcut Need Adjustment?

**Case 1: stride=1 AND in_channels=out_channels**

- Shortcut = Identity (empty Sequential)
- Input and output shapes match perfectly
- Direct addition works

**Case 2: stride≠1 OR in_channels≠out_channels**

- Shortcut = 1x1 Conv + BN
- Adjusts spatial dimensions (via stride) and/or channels
- Makes shapes compatible for addition

---

## Spatial Dimension Changes Throughout Network

```
Input Image:           [128, 3, 32, 32]
                              ↓
Initial Conv:          [128, 32, 32, 32]  ← stride=1, padding=1
                              ↓
Layer 1 (stride=1):    [128, 32, 32, 32]  ← NO change
                              ↓
Layer 2 (stride=2):    [128, 64, 16, 16]  ← HALVED (32→16)
                              ↓
Layer 3 (stride=2):    [128, 128, 8, 8]   ← HALVED (16→8)
                              ↓
AvgPool:               [128, 128, 1, 1]   ← Pooled to 1x1
                              ↓
Flatten:               [128, 128]
                              ↓
FC Layer:              [128, 10]          ← Class scores
```

---
