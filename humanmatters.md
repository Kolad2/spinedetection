| Модель | Статья | GitHub | Результат на тестовом наборе |
|---|---|---|---|
| **SGHM (ResNet-50)** | [Robust Human Matting via Semantic Guidance](https://arxiv.org/abs/2210.05210) | [cxgincsu/SemanticGuidedHumanMatting](https://github.com/cxgincsu/SemanticGuidedHumanMatting) | **PPM-100:** MAD `0.00597`, MSE `0.00258` |
| **P3M-Net (ResNet-34)** | [Privacy-Preserving Portrait Matting](https://arxiv.org/abs/2104.14222) | [JizhiziLi/P3M](https://github.com/JizhiziLi/P3M) | **P3M-500-P, B:B:** SAD `8.73`, MSE `0.0026` |
| **MODNet** | [MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition](https://arxiv.org/abs/2011.11961) | [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) | **PPM-100, без trimap:** MAD `0.0086`, MSE `0.0044` |
| **ViTMatte-B** | [ViTMatte: Boosting Image Matting with Pretrained Plain Vision Transformers](https://arxiv.org/abs/2305.15272) | [hustvl/ViTMatte](https://github.com/hustvl/ViTMatte) | **Composition-1k:** SAD `20.33`, MSE `0.0030`, Grad `6.74`, Conn `14.78` |
| **RVM (ResNet-50)** | [Robust High-Resolution Video Matting with Temporal Guidance](https://arxiv.org/abs/2108.11515) | [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | **VideoMatte240K, 1920×1080:** MAD `5.81`, MSE `0.97`, Grad `9.65`, dtSSD `1.78` |

> Все перечисленные метрики являются ошибками: меньше — лучше.