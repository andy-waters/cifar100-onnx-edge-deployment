# CIFAR-100 Model Training & Edge Deployment (ONNX)

This project demonstrates end-to-end machine learning workflows on the CIFAR-100 dataset using ResNet architectures. The implementation covers **training, evaluation, model export, and preparation for edge deployment** (e.g., NVIDIA Jetson devices).  

---

## ✨ Features
- **ResNet on CIFAR-100**: Train and evaluate ResNet models using PyTorch.  
- **Model Export**: Convert trained models to **ONNX** (edge deployment) and **YAML**.  
- **Data Pipeline**: Automated CIFAR-100 download, preprocessing, and train/val/test splits.  
- **Deployment Prep**: Stubbed HEF converter for **AI HAT** edge devices (future work).  
- **Utilities**: Training, testing, and conversion scripts to support reproducible experiments.  

---

## 📂 Project Structure
```
ResNet.ipynb                # Jupyter notebook for experimentation
output/                     # Model checkpoints and exports
scripts/                    # Core Python scripts
  ├── cifar100_data_provider.py
  ├── hef_converter.py       # (stubbed, for AI HAT edge device)
  ├── onnx_converter.py
  ├── onnx_yaml_converter.py
  ├── resnet_model.py
  ├── tester.py
  └── trainer.py
data/                       # CIFAR-100 dataset (auto-downloaded)
```

---

## 🚀 Getting Started
1. **Install dependencies**
   ```bash
   pip install torch torchvision onnx pyyaml tqdm scikit-learn
   ```
2. **Prepare CIFAR-100 dataset**  
   - Automatically downloaded during training if not already present.  
   - `cifar100_data_provider.py` handles splitting and preprocessing.  

3. **Train a model**
   ```bash
   python scripts/trainer.py
   ```

4. **Evaluate a trained model**
   ```bash
   python scripts/tester.py
   ```

5. **Export models**
   - **ONNX**:  
     ```bash
     python scripts/onnx_converter.py
     ```
   - **YAML**:  
     ```bash
     python scripts/onnx_yaml_converter.py
     ```
   - **HEF (AI HAT)**: Work in progress, see `hef_converter.py`.

---

## 📒 Notebooks
- **ResNet.ipynb**: Interactive training and evaluation with CIFAR-100.

---

## 🔮 Future Work
- Complete **knowledge distillation** between teacher and student ResNet models.  
- Finalize **HEF converter** for AI HAT edge devices.  
- Extend deployment pipeline with additional model architectures.  

---

## 📄 License
This project is licensed under the MIT License.  
