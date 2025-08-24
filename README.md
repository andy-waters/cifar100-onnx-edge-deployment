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
requirements.txt            # dependencies for the project
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
Dockerfile.cpu              # CPU-only environment (Python 3.12, PyTorch)
Dockerfile.cuda             # CUDA-enabled environment (Python 3.12, PyTorch + cu121)
```

---

## 🚀 Getting Started (Local Python)
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
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

## 🐳 Running with Docker

This project includes two Dockerfiles:

- **Dockerfile.cpu** → For training and testing on CPU (works everywhere, slower).  
- **Dockerfile.cuda** → For training on NVIDIA GPUs with CUDA 12.1 (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).  

### 1. Build the image
**CPU version**:
```bash
docker build -t cifarproject:py312-cpu -f Dockerfile.cpu .
```

**CUDA version**:
```bash
docker build -t cifarproject:py312-cuda121 -f Dockerfile.cuda .
```

### 2. Run training
Mount your code + cache so datasets and logs persist:

**CPU**:
```bash
docker run --rm -it   -v "$PWD":/app   -v "$HOME/.cache/torch":/home/app/.cache/torch   cifarproject:py312-cpu   python scripts/trainer.py
```

**CUDA (GPU)**:
```bash
docker run --rm -it --gpus all   -v "$PWD":/app   -v "$HOME/.cache/torch":/home/app/.cache/torch   cifarproject:py312-cuda121   python scripts/trainer.py
```

### 3. Run evaluation
```bash
docker run --rm -it   -v "$PWD":/app   cifarproject:py312-cpu   python scripts/tester.py
```

### 4. Quick smoke test
Verify PyTorch is installed and see if CUDA is available:
```bash
docker run --rm -it cifarproject:py312-cpu   python -c "import torch; print('Torch', torch.__version__, 'CUDA?', torch.cuda.is_available())"

docker run --rm -it --gpus all cifarproject:py312-cuda121   python -c "import torch; print('Torch', torch.__version__, 'CUDA?', torch.cuda.is_available())"
```

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
