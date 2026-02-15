# Emoji Transfer

This repository provides a highly integrated environment for expression migration (Facial Driving). By leveraging Stable Diffusion and deep learning technologies, it synchronizes facial expression details from a driving video/image onto a static reference photo in real-time.

---

## 📋 Environment Specifications

- **Core Framework**: PyTorch 2.8.0 + CUDA 12.8
- **Frontend Interface**: Gradio 5.x
- **Inference Engine**: ONNX Runtime GPU (Accelerated via CUDA 12.x)
- **Image Processing**: OpenCV-Python-Headless / Scikit-Image / Pillow
- **Model Distribution**: Modelscope / HuggingFace Mirror

---

## 🚀 Quick Start

### 1. Environment Check
Run **`launch.py`** (the graphical launcher) located in the root directory.
* **Built-in environment**: Status should display `✅ ready`.
* **GPU Status**: If the graphics card model is displayed, GPU acceleration is enabled.
* * **Network status**: make sure you connect WI-IF！！！。

### 2. Model Initialization (First Run Only)
Click the **"Initialize Environment"** button on the interface.
* The script will automatically detect and download missing weights into the `models/` folder.
* **Note**: Total weights are approximately 8GB. For users in certain regions, the system will automatically switch to the Modelscope mirror for faster downloads.

### 3. Launch Application
Click **"Start Application"**. Once the log displays `Running on local URL: http://127.0.0.1:7860`, the browser interface will pop up automatically.

---

## 🛠️ VRAM Optimization

A GPU with **>= 8GB VRAM** is recommended:
1. **Input Control**: It is recommended that uploaded image resolutions do not exceed `512x512`.
2. **System Settings**: Ensure Windows **Virtual Memory (Page File)** is manually set to **32GB** or higher to prevent crashes during model loading.
3. **Advanced Tuning**: You can modify the `pipeline_dict_len` value in `app.py` based on your VRAM size. This value represents the number of pre-loaded models.

---

## ❌ Troubleshooting

### 1. Application Crash (Error Code: 0xC0000005)
* **Symptoms**: The console reports `Access Violation` and exits.
* **Causes**: Usually insufficient virtual memory or underlying DLL (OpenCV/Torch) conflicts.
* **Solution**: Close all high-memory applications (e.g., browsers, live wallpapers) and ensure adequate virtual memory.

### 2. Interface displays `Error: @@ huggingface-cli login`?
* **Analysis**: Local model files are incomplete. This occurs if the automatic download is interrupted due to network issues or insufficient memory.
* **Solution**:
    * **Activate Modelscope Mirror**: Run the following in your project directory:
      ```bash
      .\env\python.exe -m pip install modelscope
      ```
    * **Manual Check**: Inspect the `model_cache/huggingface` folder to ensure large `.safetensors` or `.bin` files are fully present.

---

## 📂 Core Directory Structure

- `env/`: Independent Python runtime environment (must be in the root directory).
- `model_cache/huggingface`: Stores core weights such as RealisticVision, VAE, and ControlNet.
- `app.py`: Main Gradio application.
- `launch.py`: GUI launcher with log monitoring.
- `Download and loading.py`: Resource pre-check and model loading script.

---

## 📢 Release & Special Notes

### **1. Minimalist Release Note**
* **Standalone Package**: This repository does not pre-include the `env` or model files to reduce download size. You must use the launcher to initialize them.
* **Internet Required**: A stable internet connection is required for the initial 12GB setup.

### **2. Disclaimer**
* **Legal Use**: This project is for academic exchange only. It is strictly forbidden to use this project to create or spread Deepfake videos against the will of others.
* **Liability**: Users are solely responsible for the content generated using this tool.

---

## 💖 Acknowledgements

The core algorithm of this project is built upon the **HelloMeme** open-source project and has been refined by multiple developers.

* **Core Algorithm**: Sincere thanks to the original **HelloMeme** authors and their team.
  - [GitHub - HelloMeme](https://github.com/juechen-info/HelloMeme)
* **Partners**: Thanks to all collaborators who provided help during development, testing, and optimization.
* **Special Thanks**: Deepest gratitude to my friend **Lin Xiaoru** for the unwavering support and companionship throughout the development of this project.

It is through these excellent open-source works and the support of friends that this project was successfully completed.
