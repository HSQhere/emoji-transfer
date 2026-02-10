# HelloMeme Emoji Transfer (Windows 一键集成版)

本仓库提供了一个高度集成的表情迁移（人脸驱动）环境。通过一张静态照片（参考图）和一段视频（驱动视频），利用 Stable Diffusion 和深度学习技术，将视频中的表情细节实时同步到静态照片的人物脸上。



---

## 📋 环境规格

- **核心框架**: PyTorch 2.8.0 + CUDA 12.8
- **前端交互**: Gradio 5.x
- **推理引擎**: ONNX Runtime GPU (CUDA 12.x 加速)
- **图像核心**: OpenCV-Python-Headless / Scikit-Image / Pillow
- **模型分发**: Modelscope / HuggingFace Mirror

---

## 🚀 快速开始

### 1. 运行环境检查
双击运行目录下的 **`1.py`**（图形启动器）。
* **Built-in environment**: 状态应显示为 `✅ ready`。
* **GPU Status**: 若显示显卡型号，则表示显存加速已开启。

### 2. 初始化模型 (首次运行)
点击界面上的 **"Initialize Environment"**。
* 脚本将自动检测并补全 `models/` 文件夹下的权重文件。
* **注意**: 权重总大小约 8GB，国内网络将自动切换至 Modelscope 镜像。

### 3. 启动应用
点击 **"Start Application"**，待日志显示 `Running on local URL: http://127.0.0.1:7860` 后，程序会自动弹出浏览器界面。

---

## 🛠️ 显存优化 (Low VRAM 模式)

如果您的显卡显存 **≤ 8GB**，请务必执行以下操作：

1.  **代码优化**: 本版本已在 `app.py` 内部集成了 `enable_model_cpu_offload()` 逻辑，显著降低显存占用。
2.  **输入控制**: 
    - 建议上传图片分辨率不超过 `512x512`。
    - 驱动视频长度建议控制在 10 秒以内。
3.  **系统设置**: 
    - 请确保 Windows **虚拟内存** 已手动设置为 **32GB** 或更大，以防止加载扩散模型时因内存溢出导致程序闪退。

---

## ❌ 故障排除 (修复指南)

### 1. 程序闪退 (错误码: 0xC0000005)
* **现象**: 控制台报错 `Access Violation` 后退出。
* **原因**: 通常是虚拟内存不足或底层 DLL (OpenCV/Torch) 冲突。
* **解决**: 
    * 关闭所有高内存占用程序（如浏览器、壁纸引擎）。
    * 运行 `.\env\python.exe -m pip install "numpy<2.0.0"` 解决 NumPy 2.x 版本冲突。

### 2. Gradio 界面报错 (AttributeError: Blocks)
* **原因**: Gradio 库损坏或前端模板 (`index.html`) 丢失。
* **解决**: 运行以下命令强制修复：
    ```cmd
    .\env\python.exe -m pip install --force-reinstall gradio
    ```

### 3. 模型下载失败 (HuggingFace Login 错误)
* **原因**: 默认官方下载源不可用。
* **解决**: 确保已安装 `modelscope`。执行：
    ```cmd
    .\env\python.exe -m pip install modelscope
    ```

---

## 📂 核心目录结构

- `env/`: 独立的 Python 运行环境。
- `models/`: 存放 RealisticVision、VAE、ControlNet 等核心权重。
- `app.py`: Gradio 主程序。
- `1.py`: 带有日志监控功能的 GUI 启动器。
- `Download and loading.py`: 资源预检与模型加载脚本。

---

## ⚠️ 使用须知
1. 请勿随意运行非官方提供的“清理脚本”，特别是删除 `.html` 或 `.json` 文件的脚本，这会导致前端崩溃。
2. 本项目仅供学习研究使用，严禁用于任何形式的深度伪造（Deepfake）非法用途。
