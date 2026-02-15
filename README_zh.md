# Emoji Transfer

本仓库提供了一个高度集成的表情迁移（人脸驱动）环境。通过一张静态照片（参考图）和一段原始驱动视频/图像，利用 Stable Diffusion 和深度学习技术，将视频中的表情细节实时同步到静态照片的人物脸上。

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
运行目录下的 **`launch.py`**（图形启动器）。
* **Built-in environment**: 状态应显示为 `✅ ready`。
* **GPU Status**: 若显示显卡型号，则表示显存加速已开启。
* **Network status**: 确保全程联网！！！。

### 2. 初始化模型 (首次运行)
点击界面上的 **"Initialize Environment"**。
* 脚本将自动检测并补全 `models/` 文件夹下的权重文件。
* **注意**: 权重总大小约 8GB，国内网络将自动切换至 Modelscope 镜像。

### 3. 启动应用
点击 **"Start Application"**，待日志显示 `Running on local URL: http://127.0.0.1:7860` 后，程序会自动弹出浏览器界面。

---

## 🛠️ 显存优化

显卡显存建议 **>= 8GB**：
1. **输入控制**: 建议上传图片分辨率不超过 `512x512`。
2. **系统设置**: 请确保 Windows **虚拟内存** 已手动设置为 **32GB** 或更大，以防止加载扩散模型时因内存溢出导致程序闪退。
3. **进阶调优**: 可根据显存大小修改 `app.py` 里面的 `pipeline_dict_len` 值，该值表示预加载的模型数量。

---

## ❌ 故障排除 (修复指南)

### 1. 程序闪退 (错误码: 0xC0000005)
* **现象**: 控制台报错 `Access Violation` 后退出。
* **原因**: 通常是虚拟内存不足或底层 DLL (OpenCV/Torch) 冲突。
* **解决**: 关闭所有高内存占用程序（如浏览器、壁纸引擎），并确保虚拟内存充足。

### 2. 界面显示 `Error: @@ huggingface-cli login`？
* **原因分析**: 本地模型文件不完整。程序尝试联网下载时中断，或由于网络环境、虚拟内存不足导致认证报错。
* **解决方法**:
    * **激活魔搭 (Modelscope) 镜像**: 在项目目录下运行：
      ```bash
      .\env\python.exe -m pip install modelscope
      ```
    * **手动补全**: 检查 `model_cache/huggingface` 文件夹，确保大容量 `.safetensors` 或 `.bin` 文件已完整存在。

---

## 📂 核心目录结构

- `env/`: 独立的 Python 运行环境（必须放在根目录）。
- `model_cache/huggingface`: 存放 RealisticVision、VAE、ControlNet 等核心权重。
- `app.py`: Gradio 主程序。
- `launch.py`: 带有日志监控功能的 GUI 启动器。
- `Download and loading.py`: 资源预检与模型加载脚本。

---

## 📢 发布与特别说明

### **1. 极简发布版说明**
* **本仓库不含 `env` 和模型文件**：为了减小体积，你需要通过启动器进行初始化。
* **必须联网**：首次初始化时需要联网下载约 12GB 的必要组件，请保持网络畅通。

### **2. 免责声明 (Disclaimer)**
* **合法用途**: 本项目仅供学术交流，严禁利用本项目制作、传播违背他人意愿的深度伪造（Deepfake）视频。
* **版权责任**: 用户生成的内容由其自行承担法律责任。

---

## 💖 鸣谢 (Acknowledgements)

本项目核心算法基于 **HelloMeme** 开源项目构建，并由多位开发者共同完善。

* **核心算法**: 衷心感谢 **HelloMeme** 原作者及其团队。
  - [GitHub - HelloMeme](https://github.com/juechen-info/HelloMeme)
* **合作伙伴**: 感谢所有在开发、测试及优化过程中提供帮助的合作者。
* **特别致谢**: 感谢HSQ的朋友 **Lin Xiaoru** 在项目开发过程中的支持与陪伴。

正是有了这些优秀的开源工作和朋友的支持，本项目才得以顺利完成。
