# 🚀 Emoji Transfer (HelloMeme-Packed)

这是一个经过优化的表情迁移（人脸驱动）工具。只需一张照片和一段驱动视频，即可生成极具表现力的动画。



---

## 🛠️ 快速开始

1. **运行启动器**: 双击 `1.py` 启动管理控制台。
2. **环境自检**: 查看界面上方的环境状态：
   - ✅ **Built-in environment ready**: 代表内建环境正常。
   - ✅ **GPU**: 显示显卡型号则代表已开启显卡加速。
3. **初始化**: 首次运行或模型缺失时，点击 **"Initialize Environment"**。
4. **启动程序**: 点击 **"Start Application"**，待日志显示 `http://127.0.0.1:7860` 后即可在浏览器使用。

---

## ⚙️ 显存优化 (Low VRAM 模式)

如果你的显卡显存小于 **8GB**，请尝试以下优化：

* **分辨率限制**: 建议将输入图片和视频裁剪为 `512x512`。
* **手动开启 Offload**: 在 `app.py` 中确认已启用 `pipe.enable_model_cpu_offload()`。
* **虚拟内存**: 强烈建议手动设置 Windows 虚拟内存为 **32GB** 以上，以防大模型加载时崩溃。

---

## ❌ 常见报错修复

| 报错信息 | 解决方法 |
| :--- | :--- |
| **0xC0000005** | 虚拟内存不足或 DLL 冲突。请尝试重启电脑并确保安装了 VC++ 运行库。 |
| **AttributeError: ... Blocks** | Gradio 库损坏。运行 `.\env\python.exe -m pip install --force-reinstall gradio`。 |
| **ImportError: cannot load module...** | NumPy 版本冲突。运行 `.\env\python.exe -m pip install "numpy<2.0.0"`。 |
| **TemplateNotFound** | Gradio 静态文件被误删。请按照上述方法重装 Gradio。 |

---

## 📂 核心依赖组件

- **PyTorch**: 2.8.0+cu128
- **Gradio**: 5.x (Web UI 支撑)
- **OpenCV-Headless**: 图像处理（不带 GUI 以提高稳定性）
- **Modelscope**: 负责国内模型镜像下载
