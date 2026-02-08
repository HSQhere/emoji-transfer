import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
import os

# --- 1. 成对数据加载器 ---
class EmotionDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        # 读取笑脸(A)和哭脸(B)的文件路径
        self.smile_images = sorted([os.path.join(root_A, f) for f in os.listdir(root_A) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.cry_images = sorted([os.path.join(root_B, f) for f in os.listdir(root_B) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
        
        if len(self.smile_images) != len(self.cry_images):
            print(f"⚠️ 警告：笑脸({len(self.smile_images)})与哭脸({len(self.cry_images)})数量不一致！请检查文件名是否对应。")

    def __len__(self):
        return min(len(self.smile_images), len(self.cry_images))

    def __getitem__(self, idx):
        smile = Image.open(self.smile_images[idx]).convert('RGB')
        cry = Image.open(self.cry_images[idx]).convert('RGB')
        if self.transform:
            smile = self.transform(smile)
            cry = self.transform(cry)
        return smile, cry

# --- 2. 核心运行流程 ---
def run_emotion_bridge():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 64
    epochs = 500  # 数据量小时，增加训练轮数有助于模型“强行记住”表情特征
    
    # 【关键结构】in_channels=6 
    # (3通道用于接收带噪声的图 + 3通道用于接收原始笑脸作为条件引导)
    model = UNet2DModel(
        sample_size=img_size,
        in_channels=6, 
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # 数据预处理
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 路径检查
    path_a = './data/smile_to_cry/A'
    path_b = './data/smile_to_cry/B'
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        print(f"❌ 错误：未找到数据文件夹！请确保路径存在：\n{path_a}\n{path_b}")
        return

    dataset = EmotionDataset(path_a, path_b, transform=tf)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # --- 第一部分：训练 ---
    print(f"🚀 正在设备 {device} 上启动‘笑转哭’强化训练...")
    model.train()
    for epoch in range(epochs):
        loop = tqdm(loader, leave=False)
        for smiles, cries in loop:
            smiles, cries = smiles.to(device), cries.to(device)
            
            # 对目标（哭脸）添加噪声
            noise = torch.randn_like(cries)
            timesteps = torch.randint(0, 1000, (cries.shape[0],), device=device).long()
            noisy_cries = noise_scheduler.add_noise(cries, noise, timesteps)
            
            # 【核心拼接】将噪声目标和笑脸条件拼成 6 通道
            input_combined = torch.cat([noisy_cries, smiles], dim=1)
            
            # 预测噪声
            prediction = model(input_combined, timesteps).sample
            loss = nn.functional.mse_loss(prediction, noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

    # 保存模型
    torch.save(model.state_dict(), "smile_to_cry_bridge.pth")
    print("✅ 权重已保存为 smile_to_cry_bridge.pth")

    # --- 第二部分：即时生成测试 ---
    print("🎨 正在验证转换效果...")
    model.eval()
    # 使用 DDIM 采样器加速生成
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000)
    ddim_scheduler.set_timesteps(50)

    with torch.no_grad():
        # 从数据集中取出一组笑脸进行测试
        test_smiles, _ = next(iter(loader))
        test_smiles = test_smiles.to(device)
        
        # 从纯噪声开始“洗图”
        image = torch.randn_like(test_smiles)
        
        for t in tqdm(ddim_scheduler.timesteps, desc="表情转换中"):
            # 每一步都要参考原始笑脸
            combined_input = torch.cat([image, test_smiles], dim=1)
            model_output = model(combined_input, t).sample
            image = ddim_scheduler.step(model_output, t, image).prev_sample

        # 结果处理并保存：上面是输入的笑脸，下面是生成的哭脸
        result = torch.cat([test_smiles, image], dim=0)
        result = (result / 2 + 0.5).clamp(0, 1)
        save_image(result, "final_conversion_test.png", nrow=4)
        print("🎉 转换测试完成！请查看 conversion_result.png 查看对比效果。")

if __name__ == "__main__":
    run_emotion_bridge()
