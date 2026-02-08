import torch
import torch.nn as nn
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# 1. 定义感知老师 (让画面变锐利的关键)
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.slice = nn.Sequential(*[vgg[x] for x in range(16)]).eval()
        for p in self.parameters(): p.requires_grad = False

    def forward(self, pred, target):
        return nn.functional.mse_loss(self.slice(pred), self.slice(target))

def train_and_generate_ultra():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 64
    epochs = 500  # 【强化】增加到 500 次，彻底解决马赛克问题
    
    # 2. 模型配置 (带 Attention 的结构，匹配你之前的权重)
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    perceptual_fn = PerceptualLoss().to(device)

    # 3. 数据加载 (请确保 data/smile 下有你新搜集的图)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder(root='./data', transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 4. 强化训练
    print(f"🚀 正在进行 500 次深度强化训练...")
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, leave=False)
        for clean_images, _ in pbar:
            clean_images = clean_images.to(device)
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, 1000, (clean_images.shape[0],), device=device).long()
            
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps).sample
            
            # 基础损失
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f"Epoch {epoch}/{epochs} Loss: {loss.item():.4f}")

    # 5. 高质量采样
    print("🎨 正在使用强化后的‘大脑’洗出笑脸...")
    model.eval()
    with torch.no_grad():
        image = torch.randn((4, 3, image_size, image_size)).to(device)
        for t in tqdm(noise_scheduler.timesteps, desc="去噪进度"):
            model_output = model(image, t).sample
            image = noise_scheduler.step(model_output, t, image).prev_sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        save_image(image, "ultra_smile_result.png", nrow=2)
        plt.imshow(image[0].cpu().permute(1, 2, 0))
        plt.title("Finally! A Clear Smile")
        plt.show()

if __name__ == "__main__":
    run_all = input("是否启动 500 次强化训练？这可能需要 30 分钟 (y/n): ")
    if run_all.lower() == 'y':
        train_and_generate_ultra()