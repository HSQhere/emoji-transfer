import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义模型结构 (VAE)
# ==========================================
class EmojiVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(EmojiVAE, self).__init__()
        # 编码器：压缩图片
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(64 * 16 * 16, latent_dim)
        
        # 解码器：还原图片
        self.decoder_input = nn.Linear(latent_dim, 64 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.decoder_input(z)), mu, logvar

# ==========================================
# 2. 总运行函数
# ==========================================
def run_all():
    # --- 环境配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 100  # 建议至少100次，笑脸才会清晰
    batch_size = 16
    latent_dim = 128
    
    # --- 检查数据 ---
    if not os.path.exists('./data'):
        print("❌ 错误：找不到 data 文件夹，请先创建它并放进笑脸图片！")
        return

    # --- 数据加载 ---
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root='./data', transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"📦 已加载图片数量: {len(dataset)}")

    # --- 初始化模型 ---
    model = EmojiVAE(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- 阶段一：训练 ---
    print(f"🚀 开始在 {device} 上训练...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (data, _) in enumerate(loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(data)
            
            # 计算损失 (MSE重建 + KL散度)
            mse_loss = nn.functional.mse_loss(recon, data, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse_loss + kld_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(dataset):.2f}")

    # 保存模型权重以备后用
    torch.save(model.state_dict(), "smile_brain.pth")
    print("💾 训练完成，‘大脑’已保存为 smile_brain.pth")

    # --- 阶段二：即时生成 ---
    print("🎨 正在根据训练成果变出笑脸...")
    model.eval()
    with torch.no_grad():
        # 随机采样 8 个笑脸
        z = torch.randn(8, latent_dim).to(device)
        generated = model.decoder(model.decoder_input(z))
        
        # 保存到本地图片文件
        save_image(generated, "result_smiles.png", nrow=4)
        print("✅ 结果已保存为：result_smiles.png")
        
        # 弹窗展示
        plt.figure(figsize=(12, 6))
        for i in range(8):
            plt.subplot(2, 4, i+1)
            plt.imshow(generated[i].cpu().permute(1, 2, 0))
            plt.axis('off')
        plt.suptitle("Generated Smiles")
        plt.show()

if __name__ == "__main__":
    run_all()