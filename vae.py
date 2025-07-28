import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image # 尽管在此代码中未使用，但常用于保存生成的图像
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
batch_size = 100
original_dim = 784 # 原始输入维度 (28*28)
latent_dim = 2 # 潜在空间维度，为了方便绘图设为2
intermediate_dim = 256 # 中间层维度
epochs = 50
learning_rate = 1e-3 # PyTorch 优化器需要学习率

# --- 数据加载和预处理 ---
transform = transforms.Compose([
    transforms.ToTensor(), # 将 PIL 图像或 NumPy 数组转换为 PyTorch Tensor
    transforms.Lambda(lambda x: x.view(-1)) # 将 28x28 的图像展平为 784 维向量
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# --- VAE 模型定义 ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # 编码器 (Encoder)
        self.fc1 = nn.Linear(original_dim, intermediate_dim)
        self.fc2_mean = nn.Linear(intermediate_dim, latent_dim) # 计算潜在变量的均值
        self.fc2_logvar = nn.Linear(intermediate_dim, latent_dim) # 计算潜在变量的对数方差

        # 解码器 (Decoder)
        self.fc3 = nn.Linear(latent_dim, intermediate_dim)
        self.fc4 = nn.Linear(intermediate_dim, original_dim) # 输出重建的图像

    def encode(self, x):
        # 编码器的前向传播
        h = F.relu(self.fc1(x)) # 使用 ReLU 激活函数
        return self.fc2_mean(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        # 重参数化技巧
        std = torch.exp(0.5 * logvar) # 计算标准差
        epsilon = torch.randn_like(std) # 从标准正态分布中采样 epsilon
        return mu + epsilon * std

    def decode(self, z):
        # 解码器的前向传播
        h = F.relu(self.fc3(z)) # 使用 ReLU 激活函数
        return torch.sigmoid(self.fc4(h)) # 使用 Sigmoid 激活函数将输出限制在 [0, 1] 之间

    def forward(self, x):
        # 整个 VAE 的前向传播
        mu, logvar = self.encode(x.view(-1, original_dim)) # 编码
        z = self.reparameterize(mu, logvar) # 重参数化
        return self.decode(z), mu, logvar # 解码并返回重建结果、均值和对数方差

# --- 损失函数 ---
def vae_loss_function(recon_x, x, mu, logvar):
    # 重建损失 (二元交叉熵)
    # recon_x 是模型重建的图像，x 是原始图像
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, original_dim), reduction='sum')

    # KL 散度 (KL Divergence)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD # 总损失是重建损失和 KL 散度的和

# --- 模型初始化 ---
model = VAE().to(device) # 创建 VAE 实例并将其移动到指定设备
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 使用 Adam 优化器

# --- 训练循环 ---
print("开始训练...")
for epoch in range(1, epochs + 1):
    model.train() # 设置模型为训练模式
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device) # 将数据移动到设备上
        optimizer.zero_grad() # 清除之前的梯度
        recon_batch, mu, logvar = model(data) # 前向传播
        loss = vae_loss_function(recon_batch, data, mu, logvar) # 计算损失
        loss.backward() # 反向传播，计算梯度
        train_loss += loss.item() # 累加损失
        optimizer.step() # 更新模型参数

    print(f'Epoch: {epoch} 平均训练损失: {train_loss / len(train_loader.dataset):.4f}')

    # 验证 (可选，但良好实践)
    model.eval() # 设置模型为评估模式
    test_loss = 0
    with torch.no_grad(): # 在评估阶段禁用梯度计算
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss_function(recon_batch, data, mu, logvar).item()

    print(f'====> Epoch: {epoch} 测试集损失: {test_loss / len(test_loader.dataset):.4f}')

print("训练完成！")

# --- 潜在空间可视化 ---
print("\n正在可视化潜在空间...")
encoder = model # 在 PyTorch 中，可以直接使用模型的 encode 方法
x_test_encoded = []
y_test_labels = []

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        mu, _ = encoder.encode(data.view(-1, original_dim)) # 获取潜在空间的均值
        x_test_encoded.append(mu.cpu().numpy()) # 将结果从 GPU 移到 CPU 并转换为 NumPy 数组
        y_test_labels.append(target.cpu().numpy())

x_test_encoded = np.concatenate(x_test_encoded, axis=0)
y_test_labels = np.concatenate(y_test_labels, axis=0)

plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_labels, cmap='viridis') # 绘制散点图，颜色表示数字类别
plt.colorbar()
plt.title('潜在空间可视化')
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()

# --- 生成数字的可视化 ---
print("\n正在可视化从潜在空间生成的数字...")
generator = model # 在 PyTorch 中，可以直接使用模型的 decode 方法

n = 15  # 显示 15x15 个数字的图像网格
digit_size = 28 # 每个数字的尺寸
figure = np.zeros((digit_size * n, digit_size * n))

# 使用正态分布的分位数来构建潜在变量网格
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

with torch.no_grad():
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device) # 创建潜在样本并移动到设备
            x_decoded = generator.decode(z_sample).cpu().numpy() # 解码生成图像
            digit = x_decoded[0].reshape(digit_size, digit_size) # 重塑为 28x28 图像
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit # 放置到大图中

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r') # 显示生成的图像网格
plt.title('从潜在空间生成的数字')
plt.axis('off')
plt.show()