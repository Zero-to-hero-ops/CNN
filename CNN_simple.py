import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# =========================
# 1. 一些超参数
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128          # 你想要的起始 batch_size
LR = 0.001
EPOCHS = 50


# =========================
# 2. 先用训练集计算真实 mean / std
# =========================
def compute_mean_std():
    raw_transform = transforms.ToTensor()
    raw_train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=raw_transform
    )
    raw_loader = DataLoader(raw_train_dataset, batch_size=512, shuffle=False)

    mean = 0.0
    std = 0.0
    total = 0

    for imgs, _ in raw_loader:
        # imgs: [B, 1, 28, 28] -> [B, 784]
        imgs = imgs.view(imgs.size(0), -1)
        batch_mean = imgs.mean(dim=1)   # 每张图自己的 mean，[B]
        batch_std = imgs.std(dim=1)     # 每张图自己的 std，[B]

        mean += batch_mean.sum().item()
        std += batch_std.sum().item()
        total += imgs.size(0)

    mean /= total
    std /= total
    return mean, std


# =========================
# 3. 定义一个简单的 CNN
#    输入: 1x28x28
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积层: 1 通道 -> 16 通道, 卷积核 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 最大池化
        # 第二个卷积层: 16 通道 -> 32 通道
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # 28x28 -> conv1 -> 16x28x28 -> pool -> 16x14x14
        # -> conv2 -> 32x14x14 -> pool -> 32x7x7
        # 展平后是 32 * 7 * 7 维
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)   # 输出10类（数字0~9）
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # 展平成 [batch, 32*7*7]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)            # 输出 logits
        return x


# =========================
# 3.5 检查 / 调整 batch_size（简单版）
# =========================
def adjust_batch_size_for_device(model, init_batch_size, device):
    """
    更高效估算最大的batch_size的方法——不直接尝试多轮前向/反向计算，可以基于 GPU 总显存以及单样本显存占用做初步估算。
    这种方式无需真的多次跑前向/反向传播，大部分情况下精度已足够，极端场景下自行微调。
    """
    if device != "cuda":
        print(f"使用 CPU，batch_size = {init_batch_size}")
        return init_batch_size

    # 检查 GPU 总显存
    total_mem = torch.cuda.get_device_properties(0).total_memory
    # 预留一部分给系统和cuda runtime
    reserved_mem = int(total_mem * 0.10)  # 预留10%
    usable_mem = total_mem - reserved_mem

    # 先用一个小batch实际测算单个sample的显存消耗（forward+backward, 真实最大化估算）
    bs_test = 4
    torch.cuda.empty_cache()
    criterion = nn.CrossEntropyLoss()
    dummy_x = torch.randn(bs_test, 1, 28, 28, device=device)
    dummy_y = torch.randint(0, 10, (bs_test,), device=device)
    model.zero_grad()
    out = model(dummy_x)
    loss = criterion(out, dummy_y)
    loss.backward()
    torch.cuda.synchronize()
    # 获取分配的显存
    sample_mem = torch.cuda.max_memory_allocated(device) // bs_test
    del dummy_x, dummy_y, out, loss
    torch.cuda.empty_cache()

    if sample_mem == 0:
        sample_mem = 2 * 1024 * 1024  # 万一没采到，默认估2MB每样本

    # 估算最大可以batch多少
    est_max_bs = usable_mem // sample_mem
    # 限制区间最大不超过 8192，且不能小于1
    est_max_bs = min(est_max_bs, 8192)
    est_max_bs = max(est_max_bs, 1)
    print(f"显存基估算最大 batch_size ≈ {est_max_bs}")

    # 可以进一步用二分法快速实测确定
    left = 1
    right = est_max_bs
    best = 1
    while left <= right:
        mid = (left + right) // 2
        try:
            torch.cuda.empty_cache()
            dummy_x = torch.randn(mid, 1, 28, 28, device=device)
            dummy_y = torch.randint(0, 10, (mid,), device=device)
            model.zero_grad()
            out = model(dummy_x)
            loss = criterion(out, dummy_y)
            loss.backward()
            best = mid
            left = mid + 1
            del dummy_x, dummy_y, out, loss
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                right = mid - 1
                torch.cuda.empty_cache()
            else:
                raise e

    print(f"最终确定 batch_size = {best}（能力极限）")
    return best

# =========================
# 4. 训练一个 epoch
# =========================
def train_one_epoch(model, optimizer, criterion, train_loader, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch [{epoch}] Step [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {running_loss/(batch_idx+1):.4f} "
                f"Acc: {100.*correct/total:.2f}%"
            )

    print(
        f"==> Epoch {epoch} Train "
        f"Loss: {running_loss/len(train_loader):.4f} "
        f"Acc: {100.*correct/total:.2f}%"
    )


# =========================
# 5. 在测试集上评估
# =========================
@torch.no_grad()
def evaluate(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(
        f"==> Test Loss: {test_loss/len(test_loader):.4f} "
        f"Acc: {100.*correct/total:.2f}%"
    )


# =========================
# 6. 随机看看几张预测结果，图像+标签
# =========================
@torch.no_grad()
def show_some_predictions(model, test_loader, num_samples=5):
    model.eval()
    inputs, targets = next(iter(test_loader))
    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    outputs = model(inputs)
    _, predicted = outputs.max(1)

    for i in range(num_samples):
        img = inputs[i].cpu().squeeze(0)  # [1,28,28] -> [28,28]
        label = targets[i].item()
        pred = predicted[i].item()

        plt.imshow(img, cmap="gray")
        plt.title(f"True: {label} | Pred: {pred}")
        plt.axis("off")
        plt.show()


# =========================
# 7. 主程序入口
# =========================
def main():
    print(f"Using device: {DEVICE}")

    # 1. 计算 MNIST 真实 mean / std
    mean, std = compute_mean_std()
    print(f"Calculated MNIST mean: {mean:.4f}, std: {std:.4f}")

    # 2. 定义最终使用的 transform（专业版）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    # 3. 构建数据集
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # 4. 创建模型
    model = SimpleCNN().to(DEVICE)

    # 5. 检查 / 调整 batch_size（只跑一次）
    #effective_bs = adjust_batch_size_for_device(model, BATCH_SIZE, DEVICE)
    #print(f"最终使用的 batch_size = {effective_bs}")
    effective_bs = BATCH_SIZE
    print(f"最终使用的 batch_size = {effective_bs}")

    # 6. 用最终确定的 batch_size 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_bs,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_bs,
        shuffle=False
    )

    # 7. 创建损失函数、优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 8. 训练 + 测试
    print(f"\n开始训练，共 {EPOCHS} 个 epoch...\n")
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, optimizer, criterion, train_loader, epoch)
        evaluate(model, criterion, test_loader)

    print("Training finished! Let's see some predictions...")
    show_some_predictions(model, test_loader, num_samples=5)

    # 9. 保存模型
    model_path = "mnist_cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
