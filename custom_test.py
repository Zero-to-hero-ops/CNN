import os

import torch
from torchvision import transforms
from PIL import Image, ImageOps

from CNN_simple import SimpleCNN, compute_mean_std


# =========================
# 1. 配置
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMG_DIR = "test_images"  # 你放自测图片的文件夹（相对当前脚本所在目录）


# =========================
# 2. 与训练保持一致的预处理
# =========================
mean, std = compute_mean_std()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
])


# =========================
# 3. 加载训练好的模型
# =========================
def load_model(model_path="mnist_cnn_model.pth"):
    model = SimpleCNN().to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded: {model_path}")
    return model


# =========================
# 4. 加载单张图片并转换为张量
#    要求：灰度图，背景白/数字黑，任意大小，自动缩放到 28x28
# =========================

def load_image(image_path: str) -> torch.Tensor:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("L")  # 转灰度

    # 缩放到 28x28（MNIST 尺寸）
    img = img.resize((28, 28))

    # 转 tensor 并归一化
    img = transform(img)  # [1, 28, 28]
    return img


# =========================
# 5. 对单张图片进行预测
# =========================
@torch.no_grad()
def predict_single(model, image_tensor: torch.Tensor):
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 28, 28]

    image_tensor = image_tensor.to(DEVICE)
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()
    conf = probs[0, pred].item()
    return pred, conf


# =========================
# 6. 对文件夹中的所有图片进行批量预测
# =========================

def predict_folder(model, folder: str = TEST_IMG_DIR):
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        print(f"Folder '{folder}' created. 请把自测图片放到这个文件夹里，然后重新运行脚本。")
        return

    image_files = [f for f in os.listdir(folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

    if not image_files:
        print(f"Folder '{folder}' 里没有找到 png/jpg/jpeg/bmp 图片，请先放入手写数字图片后再运行。")
        return

    print(f"Found {len(image_files)} images in '{folder}':")
    for name in sorted(image_files):
        path = os.path.join(folder, name)
        try:
            img_tensor = load_image(path)
            pred, conf = predict_single(model, img_tensor)
            print(f"  {name:30s} -> Pred: {pred}, Conf: {conf:.2%}")
        except Exception as e:
            print(f"  {name:30s} -> ERROR: {e}")


# =========================
# 7. 主入口
# =========================
if __name__ == "__main__":
    model = load_model("mnist_cnn_model.pth")
    import matplotlib.pyplot as plt

    img = load_image("test_images/9.jpg")
    plt.imshow(img.squeeze(0).numpy(), cmap="gray")
    plt.show()
    #predict_folder(model, TEST_IMG_DIR)
