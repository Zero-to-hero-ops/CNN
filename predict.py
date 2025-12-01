import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from CNN_simple import SimpleCNN, compute_mean_std

# 计算并定义transform（与CNN_simple.py中保持一致）
mean, std = compute_mean_std()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# Configure matplotlib for English display (default is sufficient)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Make sure minus signs render correctly

# =========================
# 1. Configuration
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 2. Load Model
# =========================
def load_model(model_path="mnist_cnn_model.pth"):
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()
    print(f"Model loaded successfully: {model_path}")
    return model


# =========================
# 3. Prediction Function
# =========================
@torch.no_grad()
def predict(model, image_tensor):
    """Predict a single image, return predicted class and confidence"""
    # Handle input shape
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)
    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_label = outputs.argmax(1).item()
    confidence = probs[0][pred_label].item()

    return pred_label, confidence


# =========================
# 4. Visualize Predictions
# =========================
def show_predictions(model, num_samples=10):
    """Display predictions on the test set"""
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        image, true_label = test_dataset[indices[idx]]
        pred_label, confidence = predict(model, image)

        img = image.squeeze(0).cpu().numpy()
        ax.imshow(img, cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.2%}',
                     color=color, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# =========================
# 5. Evaluate Accuracy
# =========================
@torch.no_grad()
def evaluate(model, batch_size=64):
    """Evaluate model accuracy on the test set"""
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = total = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += targets.size(0)

    accuracy = 100. * correct / total
    print(f"Test accuracy: {accuracy:.2f}%")
    return accuracy


# =========================
# 6. Main Function
# =========================
if __name__ == "__main__":
    # Load model
    model = load_model("mnist_cnn_model.pth")

    # Evaluate accuracy
    evaluate(model)

    # Show predictions
    show_predictions(model, num_samples=10)
