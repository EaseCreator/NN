import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import time
import copy
import os
from tqdm import tqdm

# 1. 设置超参数与设备
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = 10 # CIFAR-10 有 10 个类别
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. 数据准备 (使用 CIFAR-10, Resize到224x224)
def get_dataloaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])

    dataloaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    return dataloaders, len(train_dataset), len(val_dataset)

# 3. 模型构建函数
def initialize_model(model_name, use_pretrained, num_classes):
    model = None
    if model_name == "resnext50":
        # 如果从头训练，weights=None；如果是微调，加载在ImageNet上的预训练权重
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.resnext50_32x4d(weights=weights)
        
        # 修改最后一层分类器
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.densenet121(weights=weights)
        
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    return model

# 4. 训练引擎
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# 5. 主执行逻辑
def run_experiment(model_name, mode):
    print(f"\n{'='*40}\nStarting Experiment: {model_name} | Mode: {mode}\n{'='*40}")
    use_pretrained = (mode == "finetune")
    
    dataloaders, train_size, val_size = get_dataloaders()
    dataset_sizes = {'train': train_size, 'val': val_size}
    
    model = initialize_model(model_name, use_pretrained, NUM_CLASSES)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # 针对微调，我们通常使用较小的学习率
    lr = 0.001 if use_pretrained else 0.01
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    
    model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=NUM_EPOCHS)
    
    # 保存权重 (不传到Git上，自己留着)
    torch.save(model.state_dict(), f"{model_name}_{mode}.pth")
    return history

import matplotlib.pyplot as plt

def plot_experiments(histories):
    """
    根据记录的 history 画出 Loss 和 Accuracy 的对比折线图
    histories: 字典格式，例如 {'ResNeXt_Scratch': history1, ...}
    """
    epochs = range(1, NUM_EPOCHS + 1)
    
    plt.figure(figsize=(16, 6))
    
    # ------------------
    # 1. 绘制验证集准确率 (Validation Accuracy)
    # ------------------
    plt.subplot(1, 2, 1)
    for exp_name, history in histories.items():
        # 这里为了图表清晰，我们主要对比 Validation Accuracy
        plt.plot(epochs, history['val_acc'], marker='o', label=exp_name)
        
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # ------------------
    # 2. 绘制验证集损失 (Validation Loss)
    # ------------------
    plt.subplot(1, 2, 2)
    for exp_name, history in histories.items():
        plt.plot(epochs, history['val_loss'], marker='x', label=exp_name)
        
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=300)
    print("Plot saved as 'experiment_comparison.png'")
    
    plt.show()

if __name__ == '__main__':
    print("GPU是否可用:", torch.cuda.is_available())

    # 实验1: ResNeXt 从头训练
    history_resnext_scratch = run_experiment("resnext50", "scratch")
    
    # 实验2: ResNeXt 微调
    history_resnext_finetune = run_experiment("resnext50", "finetune")
    
    # 实验3: DenseNet 从头训练
    history_densenet_scratch = run_experiment("densenet121", "scratch")
    
    # 实验4: DenseNet 微调
    history_densenet_finetune = run_experiment("densenet121", "finetune")
    
    print("\nAll experiments finished. Generating comparison plots...")
    
    # 调用画图函数
    plot_experiments(all_histories)
    
    print("\nDone! Please upload your code to Git and include the generated plot in your PDF.")