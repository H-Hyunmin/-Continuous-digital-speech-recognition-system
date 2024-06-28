import torch
from torch import nn
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
class Audio(nn.Module):
    def __init__(self):
        super(Audio, self).__init__()
        self.conv1 = nn.Conv2d(13, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)  # 增加Batch Normalization
        self.conv2 = nn.Conv2d(32, 32, (3, 2), (1, 2), (1, 0))
        self.bn2 = nn.BatchNorm2d(32) 
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)  
        self.conv4 = nn.Conv2d(64, 64, (3, 2), (1, 2), (1, 0))
        self.bn4 = nn.BatchNorm2d(64)  
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(128) 
        self.conv6 = nn.Conv2d(128, 128, (3, 2), 2)
        self.bn6 = nn.BatchNorm2d(128)  
        self.fc1 = nn.Linear(8*128, 256)
        self.dropout = nn.Dropout(0.5)  # 增加
        self.fc2 = nn.Linear(256, 10)

    def forward(self, inputs, labels=None):
        out = F.relu(self.bn1(self.conv1(inputs)))#增加激活函数
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = out.view(out.size(0), -1)  # flatten the tensor
        out = F.relu(self.fc1(out))
        out = self.dropout(out)  # 增加Dropout层
        out = self.fc2(out)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(out, labels)
            acc = (out.argmax(dim=1) == labels).float().mean()
            return loss, acc
        else:
            return out

def train_model(model, train_loader, dev_loader, epochs=20):
    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)  # Use AdamW optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)  # Use ReduceLROnPlateau scheduler
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for i, (features, labels) in enumerate(train_loader):
            features = features.float().to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            # Forward pass
            outputs = model(features)

            # 将所有参数堆叠成一个张量
            params = torch.cat([x.view(-1) for x in model.parameters()])

            # 计算L2范数
            l2_reg = torch.norm(params, 2)

            # 计算损失
            loss = criterion(outputs, labels) + 0.001 * l2_reg  # Add L2 regularization

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if (i+1) % 20 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

        # 在验证集上验证模型
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, labels in dev_loader:
                features = features.float().to('cuda' if torch.cuda.is_available() else 'cpu')
                labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = correct / total
            print(f'Validation Accuracy of the model on the test images: {acc * 100}%')

            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'best_model.ckpt')

            # 更新学习率
            scheduler.step(acc)

    print(f'Best Validation Accuracy: {best_acc * 100}%')


def infer_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            features = features.float().to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f'Test Accuracy of the model on the test images: {acc * 100}%')


