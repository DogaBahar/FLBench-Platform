import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_model():
    """Factory function expected by the Adaptive Layer."""
    return FlexibleCNN(in_channels=3, num_classes=100)

def train(net, trainloader, epochs, lr, optimizer_name, mu=0.0, global_params=None):
    """
    mu/global_params implement FedProx (Li et al., 2018): a (mu/2)*||w - w_global||^2
    penalty pulling local weights back toward the global model as received at
    the start of this round. mu=0 (the default) is plain local training, i.e.
    what every adapter already did before FedProx support was added.
    """
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    device = next(net.parameters()).device
    net.train()
    total_loss = 0.0
    for epoch in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            if mu > 0 and global_params is not None:
                prox_term = sum((p - g.to(p.device)).pow(2).sum() for p, g in zip(net.parameters(), global_params))
                loss = loss + (mu / 2) * prox_term
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / max(1, len(trainloader) * epochs)

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            total += labels.size(0)
    return loss / len(testloader), correct / total