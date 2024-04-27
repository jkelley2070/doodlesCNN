import torch.nn as nn

class MLP1(nn.Module):
    def __init__(self, num_classes):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP2(nn.Module):
    def __init__(self, num_classes):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP3(nn.Module):
    def __init__(self, num_classes):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, num_classes)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class MLP4(nn.Module):
    def __init__(self, num_classes):
        super(MLP4, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class MLP5(nn.Module):
    def __init__(self, num_classes):
        super(MLP5, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x

class bestMLP(nn.Module):
    def __init__(self, num_classes):
        super(bestMLP, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 384)
        self.gelu = nn.GELU()
        self.fc4 = nn.Linear(384, 256)
        self.fc5 = nn.Linear(256, num_classes)
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(384)
        self.bn4 = nn.BatchNorm1d(256)
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dropout1(self.gelu(self.bn1(self.fc1(x))))
        x = self.dropout1(self.gelu(self.bn2(self.fc2(x))))
        x = self.dropout1(self.gelu(self.bn3(self.fc3(x))))
        x = self.dropout1(self.gelu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
        return x
