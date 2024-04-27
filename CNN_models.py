import torch
import torch.nn as nn
import torch.nn.functional as F

# lenet
class CNN1(nn.Module):
    def __init__(self, num_classes):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN2(nn.Module):
    def __init__(self, num_classes):
        super(CNN2, self).__init__()
        
        # cnn block 1 
        self.conv1 = nn.Conv2d(1, 8, padding='same', kernel_size=3)  # output 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)  # output 14x14x8
        self.dropout1 = nn.Dropout(0.1)
        
        # cnn block 2
        self.conv2 = nn.Conv2d(8, 16, padding='same', kernel_size=3)  # output 14x14x16
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)  # output 7x7x16
        self.dropout2 = nn.Dropout(0.1)
        
        # cnn block 3
        self.conv3 = nn.Conv2d(16, 32, padding='same', kernel_size=3)  # output 7x7x32
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(0.1)
        
        # fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7*7*32, num_classes)  
        self.relu = nn.ReLU()
    
    # forward pass 
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x


class CNN3(nn.Module):
    def __init__(self, num_classes):
        super(CNN3, self).__init__()
        
        # cnn block 1 
        self.conv1 = nn.Conv2d(1, 32, padding='same', kernel_size=3)  # output 28x28x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # output 14x14x32
        self.dropout1 = nn.Dropout(0.1)
        
        # cnn block 2
        self.conv2 = nn.Conv2d(32, 64, padding='same', kernel_size=3)  # output 14x14x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # Output: 7x7x64
        self.dropout2 = nn.Dropout(0.1)
        
        # cnn block 3
        self.conv3 = nn.Conv2d(64, 128, padding='same', kernel_size=3)  # output 7x7x128
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.1)
        
        # fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7*7*128, num_classes)  
        self.relu = nn.ReLU()
    
    # forward pass 
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        return x

class CNN4(nn.Module):
    def __init__(self, num_classes):
        super(CNN4, self).__init__()
        
        # cnn block 1
        self.conv1 = nn.Conv2d(1, 32, padding='same', kernel_size=3)  # output 28x28x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # output 14x14x32
        self.dropout1 = nn.Dropout(0.1)
        
        # cnn block 2
        self.conv2 = nn.Conv2d(32, 64, padding='same', kernel_size=3)  # output 14x14x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # output 7x7x64
        self.dropout2 = nn.Dropout(0.1)
        
        # cnn block 3
        self.conv3 = nn.Conv2d(64, 128, padding='same', kernel_size=3)  # output 7x7x128
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.1)
        
        # fully connected layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    # forward pass
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN5(nn.Module):
    def __init__(self, num_classes):
        super(CNN5, self).__init__()
        
        # cnn block 1
        self.conv1 = nn.Conv2d(1, 32, padding='same', kernel_size=3)  # output 28x28x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # output 14x14x32
        self.dropout1 = nn.Dropout(0.1)
        
        # cnn block 2
        self.conv2 = nn.Conv2d(32, 64, padding='same', kernel_size=3)  # output 14x14x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # output 7x7x64
        self.dropout2 = nn.Dropout(0.1)
        
        # cnn block 3
        self.conv3 = nn.Conv2d(64, 128, padding='same', kernel_size=3)  # output 7x7x128
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(0.1)
        
        # fully connected layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    # forward pass
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN6(nn.Module):
    def __init__(self, num_classes):
        super(CNN6, self).__init__()
        # cnn block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)

        # cnn block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)

        # cnn block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)

        # calculate output size after pooling layers
        input_size = 28
        output_size = input_size // 2  # after first pooling layer
        output_size //= 2  # after second pooling layer
        output_size //= 2  # after third pooling layer

        # fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * output_size * output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout4 = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout4(x)
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class bestCNN(nn.Module):
    def __init__(self, num_classes):
        super(bestCNN, self).__init__()
        # cnn block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)

        # cnn block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)

        # cnn block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)

        # calculate output size after pooling layers
        input_size = 28
        output_size = input_size // 2  # after first pooling layer
        output_size //= 2  # after second pooling layer
        output_size //= 2  # after third pooling layer

        # fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * output_size * output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout4 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(self.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)
        x = self.dropout2(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout2(self.relu(self.bn4(self.conv4(x))))
        x = self.pool2(x)
        x = self.dropout3(self.relu(self.bn5(self.conv5(x))))
        x = self.dropout3(self.relu(self.bn6(self.conv6(x))))
        x = self.pool3(x)
        x = self.dropout4(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


class Residual1(nn.Module):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class CNN8(nn.Module):
    def __init__(self, num_classes):
        super(CNN8, self).__init__()
        
        # cnn block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)

        # cnn block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)

        # residual block 1
        self.res1 = Residual1(64)

        # cnn block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)

        # residual block 2
        self.res2 = Residual1(128)

        # calculate output size after pooling layers
        input_size = 28
        output_size = input_size // 2  # after first pooling layer
        output_size //= 2  # after second pooling layer
        output_size //= 2  # after third pooling layer

        # fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * output_size * output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout4 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(self.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)

        x = self.dropout2(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout2(self.relu(self.bn4(self.conv4(x))))
        x = self.pool2(x)

        x = self.res1(x)

        x = self.dropout3(self.relu(self.bn5(self.conv5(x))))
        x = self.dropout3(self.relu(self.bn6(self.conv6(x))))
        x = self.pool3(x)

        x = self.res2(x)

        x = self.dropout4(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

class Residual(nn.Module):  #@save
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class CNN9(nn.Module):
    def __init__(self, num_classes):
        super(CNN9, self).__init__()
        
        # cnn block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.1)

        # cnn block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.1)

        # residual block 1
        self.res1 = Residual(64)

        # cnn block 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding='same', bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.1)

        # residual block 2
        self.res2 = Residual(128)

        # cnn block 4
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding='same', bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding='same', bias=False)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.1)

        # residual block 3
        self.res3 = Residual(256)

        # calculate output size after pooling layers
        input_size = 28
        output_size = input_size // 2  # after first pooling layer
        output_size //= 2  # after second pooling layer
        output_size //= 2  # after third pooling layer
        output_size //= 2  # after fourth pooling layer

        # fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * output_size * output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout5 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(self.relu(self.bn2(self.conv2(x))))
        x = self.pool1(x)

        x = self.dropout2(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout2(self.relu(self.bn4(self.conv4(x))))
        x = self.pool2(x)

        x = self.res1(x)

        x = self.dropout3(self.relu(self.bn5(self.conv5(x))))
        x = self.dropout3(self.relu(self.bn6(self.conv6(x))))
        x = self.pool3(x)

        x = self.res2(x)

        x = self.dropout4(self.relu(self.bn7(self.conv7(x))))
        x = self.dropout4(self.relu(self.bn8(self.conv8(x))))
        x = self.pool4(x)

        x = self.res3(x)

        x = self.dropout5(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x