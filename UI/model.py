import torch
import torch.nn as nn
from torchsummary import summary
from torch.cuda import is_available


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # self.l1 = nn.Linear(input_size, hidden_size)
        # self.l2 = nn.Linear(hidden_size, hidden_size)
        # self.l3 = nn.Linear(hidden_size, hidden_size)
        # self.l4 = nn.Linear(hidden_size, hidden_size)
        # self.l5 = nn.Linear(hidden_size, hidden_size)
        # self.l6 = nn.Linear(hidden_size, num_classes)
        # self.relu = nn.ReLU()
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, int(hidden_size/2))
        self.l3 = nn.Linear(int(hidden_size/2),int(hidden_size/4))
        self.l4 = nn.Linear(int(hidden_size/4), int(hidden_size/8))
        # self.l5 = nn.Linear(hidden_size/8, hidden_size - 40)
        self.l6 = nn.Linear(int(hidden_size/8), num_classes)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.dropout2(out)
        # out = self.l5(out)
        # out = self.relu(out)
        # out = self.dropout2(out)
        out = self.l6(out)
        return out

# print(NeuralNet)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = NeuralNet().to(device)


