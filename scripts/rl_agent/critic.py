import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

HIDDEN1_UNITS = 50
HIDDEN2_UNITS = 30

class Critic(object):
    def __init__(self, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        

        #Now create the model
        self.model = CriticNetwork()
        self.target_model = CriticNetwork()
        self.initialize_target_network(self.target_model, self.model)
        self.model.cuda()
        self.target_model.cuda()
        self.optim = Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def initialize_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):    
            target_param.data.copy_(target_param.data.copy_(param.data))
    
    def train(self, q_values, y_t):
        loss = self.criterion(q_values, y_t)
        loss.backward()
        self.optim.step()
        return loss.item()

    def target_train(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1-self.TAU) + param.data * self.TAU
            )

class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3 ,32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.ELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32 ,64, 5, padding=2, bias=False),
            nn.BatchNorm2d(64, eps=1e-04, affine=False),
            nn.ELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64 ,128, 5, padding=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-04, affine=False),
            nn.ELU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128 ,256, 5, padding=2, bias=False),
            nn.BatchNorm2d(256, eps=1e-04, affine=False),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256*14*14, 1024, bias=False),
            nn.ELU(),
            nn.Linear(1024, 128, bias=False),
            nn.ELU(),
            nn.Linear(128, 1, bias=False),
            nn.ELU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2, 30, bias=False),
            nn.ELU()
        )


        self.fc3 = nn.Linear(1, 30)
        self.fc4 = nn.Linear(60, 30)
        self.fc5 = nn.Linear(30, 1)
    
    def forward(self, xs):
        [x0, x1], a = xs
        x0 = self.conv(x0)
        x0 = x0.view(x0.size(0), -1)
        x0 = self.fc(x0)

        x = torch.cat((x0, x1), dim=1)
        x = self.fc2(x)

        
        a = self.fc3(a)
        a = F.elu(x)
        x = torch.cat((x, a), dim=1)
        x = self.fc4(x)
        x = F.elu(x)
        x = self.fc5(x)

        return x