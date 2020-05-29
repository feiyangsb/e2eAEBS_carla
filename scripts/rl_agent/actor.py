import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch

class Actor(object):
    def __init__(self, BATCH_SIZE, TAU, LEARNING_RATE):
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        #Now create the model
        self.model = ActorNetwork()
        self.target_model = ActorNetwork()
        self.initialize_target_network(self.target_model, self.model)
        self.model.cuda()
        self.target_model.cuda()
        self.optim = Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def initialize_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):    
            target_param.data.copy_(target_param.data.copy_(param.data))

    def train(self, policy_loss):
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.optim.step()

    def target_train(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                target_param.data * (1-self.TAU) + param.data * self.TAU
            )

class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        
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
            nn.Linear(2, 1, bias=False),
            nn.Sigmoid()
        )

        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            
    
    def forward(self, x):
        x0 = self.conv(x[0])
        x0 = x0.view(x0.size(0), -1)
        x0 = self.fc(x0)

        x = torch.cat((x0, x[1]), dim=1)
        x = self.fc2(x)

        return x
