import os
import numpy as np
from scripts.rl_agent.actor import Actor
from scripts.rl_agent.critic import Critic
from scripts.rl_agent.OU import OU
from scripts.rl_agent.ReplayBuffer import ReplayBuffer
import torch
from torchvision import transforms

BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.001
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000

class ddpgAgent():
    def __init__(self, Testing=False):
        self.testing = Testing
        if not os.path.exists("./models/controller"):
            os.makedirs("./models/controller")

        self.actor = Actor(BATCH_SIZE, TAU, LRA)


        try:
            self.actor.model.load_state_dict(torch.load("./models/controller/actor.pt"))
            self.actor.target_model.load_state_dict(torch.load("./models/controller/actor.pt"))
            print("Load actor model successfully")
        except:
            print("Cannot find actor weights in this directory")

        
        if self.testing is False:
            self.buff = ReplayBuffer(BUFFER_SIZE)
            self.OU = OU()
            self.critic = Critic(BATCH_SIZE, TAU, LRC)
            try:
                self.critic.model.load_state_dict(torch.load("./models/controller/critic.pt"))
                self.critic.target_model.load_state_dict(torch.load("./models/controller/critic.pt"))
                print("Load critic model successfully")
            except:
                print("Cannot find critic weights in this directory")
        else:
            self.actor.model.eval()
    
    def getAction(self, state, epsilon):
        action = np.zeros([1, 1])
        noise = np.zeros([1, 1])
        with torch.no_grad():
            state_0 = torch.tensor(state[0]).to("cuda")
            state_0 = state_0.unsqueeze(0)
            state_1 = torch.tensor([state[1]]).to("cuda")
            state_1 = state_1.unsqueeze(0)
            action_original = self.actor.model([state_0, state_1])
        action_original = self.to_array(action_original)
        if self.testing is False:
            noise[0][0] = (1.0-float(self.testing)) * max(epsilon, 0) * self.OU.function(action_original[0][0], 0.2, 1.00, 0.10)
        action[0][0] = action_original[0][0] + noise[0][0]
        if action[0][0] < 0.0:
            action[0][0] = 0.0
        if action[0][0] > 1.0:
            action[0][0] = 1.0
        print("NN Controller: {:5.4f}, Noise NN Controller: {:5.4f}".format(action_original[0][0], action[0][0]))
        return action
    
    def storeTrajectory(self, s, a, r, s_, done):
        self.buff.add(s, a[0], r, s_, done)
    
    def learn(self):
        batch = self.buff.getBatch(BATCH_SIZE)
        states_image = torch.tensor([e[0][0] for e in batch]).to("cuda")
        #states_image = np.asarray([e[0][0] for e in batch])
        states_velocity = torch.tensor([[e[0][1]] for e in batch]).to("cuda")
        #states_velocity = np.asarray([e[0][1] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        #new_states_image = np.asarray([e[3][0] for e in batch])
        new_states_image = torch.tensor([e[3][0] for e in batch]).to("cuda")
        #new_states_velocity = np.asarray([e[3][1] for e in batch])
        new_states_velocity = torch.tensor([[e[3][1]] for e in batch]).to("cuda")
        dones = np.asarray([e[4] for e in batch])
        y_t = np.asarray([e[1] for e in batch])

        with torch.no_grad():
            target_q_values = self.critic.target_model([[new_states_image, new_states_velocity], self.actor.target_model([new_states_image, new_states_velocity])])  
        target_q_values = self.to_array(target_q_values)

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA*target_q_values[k]

        # train the critic model
        self.critic.model.zero_grad()
        q_values = self.critic.model([[states_image, states_velocity], self.to_tensor(actions)])
        critic_loss = self.critic.train(q_values, self.to_tensor(y_t))

        # train the actor model
        self.actor.model.zero_grad()
        policy_loss = -self.critic.model([[states_image, states_velocity], self.actor.model([states_image, states_velocity])])
        self.actor.train(policy_loss)

        self.actor.target_train()
        self.critic.target_train()
        return critic_loss
    
    def save_model(self):
        print("Saving model now...")
        torch.save(self.actor.model.state_dict(), "./models/controller/actor.pt")
        torch.save(self.critic.model.state_dict(), "./models/controller/critic.pt")
        
    def to_tensor(self, numpy_array):
        return torch.from_numpy(numpy_array).float().cuda()
    
    def to_array(self, torch_tensor):
        return torch_tensor.cpu().float().numpy()