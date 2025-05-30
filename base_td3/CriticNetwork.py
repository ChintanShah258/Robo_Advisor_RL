import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir = 'base_td3/critic/checkpoints'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_enhanced_td3')
        
        #Defining our Actual Neural Network
        # Linear(Inputs,Outputs)
        # FIXED
        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.ln1 = nn.LayerNorm(self.fc1_dims)
        #Since the output of the first layer acts as inputs of the second layer in a Fully Connected NN
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.ln2 = nn.LayerNorm(self.fc2_dims)        #Output layer q1 = Linear(input,output_values)
        # For our Robo Advisor output would be of the [(actions),lambda] 
        self.q1 = nn.Linear(self.fc2_dims,1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)
    
    #Feed Forward Neural Network
    def forward(self, state, action):
        # 1) concatenate state and action
        x = T.cat([state, action], dim=1)
        # 2) first hidden layer + LayerNorm + activation
        x = F.relu(self.ln1(self.fc1(x)))
        # 3) second hidden layer + LayerNorm + activation
        x = F.relu(self.ln2(self.fc2(x)))
        # 4) final Q-value head
        q1 = self.q1(x)
        return q1
    
    def save_checkpoint(self):
        print('..saving checkpoint..')
        # ensure directory still exists (in case it was removed)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print('..loading checkpoint..')
        self.load_state_dict(T.load(self.checkpoint_file))