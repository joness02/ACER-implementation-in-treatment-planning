# Import required libraries
import numpy as np  # Numerical operations
import torch  # Deep learning framework
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import torch.nn.functional as F  # Neural network functions
from collections import deque  # Efficient queue for experience replay
from gym.spaces import Box  # Continuous action space definition
import random  # Random sampling

class LambdaTuningACER(nn.Module):
    """Actor-Critic with Experience Replay (ACER) agent for continuous lambda tuning"""
    def __init__(self, gamma=0.99, lr=1e-4):
        """
        Initialize the ACER agent
        Args:
            gamma: Discount factor for future rewards
            lr: Learning rate for optimizer
        """
        super().__init__()
        
        # Actor network - learns policy (state -> action)
        self.actor = nn.Sequential(
            nn.Linear(300, 256),  # Input layer (300D DVH state)
            nn.ReLU(),  # Activation function
            nn.Linear(256, 1),  # Output layer (1D action)
            nn.Tanh()  # Constrain action to [-1, 1]
        )
        
        # Critic network - learns state value function
        self.critic = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs state value
        )
        
        # Training setup
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # Optimizer
        self.gamma = gamma  # How much we discount future rewards
        self.replay_buffer = deque(maxlen=10000)  # Circular buffer for experience replay
        self.batch_size = 64  # Number of experiences per training batch
        self.exploration_noise = 0.1  # Noise added to actions during exploration

    def select_action(self, state, explore=True):
        """
        Select an action given the current state
        Args:
            state: Current DVH state (300D vector)
            explore: Whether to add exploration noise
        Returns:
            action: Scalar value between -1 and 1
        """
        with torch.no_grad():  # Disable gradient calculation for inference
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert to tensor
            action = self.actor(state_tensor)  # Get action from policy
            
            if explore:
                # Add Gaussian noise for exploration
                action += torch.randn_like(action) * self.exploration_noise
            
            # Ensure action stays in valid range
            return torch.clamp(action, -1, 1).item()

    def apply_lambda_adjustments(self, current_lambdas, action):
        """
        Apply the custom action-to-lambda adjustment rules
        Args:
            current_lambdas: [λ_ptv, λ_bla, λ_rec] current values
            action: Scalar between -1 and 1
        Returns:
            Updated lambda values after applying action rules
        """
        λ_ptv, λ_bla, λ_rec = current_lambdas
        a = action  # Scalar action value
        
        # Apply custom action mapping rules
        if -0.33 <= a <= 0.33:  # Small adjustment -> Modify PTV
            λ_ptv *= (1 + a)
        elif 0.33 < a <= 0.66:  # Medium positive -> Modify Bladder
            λ_bla *= (1 + a - 0.33)
        elif 0.66 <= a < 1.0:  # Large positive -> Modify Rectum
            λ_rec *= (1 + a - 0.33)
        elif -0.66 <= a < -0.33:  # Medium negative -> Modify Bladder
            λ_bla *= (1 + a + 0.33)
        elif -1.0 <= a < -0.66:  # Large negative -> Modify Rectum
            λ_rec *= (1 + a + 0.66)
            
        # Enforce safety bounds on lambda values
        λ_ptv = np.clip(λ_ptv, 0.1, 10.0)  # PTV weight bounds
        λ_bla = np.clip(λ_bla, 0.1, 20.0)  # Bladder weight bounds
        λ_rec = np.clip(λ_rec, 0.1, 20.0)  # Rectum weight bounds
        
        return [λ_ptv, λ_bla, λ_rec]

    def store_transition(self, state, action, reward, next_state):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state))

    def update(self):
        """Update actor and critic networks using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None  # Not enough experiences yet
            
        # Randomly sample a batch of experiences
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))

        # Critic update - minimize TD error
        values = self.critic(states)  # Current state values
        with torch.no_grad():
            target_values = rewards + self.gamma * self.critic(next_states)  # TD targets
        critic_loss = F.mse_loss(values, target_values)  # Mean squared error

        # Actor update - policy gradient with advantage
        advantage = target_values - values.detach()  # Advantage estimate
        mu = self.actor(states)  # Current policy actions
        actor_loss = -(mu * advantage).mean()  # Maximize advantage-weighted actions

        # Combined update
        self.optimizer.zero_grad()  # Clear gradients
        (critic_loss + actor_loss).backward()  # Backpropagate
        self.optimizer.step()  # Update weights

        return critic_loss.item(), actor_loss.item()  # Return losses for monitoring

class TreatmentEnv:
    """Environment for radiation therapy treatment planning"""
    def __init__(self):
        # Action space: single continuous value between -1 and 1
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        # Observation space: 300D DVH curves
        self.observation_space = np.zeros(300)
        # Episode length limit
        self.time_limit = 30

    def step(self, action, t, Score_fine, Score, MPTV, MBLA, MREC, MBLA1, MREC1,
             tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC, VPTV, VBLA, VREC, pdose, maxiter):
        """
        Execute one timestep in the environment
        Args:
            action: Scalar action value
            All other args: Current treatment planning parameters
        Returns:
            Tuple containing:
            - next_state: New DVH curves
            - reward: Calculated from planIQ scores
            - Score_fine1, Score1: New scores
            - Done: Whether episode terminated
            - Other updated parameters
        """
        # Apply action to adjust lambda weights
        current_lambdas = [lambdaPTV, lambdaBLA, lambdaREC]
        adjusted_lambdas = agent.apply_lambda_adjustments(current_lambdas, action[0])
        lambdaPTV, lambdaBLA, lambdaREC = adjusted_lambdas

        # Run treatment planning optimization
        xVec = np.ones((MPTV.shape[1],))
        gamma = np.zeros((MPTV.shape[0],))
        n_state, _, xVec = runOpt_dvh(
            MPTV, MBLA, MREC, tPTV, tBLA, tREC,
            lambdaPTV, lambdaBLA, lambdaREC,
            VPTV, VBLA, VREC, xVec, gamma, pdose, maxiter
        )

        # Calculate plan quality scores
        Score_fine1, Score1, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)
        # Original reward calculation
        reward = (Score_fine1 - Score_fine) + (Score1 - Score) * 4
        # Termination condition
        Done = Score1 == 9
        
        return (n_state, reward, Score_fine1, Score1, Done, 
                tPTV, tBLA, tREC, lambdaPTV, lambdaBLA, lambdaREC,
                VPTV, VBLA, VREC, xVec)

    def reset(self):
        """Reset the environment state"""
        return self.observation_space  # Return initial zero state

def train_agent():
    """Main training loop"""
    # Initialize agent and environment
    agent = LambdaTuningACER()
    env = TreatmentEnv()
    
    # Start with original lambda values
    current_lambdas = [1.0, 5.0, 5.0]  # lambdaPTV, lambdaBLA, lambdaREC

    # Training loop
    for episode in range(1000):
        state = env.reset()  # Reset environment
        Score_fine, Score = 0, 0  # Reset scores
        
        # Episode loop
        for t in range(env.time_limit):
            # Get action from policy
            action_scalar = agent.select_action(state)
            
            # Execute action in environment
            result = env.step(
                [action_scalar], t, Score_fine, Score,
                MPTV, MBLA, MREC, MBLA1, MREC1,
                tPTV, tBLA, tREC, *current_lambdas,
                VPTV, VBLA, VREC, pdose, maxiter
            )
            
            # Unpack results
            next_state, reward, Score_fine, Score, done, *rest = result
            current_lambdas = rest[3:6]  # Extract updated lambdas
            
            # Store experience
            agent.store_transition(state, action_scalar, reward, next_state)
            # Update networks
            agent.update()
            
            if done:
                break  # Early termination if perfect score

        # Periodic logging
        if episode % 20 == 0:
            print(f"Episode {episode}, Lambdas: {current_lambdas}")