#!/usr/bin/env python3
"""
Quick test of ACER without loading patient data
"""

import numpy as np
import torch

class LambdaTuningACER(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(300, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Tanh()
        )
    
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.actor(state_tensor).item()
    
    def apply_lambda_adjustments(self, current_lambdas, action):
        λ_ptv, λ_bla, λ_rec = current_lambdas
        
        if -0.33 <= action <= 0.33:
            λ_ptv *= (1 + action)
        elif 0.33 < action <= 0.66:
            λ_bla *= (1 + action - 0.33)
        elif 0.66 < action <= 1.0:
            λ_rec *= (1 + action - 0.33)
        elif -0.66 <= action < -0.33:
            λ_bla *= (1 + action + 0.33)
        elif -1.0 <= action < -0.66:
            λ_rec *= (1 + action + 0.66)
            
        return [max(0.1, min(10, λ_ptv)), 
                max(0.1, min(20, λ_bla)), 
                max(0.1, min(20, λ_rec))]

# Test it
agent = LambdaTuningACER()
state = np.random.rand(300)  # Random DVH state

print("Quick ACER Test:")
print("=" * 50)

for i in range(5):
    action = agent.select_action(state)
    lambdas = agent.apply_lambda_adjustments([1.0, 5.0, 5.0], action)
    print(f"Test {i+1}: Action={action:6.3f} -> Lambdas={[f'{x:.2f}' for x in lambdas]}")