import numpy as np
import torch
import matplotlib.pyplot as plt
from SinglePatientScore import (
    loadDoseMatrix, loadMask, ProcessDmat, 
    runOpt_dvh, planIQ_train, pdose, maxiter
)
from SinglePatientScore import LambdaTuningACER, TreatmentEnv

def main():
    # Load patient data
    print("Loading patient data...")
    data_path = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/prostate_dijs/f_dijs/'
    data_path2 = '/media/mainul/Chi-Drives/128641-3-E21/var/lib/docker/overlay2/8fa0092f3bc6971c1752f734f8fd34c782377f383ffe6f5a55629a01d3b0f185/diff/home/exx/dose_deposition_full/plostate_dijs/f_masks/'

    repList = ['010']
    sampleid = repList[0]
    
    # Load dose matrix and masks
    doseMatrix = loadDoseMatrix(data_path + sampleid + '.hdf5')
    targetLabels, bladderLabel, rectumLabel, PTVLabel = loadMask(data_path2 + sampleid + '.h5')
    
    # Process dose matrix
    MPTV, MBLA, MREC, MBLA1, MREC1 = ProcessDmat(doseMatrix, targetLabels, bladderLabel, rectumLabel)
    
    print(f"Data loaded: MPTV shape {MPTV.shape}, MBLA shape {MBLA.shape}, MREC shape {MREC.shape}")
    
    # Initialize ACER agent and environment
    agent = LambdaTuningACER()
    env = TreatmentEnv()
    
    # Initial treatment planning parameters
    initial_params = {
        'tPTV': 1.0,
        'tBLA': 0.9,
        'tREC': 0.9,
        'lambdaPTV': 1.0,
        'lambdaBLA': 5.0,
        'lambdaREC': 5.0,
        'VPTV': 0.1,
        'VBLA': 0.4,
        'VREC': 0.4
    }
    
    # Training parameters
    num_episodes = 100
    current_lambdas = [initial_params['lambdaPTV'], initial_params['lambdaBLA'], initial_params['lambdaREC']]
    
    # Store results for analysis
    episode_rewards = []
    episode_scores = []
    lambda_history = []
    
    print("Starting ACER training...")
    
    for episode in range(num_episodes):
        state = env.reset()
        Score_fine, Score = 0, 0
        episode_reward = 0
        
        for t in range(env.time_limit):
            # Select action
            action_scalar = agent.select_action(state)
            
            # Execute action
            result = env.step(
                [action_scalar], t, Score_fine, Score,
                MPTV, MBLA, MREC, MBLA1, MREC1,
                initial_params['tPTV'], initial_params['tBLA'], initial_params['tREC'],
                *current_lambdas,
                initial_params['VPTV'], initial_params['VBLA'], initial_params['VREC'],
                pdose, maxiter
            )
            
            # Unpack results
            next_state, reward, Score_fine, Score, done, *rest = result
            current_lambdas = rest[3:6]  # Extract updated lambdas
            
            # Store experience and update
            agent.store_transition(state, action_scalar, reward, next_state)
            critic_loss, actor_loss = agent.update()
            
            episode_reward += reward
            state = next_state
            
            if done:
                print(f"Episode {episode}: Perfect score achieved at step {t}!")
                break
        
        # Store results
        episode_rewards.append(episode_reward)
        episode_scores.append(Score)
        lambda_history.append(current_lambdas.copy())
        
        # Log progress
        if episode % 10 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Score={Score}, "
                  f"Lambdas={[f'{x:.2f}' for x in current_lambdas]}")
    
    # Plot results
    plot_results(episode_rewards, episode_scores, lambda_history)
    
    # Test final policy
    test_final_policy(agent, env, MPTV, MBLA, MREC, MBLA1, MREC1, initial_params, current_lambdas)

def plot_results(rewards, scores, lambda_history):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot scores
    ax2.plot(scores)
    ax2.set_title('Plan Scores')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    
    # Plot lambda history
    lambda_history = np.array(lambda_history)
    ax3.plot(lambda_history[:, 0], label='λ_PTV')
    ax3.plot(lambda_history[:, 1], label='λ_BLA')
    ax3.plot(lambda_history[:, 2], label='λ_REC')
    ax3.set_title('Lambda Parameter Evolution')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Lambda Value')
    ax3.legend()
    ax3.grid(True)
    
    # Plot lambda ratios
    ax4.plot(lambda_history[:, 1] / lambda_history[:, 0], label='λ_BLA/λ_PTV')
    ax4.plot(lambda_history[:, 2] / lambda_history[:, 0], label='λ_REC/λ_PTV')
    ax4.set_title('Lambda Ratios')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Ratio')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('acer_training_results.png')
    plt.show()

def test_final_policy(agent, env, MPTV, MBLA, MREC, MBLA1, MREC1, initial_params, final_lambdas):

    print("\nTesting final policy...")
    
    # Run optimization with final lambdas
    xVec = np.ones((MPTV.shape[1],))
    gamma = np.zeros((MPTV.shape[0],))
    
    state_test, iter, xVec = runOpt_dvh(
        MPTV, MBLA, MREC, 
        initial_params['tPTV'], initial_params['tBLA'], initial_params['tREC'],
        final_lambdas[0], final_lambdas[1], final_lambdas[2],
        initial_params['VPTV'], initial_params['VBLA'], initial_params['VREC'],
        xVec, gamma, pdose, maxiter
    )
    
    # Calculate final scores
    Score_fine, Score, scoreall = planIQ_train(MPTV, MBLA1, MREC1, xVec, pdose, True)
    
    print(f"Final Results:")
    print(f"  Iterations: {iter}")
    print(f"  Fine Score: {Score_fine:.2f}")
    print(f"  Total Score: {Score}")
    print(f"  Lambda PTV: {final_lambdas[0]:.2f}")
    print(f"  Lambda BLA: {final_lambdas[1]:.2f}")
    print(f"  Lambda REC: {final_lambdas[2]:.2f}")
    print(f"  Score Breakdown: {scoreall}")
    
    # Plot final DVH
    plot_final_dvh(state_test)

def plot_final_dvh(state):

    Y = np.reshape(state, (100, 3), order='F')
    
    y_ptv = Y[:, 0] * 100
    y_bladder = Y[:, 1] * 100
    y_rectum = Y[:, 2] * 100
    
    x_ptv = np.linspace(1, 1.15, 100) * 100
    x_bladder = np.linspace(0.6, 1.1, 100) * 100
    x_rectum = np.linspace(0.6, 1.1, 100) * 100
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_ptv, y_ptv, color='red', linewidth=2, label='PTV')
    plt.plot(x_bladder, y_bladder, color='green', linewidth=2, label='Bladder')
    plt.plot(x_rectum, y_rectum, color='blue', linewidth=2, label='Rectum')
    
    plt.xlabel('Relative Dose (%)')
    plt.ylabel('Relative Volume (%)')
    plt.title('Final DVH - ACER Optimized Plan')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('final_dvh_acer.png')
    plt.show()

if __name__ == "__main__":
    main()