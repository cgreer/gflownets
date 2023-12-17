import argparse
from collections import defaultdict
import math
import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from FourierEnv import Env as FourierEnvironment
from smiley import Env as SmileyEnvironment
from traj_balance import Trainer, TrajBalMLP

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Trajectory Balancing Training Script')
    
    # Arguments for Trainer
    parser.add_argument('--n_episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon value for training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--temp', type=float, default=1, help='Temperature')
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints', help='Directory to save model checkpoints')

    # GPU flag
    parser.add_argument('--gpu', type=bool, default=False, help='Use GPU if available')

    return parser.parse_args()

def main():
    args = parse_arguments()
    print(args.gpu)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    env = FourierEnvironment()

    trainer = Trainer(env=env, device=device)
    trainer.train(n_episodes=args.n_episodes, eps=args.eps, batch_size=args.batch_size, temp=args.temp)
    trainer.dashboard()

    predicted, target = compare_predicted_to_target(env, device)
    print(predicted)


def evaluate_distribution(model_checkpoint_path, env, device, num_simulations=10000):
    """
    Evaluate the distribution of terminal states for a given model checkpoint.
    """
    model = TrajBalMLP(input_dim=env.encoded_state_size(),
                       output_dim=env.encoded_action_size(),
                       device=device)
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.to(device)
    model.eval()

    termination_frequencies = np.zeros((env.spawn().H, env.spawn().H))

    for i in range(num_simulations):
        #print("Simulation: ", i)
        episode = env.spawn()
        while not episode.done():
            state = episode.current()
            pf_logits, _ = model(state.encode())
            action_idx = Categorical(logits=pf_logits).sample().item()
            action = env.to_action(action_idx)
            episode.step(action)
        
        terminal_state = episode.current()
        x, y, _ = terminal_state.features
        termination_frequencies[x, y] += 1 

    return termination_frequencies


def normalize_distribution(distribution):
    """
    Normalize the distribution so that it sums to 1.
    """
    return distribution / np.sum(distribution)


def calculate_L1_norm(dist1, dist2):
    """
    Calculate the L1 norm between the true reward distribution and the termination frequencies.
    """
    l1_norm = np.sum(np.abs(dist1 - dist2))
    return l1_norm


def visualize_arrays(self, array):
    """
    Visualize  numpy array.
    """
    plt.imshow(array, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title("64x64 Array Visualization")
    plt.show()


def evaluate_checkpoints(base_directory, env, device, num_simulations=5000):
    """
    Evaluate the L1 norm of the model checkpoints over training steps.
    """
    checkpoints = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    l1_norms = []
    step_numbers = []

    for checkpoint_dir in sorted(checkpoints, key=lambda x: int(x.split('_')[1])):
        checkpoint_path = os.path.join(base_directory, checkpoint_dir)
        model_path = os.path.join(checkpoint_path, 'model.pth')
        json_path = os.path.join(checkpoint_path, 'checkpoint_info.json')

        with open(json_path, 'r') as f:
            checkpoint_info = json.load(f)

        print("Evaluating model at step number: ", checkpoint_info['step_number'])
        empirical_distribution = evaluate_distribution(model_path, env, device, num_simulations)

        l1_norm = calculate_L1_norm(normalize_distribution(env.reward_distribution), normalize_distribution(empirical_distribution))
        l1_norms.append(l1_norm)
        step_numbers.append(checkpoint_info['step_number'])

    plt.plot(step_numbers, l1_norms, marker='o')
    plt.xlabel('Step Number')
    plt.ylabel('L1 Norm')
    plt.title('L1 Norm of Model Checkpoints Over Training Steps')
    plt.grid(True)
    plt.show()

    return step_numbers, l1_norms

def compare_predicted_to_target(env, device, model_checkpoint_path="./model_checkpoints/checkpoint_last", num_simulations=2000):
    """
    Compare the predicted distribution to the target distribution.
    """
    path = os.path.join(model_checkpoint_path, 'model.pth')
    predicted = evaluate_distribution(path, env, device, num_simulations)
    target = env.reward_distribution
    normalised_predicted = normalize_distribution(predicted)
    normalised_target = normalize_distribution(target)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax = axes[0]
    im = ax.imshow(normalised_predicted, cmap='gray', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title("Predicted Distribution")

    ax = axes[1]
    im = ax.imshow(normalised_target, cmap='gray', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title("Target Distribution")

    plt.show()

    return predicted, target


if __name__ == "__main__":
    main()


