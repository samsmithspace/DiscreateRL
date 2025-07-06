#!/usr/bin/env python3
"""
Test script for saved autoencoder and transition models
Performs 3-step rollouts and creates visualizations
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add the parent directory to path to import modules
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from env_helpers import make_env
from model_construction import construct_ae_model, construct_trans_model
from training_helpers import get_args
from visualization import states_to_imgs
from utils import obs_to_img


def load_models(args):
    """Load the saved autoencoder and transition models"""

    # Create sample environment to get observation shape
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    sample_obs = env.reset()
    if isinstance(sample_obs, tuple):
        sample_obs = sample_obs[0]
    sample_obs = torch.from_numpy(sample_obs).float().unsqueeze(0)

    print(f"Loading models for environment: {args.env_name}")
    print(f"Sample observation shape: {sample_obs.shape}")

    # Load autoencoder
    encoder_model, _ = construct_ae_model(sample_obs.shape[1:], args, load=True)
    encoder_model = encoder_model.to(args.device)
    encoder_model.eval()
    print(f"✓ Loaded autoencoder with {args.latent_dim}D latent space")

    # Load transition model
    trans_model, _ = construct_trans_model(encoder_model, args, env.action_space, load=True)
    trans_model = trans_model.to(args.device)
    trans_model.eval()
    print(f"✓ Loaded transition model")

    env.close()
    return encoder_model, trans_model


def collect_real_rollouts(env, n_rollouts=5, n_steps=3):
    """Collect real environment rollouts for comparison"""
    rollouts = []

    for _ in range(n_rollouts):
        env.reset()
        observations = []
        actions = []
        rewards = []
        dones = []

        # Collect initial observation
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        observations.append(obs.copy())

        # Perform n_steps
        for step in range(n_steps):
            action = env.action_space.sample()
            actions.append(action)

            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            observations.append(obs.copy())
            rewards.append(reward)
            dones.append(done)

            if done:
                break

        rollouts.append({
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        })

    return rollouts


def predict_rollouts(encoder_model, trans_model, real_rollouts, device):
    """Generate predicted rollouts using the models"""
    predicted_rollouts = []

    for rollout in real_rollouts:
        real_obs = rollout['observations']
        actions = rollout['actions']

        # Encode initial observation
        init_obs = torch.from_numpy(real_obs[0]).float().unsqueeze(0).to(device)
        current_latent = encoder_model.encode(init_obs)

        # Store predictions
        predicted_obs = [real_obs[0]]  # Start with real initial observation
        predicted_rewards = []
        predicted_dones = []

        # Predict each step
        for i, action in enumerate(actions):
            if i >= len(actions):
                break

            action_tensor = torch.tensor([action]).long().to(device)

            # Predict next state
            with torch.no_grad():
                next_latent, pred_reward, pred_gamma = trans_model(current_latent, action_tensor)
                predicted_obs_tensor = encoder_model.decode(next_latent)

            # Convert back to numpy
            predicted_obs.append(predicted_obs_tensor.cpu().numpy().squeeze())
            predicted_rewards.append(pred_reward.cpu().item())
            predicted_dones.append(pred_gamma.cpu().item() < 0.5)  # Assuming gamma < 0.5 means done

            # Update for next iteration
            current_latent = next_latent

            # Stop if predicted done
            if predicted_dones[-1]:
                break

        predicted_rollouts.append({
            'observations': predicted_obs,
            'rewards': predicted_rewards,
            'dones': predicted_dones
        })

    return predicted_rollouts


def create_comparison_visualization(real_rollouts, predicted_rollouts, env_name, save_path=None):
    """Create side-by-side visualization of real vs predicted rollouts"""

    n_rollouts = min(len(real_rollouts), len(predicted_rollouts), 3)  # Show max 3 rollouts

    fig = plt.figure(figsize=(16, 4 * n_rollouts))

    for rollout_idx in range(n_rollouts):
        real_rollout = real_rollouts[rollout_idx]
        pred_rollout = predicted_rollouts[rollout_idx]

        real_obs = real_rollout['observations']
        pred_obs = pred_rollout['observations']
        real_rewards = real_rollout['rewards']
        pred_rewards = pred_rollout['rewards']

        max_steps = min(len(real_obs), len(pred_obs))

        # Create subplot for this rollout
        gs = GridSpec(2, max_steps, figure=fig,
                      height_ratios=[1, 1],
                      hspace=0.3, wspace=0.1,
                      top=1 - rollout_idx * (1 / n_rollouts),
                      bottom=1 - (rollout_idx + 1) * (1 / n_rollouts))

        for step in range(max_steps):
            # Real observation
            ax_real = fig.add_subplot(gs[0, step])
            real_img = states_to_imgs(
                torch.from_numpy(real_obs[step]).unsqueeze(0),
                env_name
            )[0]
            if len(real_img.shape) == 3 and real_img.shape[0] <= 3:
                real_img = real_img.transpose(1, 2, 0)
            ax_real.imshow(real_img.clip(0, 1))
            ax_real.set_title(f'Real Step {step}' +
                              (f'\nR: {real_rewards[step - 1]:.2f}' if step > 0 and step - 1 < len(
                                  real_rewards) else ''))
            ax_real.axis('off')

            # Predicted observation
            ax_pred = fig.add_subplot(gs[1, step])
            if step < len(pred_obs):
                pred_img = states_to_imgs(
                    torch.from_numpy(pred_obs[step]).unsqueeze(0),
                    env_name
                )[0]
                if len(pred_img.shape) == 3 and pred_img.shape[0] <= 3:
                    pred_img = pred_img.transpose(1, 2, 0)
                ax_pred.imshow(pred_img.clip(0, 1))
                ax_pred.set_title(f'Pred Step {step}' +
                                  (f'\nR: {pred_rewards[step - 1]:.2f}' if step > 0 and step - 1 < len(
                                      pred_rewards) else ''))
            else:
                ax_pred.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax_pred.transAxes)
                ax_pred.set_title('N/A')
            ax_pred.axis('off')

    plt.suptitle(f'Real vs Predicted Rollouts - {env_name}', fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()
    return fig


def calculate_metrics(real_rollouts, predicted_rollouts):
    """Calculate comparison metrics between real and predicted rollouts"""
    metrics = {
        'mse_errors': [],
        'reward_errors': [],
        'steps_completed': []
    }

    for real_rollout, pred_rollout in zip(real_rollouts, predicted_rollouts):
        real_obs = real_rollout['observations']
        pred_obs = pred_rollout['observations']
        real_rewards = real_rollout['rewards']
        pred_rewards = pred_rollout['rewards']

        # Calculate MSE for observations
        min_steps = min(len(real_obs), len(pred_obs))
        for i in range(1, min_steps):  # Skip initial observation (same for both)
            real_tensor = torch.from_numpy(real_obs[i]).float()
            pred_tensor = torch.from_numpy(pred_obs[i]).float()
            mse = torch.mean((real_tensor - pred_tensor) ** 2).item()
            metrics['mse_errors'].append(mse)

        # Calculate reward prediction errors
        min_rewards = min(len(real_rewards), len(pred_rewards))
        for i in range(min_rewards):
            reward_error = abs(real_rewards[i] - pred_rewards[i])
            metrics['reward_errors'].append(reward_error)

        metrics['steps_completed'].append(min_steps - 1)  # -1 because we don't count initial step

    return metrics


def main():
    """Main testing function"""
    import argparse
    from training_helpers import make_argparser, process_args

    # Use the same argument parser as the training script
    parser = make_argparser()

    # Add testing-specific arguments
    parser.add_argument('--n_rollouts', type=int, default=5,
                        help='Number of rollouts to test')
    parser.add_argument('--n_steps', type=int, default=3,
                        help='Number of steps per rollout')
    parser.add_argument('--save_viz', type=str, default='rollout_comparison.png',
                        help='Path to save visualization')

    # Parse and process arguments (this adds all the missing attributes)
    args = parser.parse_args()
    args = process_args(args)

    print("=" * 60)
    print("TESTING SAVED MODELS")
    print("=" * 60)

    # Load models
    try:
        encoder_model, trans_model = load_models(args)
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Make sure you have trained and saved models with the same parameters!")
        return

    # Create environment
    env = make_env(args.env_name)
    print(f"✓ Created test environment: {args.env_name}")

    # Collect real rollouts
    print(f"\n📊 Collecting {args.n_rollouts} real rollouts ({args.n_steps} steps each)...")
    real_rollouts = collect_real_rollouts(env, args.n_rollouts, args.n_steps)
    print(f"✓ Collected {len(real_rollouts)} real rollouts")

    # Generate predictions
    print(f"🤖 Generating predicted rollouts...")
    predicted_rollouts = predict_rollouts(encoder_model, trans_model, real_rollouts, args.device)
    print(f"✓ Generated {len(predicted_rollouts)} predicted rollouts")

    # Calculate metrics
    print(f"\n📈 Calculating comparison metrics...")
    metrics = calculate_metrics(real_rollouts, predicted_rollouts)

    print(f"\n📊 RESULTS:")
    print(f"Average observation MSE: {np.mean(metrics['mse_errors']):.4f}")
    print(f"Average reward error: {np.mean(metrics['reward_errors']):.4f}")
    print(f"Average steps completed: {np.mean(metrics['steps_completed']):.1f}")

    # Create visualization
    print(f"\n🎨 Creating visualization...")
    fig = create_comparison_visualization(real_rollouts, predicted_rollouts,
                                          args.env_name, args.save_viz)

    env.close()
    print(f"\n✅ Testing complete!")


if __name__ == "__main__":
    main()