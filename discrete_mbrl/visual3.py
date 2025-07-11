#!/usr/bin/env python3
"""
Working Model Visualizer - Loads your actual trained models with debugging
"""

import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import glob
import re

# Import your modules
from env_helpers import make_env
from model_construction import construct_ae_model, construct_trans_model
from training_helpers import batch_obs_resize, get_obs_target_size
from visualization import states_to_imgs
from utils import obs_to_img


def try_all_model_configs():
    """Try different model configurations to match your trained models"""

    # Common configurations that might match your training
    configs = [
        # Config 1: Basic AE with different latent dims
        {'ae_model_type': 'ae', 'trans_model_type': 'continuous', 'latent_dim': 64, 'embedding_dim': 64},
        {'ae_model_type': 'ae', 'trans_model_type': 'continuous', 'latent_dim': 256, 'embedding_dim': 64},
        {'ae_model_type': 'ae', 'trans_model_type': 'continuous', 'latent_dim': 1024, 'embedding_dim': 64},
        {'ae_model_type': 'ae', 'trans_model_type': 'continuous', 'latent_dim': 16, 'embedding_dim': 64},

        # Config 2: Different stochastic settings
        {'ae_model_type': 'ae', 'trans_model_type': 'continuous', 'latent_dim': 256, 'stochastic': 'categorical'},
        {'ae_model_type': 'ae', 'trans_model_type': 'continuous', 'latent_dim': 64, 'stochastic': None},

        # Config 3: VQVAE variants
        {'ae_model_type': 'vqvae', 'trans_model_type': 'discrete', 'codebook_size': 64, 'filter_size': 6},
        {'ae_model_type': 'vqvae', 'trans_model_type': 'discrete', 'codebook_size': 256, 'filter_size': 6},
        {'ae_model_type': 'vqvae', 'trans_model_type': 'discrete', 'codebook_size': 16, 'filter_size': 8},

        # Config 4: Different embedding dims
        {'ae_model_type': 'ae', 'trans_model_type': 'continuous', 'latent_dim': 256, 'embedding_dim': 128},
        {'ae_model_type': 'ae', 'trans_model_type': 'continuous', 'latent_dim': 256, 'embedding_dim': 32},
    ]

    return configs


def create_args_from_config(config):
    """Create full args object from config"""

    class ModelArgs:
        def __init__(self):
            # Default values
            self.ae_model_type = 'ae'
            self.ae_model_version = '2'
            self.trans_model_type = 'continuous'
            self.trans_model_version = '1'
            self.embedding_dim = 64
            self.latent_dim = 256
            self.filter_size = 8
            self.codebook_size = 16
            self.trans_hidden = 256
            self.trans_depth = 3
            self.stochastic = 'simple'
            self.env_name = 'MiniGrid-Empty-6x6-v0'
            self.env_max_steps = None
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.load = True

            # All other required attributes
            self.wandb = False
            self.comet_ml = False
            self.tags = None
            self.epochs = 0
            self.batch_size = 512
            self.learning_rate = 0.0002
            self.trans_learning_rate = 0.0002
            self.log_freq = 500
            self.checkpoint_freq = 10
            self.save = False
            self.upload_model = False
            self.max_transitions = None
            self.preprocess = False
            self.unique_data = False
            self.cache = True
            self.n_preload = 0
            self.preload_data = False
            self.n_train_unroll = 4
            self.exact_comp = False
            self.log_state_reprs = False
            self.eval_batch_size = 128
            self.eval_unroll_steps = 20
            self.eval_policies = ['random']
            self.extra_info = None
            self.repr_sparsity = 0
            self.sparsity_type = 'random'
            self.vq_trans_1d_conv = False
            self.vq_trans_state_snap = False
            self.vq_trans_loss_type = 'mse'
            self.e2e_loss = False
            self.log_norms = False
            self.ae_grad_clip = 0
            self.recon_loss_clip = 0
            self.extra_buffer_keys = []
            self.fta_tiles = 20
            self.fta_bound_low = -2
            self.fta_bound_high = 2
            self.fta_eta = 0.2
            self.ae_model_hash = None
            self.trans_model_hash = None
            self.rl_train_steps = 0
            self.rl_unroll_steps = -1
            self.obs_resize = None
            self.obs_resize_mode = 'bilinear'
            self.no_obs_resize = False

    args = ModelArgs()

    # Override with config
    for key, value in config.items():
        setattr(args, key, value)

    return args


def debug_models(encoder_model, trans_model, env, device):
    """Quick debug to check if models work"""
    print("\n=== DEBUGGING MODELS ===")

    # Get a fresh observation
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    print(f"Original obs shape: {obs.shape}, dtype: {obs.dtype}, range: [{obs.min()}, {obs.max()}]")

    # Check target size
    target_size = get_obs_target_size(env.spec.id if hasattr(env, 'spec') else 'MiniGrid-Empty-6x6-v0')
    print(f"Target resize: {target_size}")

    # Prepare observation
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

    # Try different normalizations
    best_config = None
    for norm_type in ['none', 'div255', 'div255_resized', 'standardize', 'minigrid_special']:
        print(f"\n--- Testing with {norm_type} normalization ---")

        test_obs = obs_tensor.clone()

        # Apply normalization
        if norm_type == 'div255':
            test_obs = test_obs / 255.0
        elif norm_type == 'div255_resized':
            # Resize first, then normalize
            if target_size:
                test_obs = batch_obs_resize(test_obs, env_name='MiniGrid-Empty-6x6-v0')
            test_obs = test_obs / 255.0
        elif norm_type == 'standardize':
            test_obs = (test_obs - 128.0) / 128.0
        elif norm_type == 'minigrid_special':
            # MiniGrid specific: observations might already be normalized
            if target_size:
                test_obs = batch_obs_resize(test_obs, env_name='MiniGrid-Empty-6x6-v0')
            # Check if already normalized
            if test_obs.max() <= 1.0:
                pass  # Already normalized
            else:
                test_obs = test_obs / 255.0
        else:  # 'none'
            if target_size and norm_type == 'none':
                test_obs = batch_obs_resize(test_obs, env_name='MiniGrid-Empty-6x6-v0')

        print(f"Input tensor shape: {test_obs.shape}, range: [{test_obs.min():.3f}, {test_obs.max():.3f}]")

        # Encode
        with torch.no_grad():
            try:
                z = encoder_model.encode(test_obs)
                print(f"Encoded z shape: {z.shape}, range: [{z.min():.3f}, {z.max():.3f}]")

                # Check for NaN or Inf
                if torch.isnan(z).any() or torch.isinf(z).any():
                    print("WARNING: z contains NaN or Inf values!")
                    continue

                # Decode
                reconstructed = encoder_model.decode(z)
                print(
                    f"Decoded shape: {reconstructed.shape}, range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

                # Check if it's not all zeros
                if reconstructed.abs().max() > 0.001:
                    print("✓ Model produces non-zero output!")

                    # Test transition model too
                    action = torch.tensor([0]).to(device)
                    trans_output = trans_model(z, action)
                    if isinstance(trans_output, tuple):
                        z_next = trans_output[0]
                    else:
                        z_next = trans_output

                    print(f"Transition output shape: {z_next.shape}, range: [{z_next.min():.3f}, {z_next.max():.3f}]")

                    # Decode next state
                    next_reconstructed = encoder_model.decode(z_next)
                    print(f"Next decoded range: [{next_reconstructed.min():.3f}, {next_reconstructed.max():.3f}]")

                    if next_reconstructed.abs().max() > 0.001:
                        best_config = {
                            'normalization': norm_type,
                            'needs_resize': target_size is not None,
                            'input_range': (test_obs.min().item(), test_obs.max().item()),
                            'output_range': (reconstructed.min().item(), reconstructed.max().item())
                        }
                        break
                else:
                    print("✗ Model output is all zeros")

            except Exception as e:
                print(f"Error with {norm_type}: {e}")
                continue

    return best_config


def load_any_available_models():
    """Try to load any available trained models"""
    print("Trying to load your trained models...")
    print("Available model files:")

    # List available models
    model_files = glob.glob('./models/MiniGrid-Empty-6x6-v0/*.pt')
    for i, f in enumerate(model_files):
        size = os.path.getsize(f) / 1024 / 1024
        print(f"  {i + 1}. {os.path.basename(f)} ({size:.1f} MB)")

    # Create environment
    env = make_env('MiniGrid-Empty-6x6-v0')
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs_shape = obs.shape

    # Try different configurations
    configs = try_all_model_configs()

    for i, config in enumerate(configs):
        print(f"\nTrying configuration {i + 1}: {config}")

        args = create_args_from_config(config)

        try:
            # Try to load encoder
            encoder_result = construct_ae_model(obs_shape, args, load=True)
            encoder_model = encoder_result[0] if isinstance(encoder_result, tuple) else encoder_result

            if encoder_model is not None:
                encoder_model = encoder_model.to(args.device)
                encoder_model.eval()

                # Try to load transition model
                trans_result = construct_trans_model(encoder_model, args, env.action_space, load=True)
                trans_model = trans_result[0] if isinstance(trans_result, tuple) else trans_result

                if trans_model is not None:
                    trans_model = trans_model.to(args.device)
                    trans_model.eval()

                    print(f"✓ SUCCESS! Loaded models with configuration {i + 1}")
                    print(f"  Encoder: {args.ae_model_type}")
                    print(f"  Transition: {args.trans_model_type}")
                    print(f"  Latent dim: {getattr(args, 'latent_dim', 'N/A')}")
                    print(f"  Codebook size: {getattr(args, 'codebook_size', 'N/A')}")

                    # Run debug check
                    debug_config = debug_models(encoder_model, trans_model, env, args.device)

                    return encoder_model, trans_model, env, args, debug_config

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    print("\n✗ Could not load any trained models with available configurations")
    return None, None, None, None, None




def preprocess_observation(obs, obs_tensor, debug_config, env_name='MiniGrid-Empty-6x6-v0'):
    """Preprocess observation based on debug findings"""
    if debug_config is None:
        # Fallback to default
        target_size = get_obs_target_size(env_name)
        if target_size:
            obs_tensor = batch_obs_resize(obs_tensor, env_name=env_name)
        return obs_tensor / 255.0

    # Apply the working configuration
    norm_type = debug_config['normalization']

    if norm_type == 'div255':
        return obs_tensor / 255.0
    elif norm_type == 'div255_resized':
        if debug_config['needs_resize']:
            obs_tensor = batch_obs_resize(obs_tensor, env_name=env_name)
        return obs_tensor / 255.0
    elif norm_type == 'standardize':
        return (obs_tensor - 128.0) / 128.0
    elif norm_type == 'minigrid_special':
        if debug_config['needs_resize']:
            obs_tensor = batch_obs_resize(obs_tensor, env_name=env_name)
        if obs_tensor.max() <= 1.0:
            return obs_tensor
        else:
            return obs_tensor / 255.0
    else:  # 'none'
        if debug_config['needs_resize']:
            obs_tensor = batch_obs_resize(obs_tensor, env_name=env_name)
        return obs_tensor

def generate_rollout(encoder_model, trans_model, env, args, debug_config, n_steps=10, seed=42):
    """Generate rollout comparing real environment vs model predictions"""
    print(f"\nGenerating {n_steps}-step rollout...")

    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reset environment
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Storage for trajectories
    real_obs = [obs.copy()]
    model_obs = []
    actions = []
    rewards_real = []
    rewards_model = []

    # Get initial encoded state
    device = next(encoder_model.parameters()).device
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)

    # Preprocess observation
    obs_tensor = preprocess_observation(obs, obs_tensor, debug_config, args.env_name)

    print(f"Initial obs tensor range: [{obs_tensor.min():.3f}, {obs_tensor.max():.3f}]")

    # Get initial encoding
    with torch.no_grad():
        z = encoder_model.encode(obs_tensor)
        print(f"Initial z range: [{z.min():.3f}, {z.max():.3f}]")

        # Test initial reconstruction
        test_recon = encoder_model.decode(z)
        print(f"Initial reconstruction range: [{test_recon.min():.3f}, {test_recon.max():.3f}]")

    print("\nStep-by-step rollout:")
    # Generate rollout
    for step in range(n_steps):
        # Sample action
        action = env.action_space.sample()
        actions.append(action)

        print(f"  Step {step + 1}: Action = {action}")

        # Real environment step
        real_result = env.step(action)
        if len(real_result) == 4:
            real_obs_next, real_reward, done, info = real_result
        else:
            real_obs_next, real_reward, terminated, truncated, info = real_result
            done = terminated or truncated

        real_obs.append(real_obs_next.copy())
        rewards_real.append(real_reward)

        # Model prediction
        with torch.no_grad():
            action_tensor = torch.tensor([action]).to(device)

            # Handle different transition model outputs
            trans_output = trans_model(z, action_tensor)
            if isinstance(trans_output, tuple):
                if len(trans_output) == 3:
                    z_next, reward_pred, gamma_pred = trans_output
                elif len(trans_output) == 2:
                    z_next, reward_pred = trans_output
                    gamma_pred = torch.tensor([1.0])
                else:
                    z_next = trans_output[0]
                    reward_pred = torch.tensor([0.0])
                    gamma_pred = torch.tensor([1.0])
            else:
                z_next = trans_output
                reward_pred = torch.tensor([0.0])
                gamma_pred = torch.tensor([1.0])

            print(f"    z_next range: [{z_next.min():.3f}, {z_next.max():.3f}]")

            # Decode next observation
            obs_pred = encoder_model.decode(z_next)
            print(f"    decoded range: [{obs_pred.min():.3f}, {obs_pred.max():.3f}]")

            # Convert back to original observation space
            if debug_config and 'output_range' in debug_config:
                out_min, out_max = debug_config['output_range']
                # Denormalize if needed
                if out_min >= -1 and out_max <= 1:
                    # Output is normalized, denormalize
                    if out_min >= 0:
                        # [0, 1] range
                        obs_pred = obs_pred * 255.0
                    else:
                        # [-1, 1] range
                        obs_pred = (obs_pred + 1) * 127.5

            # Resize back if needed
            target_size = get_obs_target_size(args.env_name)
            if target_size and obs_pred.shape[-2:] != obs.shape[-2:]:
                obs_pred = batch_obs_resize(obs_pred, target_size=obs.shape[-2:])

            obs_pred_np = obs_pred[0].cpu().numpy()
            print(f"    final pred range: [{obs_pred_np.min():.3f}, {obs_pred_np.max():.3f}]")

        model_obs.append(obs_pred_np)
        reward_val = reward_pred.item() if hasattr(reward_pred, 'item') else float(reward_pred)
        rewards_model.append(reward_val)

        print(f"    Real reward: {real_reward:.3f}, Model reward: {reward_val:.3f}")

        # Update for next iteration
        z = z_next
        obs = real_obs_next

        if done:
            print(f"    Episode ended at step {step + 1}")
            break

    return {
        'real_obs': real_obs,
        'model_obs': model_obs,
        'actions': actions,
        'rewards_real': rewards_real,
        'rewards_model': rewards_model
    }

def obs_to_display_img(obs, debug_info=None):
    """Convert observation to displayable image with better handling"""
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()

    # Handle different observation formats
    if len(obs.shape) == 3:
        if obs.shape[0] <= 3:  # CHW format
            obs = obs.transpose(1, 2, 0)

    # Debug: Check value range
    original_range = (obs.min(), obs.max())
    print(f"DEBUG obs_to_display_img: min={obs.min():.3f}, max={obs.max():.3f}, shape={obs.shape}")

    # Handle different value ranges
    if obs.max() > 10:
        # Assume [0, 255] range
        obs = obs / 255.0
    elif obs.max() > 1.0 and obs.max() <= 10:
        # Might be in [0, N] range where N is small
        obs = obs / obs.max()
    elif obs.min() < -0.1:
        # Assume [-1, 1] range (common for some models)
        obs = (obs + 1) / 2
    elif obs.max() < 0.1:
        # Might be all zeros or very small values
        print(f"WARNING: Very small values detected. Max value: {obs.max()}")
        # Try to enhance contrast
        if obs.max() > 0:
            obs = obs / obs.max()

    # Ensure values are in [0, 1] range
    obs = np.clip(obs, 0, 1)

    # Convert to RGB if grayscale
    if len(obs.shape) == 2:
        obs = np.stack([obs] * 3, axis=-1)
    elif obs.shape[-1] == 1:
        obs = np.stack([obs.squeeze(-1)] * 3, axis=-1)
    elif obs.shape[-1] > 3:
        # Take first 3 channels
        obs = obs[:, :, :3]

    return obs

def create_visualization(rollout_data):
    """Create comprehensive visualization"""
    print("\nCreating visualization...")

    n_steps = len(rollout_data['actions'])

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))

    # Individual step comparison (top section)
    n_show = min(6, n_steps + 1)

    # Real observations row
    for i in range(n_show):
        ax = plt.subplot(4, n_show, i + 1)
        if i < len(rollout_data['real_obs']):
            img = obs_to_display_img(rollout_data['real_obs'][i])
            ax.imshow(img)
            ax.set_title(f'Real Step {i}')
        ax.set_xticks([])
        ax.set_yticks([])

    # Model predictions row
    for i in range(n_show):
        ax = plt.subplot(4, n_show, n_show + i + 1)
        if i > 0 and i - 1 < len(rollout_data['model_obs']):
            img = obs_to_display_img(rollout_data['model_obs'][i - 1])
            ax.imshow(img)
            ax.set_title(f'Model Step {i}')
            # Add debug info
            if img.max() < 0.1:
                ax.text(0.5, 0.5, 'Low/Zero\nValues', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10, color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        elif i == 0:
            ax.text(0.5, 0.5, 'Initial\nState', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('Model Step 0')
        ax.set_xticks([])
        ax.set_yticks([])

    # Difference row
    for i in range(n_show):
        ax = plt.subplot(4, n_show, 2 * n_show + i + 1)
        if (i > 0 and i < len(rollout_data['real_obs']) and
                i - 1 < len(rollout_data['model_obs'])):
            real_img = obs_to_display_img(rollout_data['real_obs'][i])
            model_img = obs_to_display_img(rollout_data['model_obs'][i - 1])
            diff_img = np.abs(real_img.astype(float) - model_img.astype(float))
            im = ax.imshow(diff_img, cmap='hot')
            ax.set_title(f'Diff Step {i}')
            # Add colorbar for difference
            if i == n_show - 1:  # Add colorbar to last diff
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)
        elif i == 0:
            ax.text(0.5, 0.5, 'No\nDifference', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('Diff Step 0')
        ax.set_xticks([])
        ax.set_yticks([])

    # Metrics plot (bottom section)
    ax_metrics = plt.subplot(4, 1, 4)
    steps = range(len(rollout_data['rewards_real']))
    ax_metrics.plot(steps, rollout_data['rewards_real'], 'b-o', label='Real Environment', linewidth=2)
    ax_metrics.plot(steps, rollout_data['rewards_model'], 'r--s', label='Model Prediction', linewidth=2)
    ax_metrics.set_xlabel('Step')
    ax_metrics.set_ylabel('Reward')
    ax_metrics.set_title('Reward Comparison')
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3)

    # Add error statistics
    reward_errors = [abs(r - m) for r, m in zip(rollout_data['rewards_real'], rollout_data['rewards_model'])]
    mean_error = np.mean(reward_errors) if reward_errors else 0
    max_error = np.max(reward_errors) if reward_errors else 0
    ax_metrics.text(0.02, 0.98, f'Mean Error: {mean_error:.3f}\nMax Error: {max_error:.3f}', transform = ax_metrics.transAxes, va = 'top',
    bbox = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Add row labels
    fig.text(0.02, 0.85, 'Real Environment', rotation=90, va='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.65, 'Model Prediction', rotation=90, va='center', fontsize=12, weight='bold')
    fig.text(0.02, 0.45, 'Absolute Difference', rotation=90, va='center', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.1)

    # Save the plot
    save_path = 'trained_model_rollout_debug.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")

    plt.show()

def print_summary(rollout_data, args, debug_config):
    """Print comprehensive summary"""
    print("\n" + "=" * 60)
    print("TRAINED MODEL ROLLOUT SUMMARY")
    print("=" * 60)

    print(f"Model Configuration:")
    print(f"  Encoder: {args.ae_model_type}")
    print(f"  Transition: {args.trans_model_type}")
    print(f"  Latent dim: {getattr(args, 'latent_dim', 'N/A')}")
    print(f"  Embedding dim: {args.embedding_dim}")
    if hasattr(args, 'codebook_size') and args.codebook_size:
        print(f"  Codebook size: {args.codebook_size}")

    if debug_config:
        print(f"\nWorking Configuration:")
        print(f"  Normalization: {debug_config['normalization']}")
        print(f"  Needs resize: {debug_config['needs_resize']}")
        print(f"  Input range: {debug_config['input_range']}")
        print(f"  Output range: {debug_config['output_range']}")

    n_steps = len(rollout_data['actions'])
    print(f"\nRollout Results:")
    print(f"  Total steps: {n_steps}")
    print(f"  Actions: {rollout_data['actions']}")

    # Reward analysis
    real_rewards = rollout_data['rewards_real']
    model_rewards = rollout_data['rewards_model']

    print(f"\nReward Analysis:")
    print(f"  Real total: {sum(real_rewards):.3f}")
    print(f"  Model total: {sum(model_rewards):.3f}")
    print(f"  Real mean: {np.mean(real_rewards):.3f}")
    print(f"  Model mean: {np.mean(model_rewards):.3f}")

    # Error analysis
    reward_errors = [abs(r - m) for r, m in zip(real_rewards, model_rewards)]
    print(f"\nPrediction Accuracy:")
    print(f"  Mean absolute error: {np.mean(reward_errors):.3f}")
    print(f"  Max absolute error: {np.max(reward_errors):.3f}")
    print(f"  Std of errors: {np.std(reward_errors):.3f}")

    # Visual quality check
    print(f"\nVisual Quality Check:")
    for i, model_obs in enumerate(rollout_data['model_obs']):
        obs_min = model_obs.min()
        obs_max = model_obs.max()
        obs_mean = model_obs.mean()
        is_zero = obs_max < 0.001
        print(
            f"  Step {i + 1}: min={obs_min:.3f}, max={obs_max:.3f}, mean={obs_mean:.3f} {'[ALL ZEROS!]' if is_zero else ''}")

    print("=" * 60)

def diagnose_model_issue(encoder_model, trans_model, env, device):
    """Additional diagnostics if models still produce zeros"""
    print("\n=== ADVANCED DIAGNOSTICS ===")

    # Check model parameters
    print("\nModel parameter statistics:")

    # Encoder stats
    total_params = 0
    zero_params = 0
    for name, param in encoder_model.named_parameters():
        total_params += param.numel()
        zero_params += (param.abs() < 1e-8).sum().item()
        if param.abs().max() < 1e-8:
            print(f"  WARNING: Encoder layer {name} has all zero parameters!")

    print(f"  Encoder: {zero_params}/{total_params} zero parameters ({100 * zero_params / total_params:.1f}%)")

    # Check specific layers
    if hasattr(encoder_model, 'encoder'):
        print("\nEncoder network details:")
        for name, module in encoder_model.encoder.named_modules():
            if hasattr(module, 'weight'):
                w_mean = module.weight.mean().item()
                w_std = module.weight.std().item()
                print(f"  {name}: weight mean={w_mean:.3f}, std={w_std:.3f}")

    # Test with known patterns
    print("\nTesting with known patterns:")

    # Test 1: All ones
    test_input = torch.ones(1, *env.observation_space.shape).to(device)
    with torch.no_grad():
        z = encoder_model.encode(test_input)
        recon = encoder_model.decode(z)
    print(f"  All ones test: z mean={z.mean():.3f}, recon mean={recon.mean():.3f}")

    # Test 2: Random noise
    test_input = torch.rand(1, *env.observation_space.shape).to(device)
    with torch.no_grad():
        z = encoder_model.encode(test_input)
        recon = encoder_model.decode(z)
    print(f"  Random test: z mean={z.mean():.3f}, recon mean={recon.mean():.3f}")

    # Test 3: Actual observation pattern
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    test_input = torch.from_numpy(obs).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        z = encoder_model.encode(test_input)
        recon = encoder_model.decode(z)
    print(f"  Real obs test: z mean={z.mean():.3f}, recon mean={recon.mean():.3f}")

    return z, recon

def main():
    print("Trained Model Rollout Visualizer with Debugging")
    print("=" * 50)

    # Load models
    encoder_model, trans_model, env, args, debug_config = load_any_available_models()

    if encoder_model is None:
        print("\nCould not load trained models. Please check:")
        print("1. Models exist in ./models/MiniGrid-Empty-6x6-v0/")
        print("2. Training completed successfully")
        print("3. Try running full_train_eval.py first")
        return

    # If debug_config is None, run additional diagnostics
    if debug_config is None:
        print("\n⚠️  WARNING: Could not find working configuration!")
        print("Running advanced diagnostics...")
        z_test, recon_test = diagnose_model_issue(encoder_model, trans_model, env, args.device)

        # Try to proceed with default config
        debug_config = {
            'normalization': 'div255',
            'needs_resize': True,
            'input_range': (0, 255),
            'output_range': (0, 1)
        }
        print("\nProceeding with default configuration...")

    # Generate rollout
    rollout_data = generate_rollout(encoder_model, trans_model, env, args, debug_config, n_steps=10, seed=42)

    # Check if all model predictions are zeros
    all_zeros = all(obs.max() < 0.001 for obs in rollout_data['model_obs'])
    if all_zeros:
        print("\n⚠️  WARNING: All model predictions are zero/near-zero!")
        print("This typically indicates:")
        print("  1. Model weights are corrupted or not properly loaded")
        print("  2. Input preprocessing mismatch")
        print("  3. Model architecture mismatch")
        print("  4. The model was not trained properly")
        print("\nPlease verify your training completed successfully.")

    # Print summary
    print_summary(rollout_data, args, debug_config)

    # Create visualization
    create_visualization(rollout_data)

    print("\n✓ Visualization complete!")
    print("Check 'trained_model_rollout_debug.png' for results")

    if all_zeros:
        print("\n⚠️  Note: The model predictions appear to be all zeros.")
        print("This visualization shows the issue, but the model needs to be retrained or debugged.")

if __name__ == "__main__":
    main()

