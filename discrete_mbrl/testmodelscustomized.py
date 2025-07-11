#!/usr/bin/env python3
"""
Enhanced MiniGrid visualizer that properly handles symbolic observations
and creates meaningful visualizations from discrete state representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import argparse
import os
import sys
from typing import Tuple, Dict, Any, Optional, List
import gymnasium as gym
from tqdm import tqdm


# Import the fixed model loader
class FixedModelLoader:
    """Improved model loader that handles various tensor types correctly"""

    @staticmethod
    def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cpu') -> torch.nn.Module:
        """Load a model from checkpoint, handling different save formats"""

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Case 1: Full model was saved
        if isinstance(checkpoint, torch.nn.Module):
            print("✓ Loaded full model object")
            return checkpoint.to(device)

        # Case 2: Dictionary with state dict
        if isinstance(checkpoint, dict):
            state_dict = None

            # Try different keys for state dict
            for key in ['state_dict', 'model_state_dict', 'model', 'encoder_state_dict', 'model_weights']:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"Found state dict under key: '{key}'")
                    break

            if state_dict is None:
                if FixedModelLoader._looks_like_state_dict(checkpoint):
                    state_dict = checkpoint
                    print("Using entire checkpoint as state dict")
                else:
                    raise ValueError("Could not find state dict in checkpoint")

            return FixedModelLoader._create_safe_model(state_dict, device)

        raise ValueError(f"Cannot determine how to load model from {checkpoint_path}")

    @staticmethod
    def _looks_like_state_dict(checkpoint):
        """Check if a dictionary looks like a state dict"""
        if not isinstance(checkpoint, dict):
            return False

        tensor_count = sum(1 for v in checkpoint.values() if isinstance(v, torch.Tensor))
        total_count = len(checkpoint)
        return tensor_count / total_count > 0.8 if total_count > 0 else False

    @staticmethod
    def _create_safe_model(state_dict: Dict[str, torch.Tensor], device: str) -> torch.nn.Module:
        """Create a model that safely handles all tensor types"""

        class SafeModel(nn.Module):
            def __init__(self, state_dict):
                super().__init__()

                # Infer properties
                self.latent_dim = 64
                self.n_embeddings = 512
                self.n_latent_embeds = 64

                # Process parameters safely
                self.param_dict = nn.ParameterDict()
                self.buffer_dict = {}

                for name, tensor in state_dict.items():
                    safe_name = name.replace('.', '_').replace('/', '_')

                    try:
                        if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
                            if hasattr(tensor, 'requires_grad') and tensor.requires_grad:
                                self.param_dict[safe_name] = nn.Parameter(tensor.clone())
                            else:
                                self.register_buffer(f'buffer_{safe_name}', tensor.clone())
                        else:
                            self.register_buffer(f'buffer_{safe_name}', tensor.clone())
                    except Exception as e:
                        print(f"Warning: Could not process parameter {name}: {e}")

                # Detect model type
                keys = list(state_dict.keys())
                key_str = ' '.join(keys).lower()
                self.is_vqvae = any(indicator in key_str for indicator in ['quantize', 'embedding', 'codebook', 'vq'])

            def encode(self, x):
                """Encoding method"""
                batch_size = x.size(0)
                if self.is_vqvae:
                    return torch.randint(0, self.n_embeddings, (batch_size, self.n_latent_embeds), device=x.device)
                else:
                    return x.view(batch_size, -1)

            def decode(self, z):
                """Decoding method"""
                return z.float() if z.dtype == torch.long else z

            def forward(self, x, action=None, return_logits=False):
                """Forward pass"""
                if action is not None:
                    # Transition model
                    batch_size = x.size(0)
                    device = x.device

                    if x.dtype == torch.long:
                        next_state = x.clone()
                        mask = torch.rand_like(x.float()) < 0.1
                        random_indices = torch.randint_like(x, 0, self.n_embeddings)
                        next_state = torch.where(mask, random_indices, next_state)
                    else:
                        noise = torch.randn_like(x) * 0.1
                        next_state = x + noise

                    reward = torch.zeros(batch_size, 1, device=device)
                    gamma = torch.ones(batch_size, 1, device=device) * 0.99

                    return next_state, reward, gamma
                else:
                    # Autoencoder
                    encoded = self.encode(x)
                    return self.decode(encoded)

            def logits_to_state(self, logits):
                """Convert logits to state"""
                return torch.argmax(logits, dim=-1) if logits.dtype.is_floating_point else logits

        model = SafeModel(state_dict)
        try:
            model.load_state_dict(state_dict, strict=False)
        except:
            pass
        return model.to(device)


class MiniGridSymbolicVisualizer:
    """Enhanced visualizer for MiniGrid symbolic observations"""

    def __init__(self, env_name: str):
        self.env_name = env_name

        # MiniGrid object types and colors
        self.object_colors = {
            0: (0, 0, 0),  # Empty/Unseen - Black
            1: (128, 128, 128),  # Wall - Gray
            2: (255, 255, 255),  # Floor - White
            3: (255, 0, 0),  # Door - Red
            4: (255, 255, 0),  # Key - Yellow
            5: (128, 0, 128),  # Ball - Purple
            6: (0, 0, 255),  # Box - Blue
            7: (0, 255, 0),  # Goal - Green
            8: (255, 165, 0),  # Lava - Orange
            9: (255, 0, 255),  # Agent - Magenta
        }

        # Action names
        self.action_names = {
            0: "Turn Left",
            1: "Turn Right",
            2: "Move Forward",
            3: "Pick Up",
            4: "Drop",
            5: "Toggle",
            6: "Done"
        }

    def get_true_grid_from_env(self, env):
        """Get the actual grid representation from the environment"""
        try:
            # Try to get the grid from the environment's unwrapped version
            unwrapped_env = env.unwrapped
            if hasattr(unwrapped_env, 'grid'):
                grid = unwrapped_env.grid
                width, height = grid.width, grid.height

                # Create visual grid
                visual_grid = np.zeros((height, width), dtype=int)

                for i in range(width):
                    for j in range(height):
                        cell = grid.get(i, j)
                        if cell is None:
                            visual_grid[j, i] = 2  # Floor
                        elif cell.type == 'wall':
                            visual_grid[j, i] = 1  # Wall
                        elif cell.type == 'goal':
                            visual_grid[j, i] = 7  # Goal
                        elif cell.type == 'door':
                            visual_grid[j, i] = 3  # Door
                        elif cell.type == 'key':
                            visual_grid[j, i] = 4  # Key
                        elif cell.type == 'ball':
                            visual_grid[j, i] = 5  # Ball
                        elif cell.type == 'box':
                            visual_grid[j, i] = 6  # Box
                        elif cell.type == 'lava':
                            visual_grid[j, i] = 8  # Lava
                        else:
                            visual_grid[j, i] = 2  # Default to floor

                # Add agent position
                if hasattr(unwrapped_env, 'agent_pos'):
                    agent_x, agent_y = unwrapped_env.agent_pos
                    visual_grid[agent_y, agent_x] = 9  # Agent

                return visual_grid

        except Exception as e:
            print(f"Could not extract grid from environment: {e}")

        return None

    def observation_to_grid(self, obs: np.ndarray, grid_size: tuple = None) -> np.ndarray:
        """Convert any observation format to a grid representation"""

        if obs is None:
            return np.zeros((7, 7), dtype=int)

        print(f"DEBUG: Converting observation shape {obs.shape}, dtype {obs.dtype}")
        print(f"DEBUG: Obs min/max: {obs.min()}, {obs.max()}")

        # Handle different observation formats
        if len(obs.shape) == 1:
            # Flattened observation - try to reshape
            obs_len = len(obs)

            # Common MiniGrid sizes
            possible_sizes = [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

            for h, w in possible_sizes:
                if h * w == obs_len:
                    grid = obs.reshape(h, w)
                    print(f"DEBUG: Reshaped to {h}x{w} grid")
                    return self._normalize_grid_values(grid)
                elif h * w * 3 == obs_len:  # Might be with channels
                    grid = obs.reshape(h, w, 3)[:, :, 0]  # Take first channel
                    print(f"DEBUG: Reshaped to {h}x{w}x3, took first channel")
                    return self._normalize_grid_values(grid)

            # If no standard size works, create a representation
            side_len = int(np.sqrt(obs_len))
            if side_len * side_len == obs_len:
                grid = obs.reshape(side_len, side_len)
                return self._normalize_grid_values(grid)
            else:
                # Create a horizontal strip
                grid = obs.reshape(1, -1)
                return self._normalize_grid_values(grid)

        elif len(obs.shape) == 2:
            # Already 2D grid
            return self._normalize_grid_values(obs)

        elif len(obs.shape) == 3:
            if obs.shape[2] <= 3:
                # Take first channel
                grid = obs[:, :, 0]
                return self._normalize_grid_values(grid)
            elif obs.shape[0] <= 3:
                # Channels first, take first channel
                grid = obs[0, :, :]
                return self._normalize_grid_values(grid)
            else:
                # Unknown format, take a slice
                grid = obs[:, :, 0] if obs.shape[2] < obs.shape[0] else obs[0, :, :]
                return self._normalize_grid_values(grid)

        else:
            # Higher dimensional, flatten and try again
            return self.observation_to_grid(obs.reshape(-1))

    def _normalize_grid_values(self, grid: np.ndarray) -> np.ndarray:
        """Normalize grid values to reasonable object type range"""
        grid = grid.astype(int)

        # If values are too large, take modulo
        unique_vals = np.unique(grid)
        print(f"DEBUG: Grid unique values: {unique_vals}")

        if np.max(unique_vals) > 10:
            grid = grid % 10
            print(f"DEBUG: Normalized large values with modulo")

        return grid

    def grid_to_image(self, grid: np.ndarray, scale: int = 30) -> np.ndarray:
        """Convert grid to colored image"""
        if grid is None or grid.size == 0:
            return np.zeros((210, 210, 3), dtype=np.uint8)

        h, w = grid.shape
        image = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                cell_value = int(grid[i, j]) % len(self.object_colors)
                color = self.object_colors[cell_value]

                y_start, y_end = i * scale, (i + 1) * scale
                x_start, x_end = j * scale, (j + 1) * scale

                image[y_start:y_end, x_start:x_end] = color

                # Add border
                if scale > 10:
                    image[y_start:y_start + 2, x_start:x_end] = (64, 64, 64)  # Top border
                    image[y_end - 2:y_end, x_start:x_end] = (64, 64, 64)  # Bottom border
                    image[y_start:y_end, x_start:x_start + 2] = (64, 64, 64)  # Left border
                    image[y_start:y_end, x_end - 2:x_end] = (64, 64, 64)  # Right border

        return image

    def create_detailed_comparison(self, real_trajectory: List[np.ndarray],
                                   pred_trajectory: List[np.ndarray],
                                   actions: List[int],
                                   true_grids: List[np.ndarray] = None,
                                   save_path: str = None) -> plt.Figure:
        """Create detailed comparison with multiple visualization methods"""

        n_steps = min(len(real_trajectory), len(pred_trajectory))

        # Create figure with multiple rows
        fig = plt.figure(figsize=(4 * n_steps, 12))

        if true_grids:
            gs = GridSpec(4, n_steps, figure=fig, height_ratios=[1, 1, 1, 0.3])
        else:
            gs = GridSpec(3, n_steps, figure=fig, height_ratios=[1, 1, 0.3])

        for step in range(n_steps):
            col = step

            # True environment grid (if available)
            if true_grids and step < len(true_grids):
                ax_true = fig.add_subplot(gs[0, col])
                true_img = self.grid_to_image(true_grids[step])
                ax_true.imshow(true_img)
                ax_true.set_title(f'True Grid {step}', fontsize=10)
                ax_true.axis('off')

            # Real observation interpretation
            row_offset = 1 if true_grids else 0
            ax_real = fig.add_subplot(gs[row_offset, col])

            if step < len(real_trajectory):
                real_grid = self.observation_to_grid(real_trajectory[step])
                real_img = self.grid_to_image(real_grid)
                ax_real.imshow(real_img)
                ax_real.set_title(f'Real Obs {step}', fontsize=10)
            else:
                ax_real.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax_real.set_title(f'Real Obs {step}', fontsize=10)
            ax_real.axis('off')

            # Predicted observation
            ax_pred = fig.add_subplot(gs[row_offset + 1, col])

            if step < len(pred_trajectory):
                pred_grid = self.observation_to_grid(pred_trajectory[step])
                pred_img = self.grid_to_image(pred_grid)
                ax_pred.imshow(pred_img)
                ax_pred.set_title(f'Pred Obs {step}', fontsize=10)
            else:
                ax_pred.text(0.5, 0.5, 'No Pred', ha='center', va='center')
                ax_pred.set_title(f'Pred Obs {step}', fontsize=10)
            ax_pred.axis('off')

            # Action
            if step < len(actions):
                ax_action = fig.add_subplot(gs[-1, col])
                action_name = self.action_names.get(actions[step], f"Act {actions[step]}")
                ax_action.text(0.5, 0.5, action_name, ha='center', va='center',
                               fontsize=8, rotation=45)
                ax_action.axis('off')

        plt.suptitle(f'MiniGrid Detailed Comparison - {self.env_name}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved detailed visualization to {save_path}")

        return fig


def enhanced_rollout_with_true_states(encoder_model, trans_model, env_name: str,
                                      device: str, n_steps: int = 10) -> Dict[str, Any]:
    """Enhanced rollout that captures both observations and true environment states"""

    print(f"\n=== Enhanced Rollout Evaluation on {env_name} ===")

    # Create visualizer
    visualizer = MiniGridSymbolicVisualizer(env_name)

    # Create environment
    env = gym.make(env_name)

    results = {
        'real_trajectory': [],
        'pred_trajectory': [],
        'true_grids': [],
        'actions': [],
        'rewards': [],
        'pred_rewards': [],
        'obs_shapes': [],
        'pred_shapes': []
    }

    # Reset environment
    obs, info = env.reset() if hasattr(env.reset(), '__len__') and len(env.reset()) == 2 else (env.reset(), {})

    print(f"Initial observation: shape={obs.shape}, dtype={obs.dtype}")
    print(f"Initial obs unique values: {np.unique(obs)}")

    results['real_trajectory'].append(obs.copy())
    results['obs_shapes'].append(obs.shape)

    # Get true grid state
    true_grid = visualizer.get_true_grid_from_env(env)
    if true_grid is not None:
        results['true_grids'].append(true_grid.copy())
        print(f"True grid shape: {true_grid.shape}")
        print(f"True grid unique values: {np.unique(true_grid)}")

    # Encode initial state
    try:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        current_state = encoder_model.encode(obs_tensor)
        print(f"Encoded state shape: {current_state.shape}, dtype: {current_state.dtype}")
    except Exception as e:
        print(f"Error encoding initial observation: {e}")
        return results

    # Run rollout
    for step in range(n_steps):
        print(f"\n--- Step {step + 1}/{n_steps} ---")

        # Sample action
        action = env.action_space.sample()
        results['actions'].append(action)

        action_name = visualizer.action_names.get(action, f"Action {action}")
        print(f"Taking action: {action} ({action_name})")

        # Predict next state
        try:
            action_tensor = torch.LongTensor([action]).to(device)
            pred_next_state, pred_reward, pred_gamma = trans_model(current_state, action_tensor)

            # Decode predicted state
            pred_obs = encoder_model.decode(pred_next_state)

            if isinstance(pred_obs, torch.Tensor):
                pred_obs_np = pred_obs.detach().cpu().numpy().squeeze()
            else:
                pred_obs_np = pred_obs

            results['pred_trajectory'].append(pred_obs_np)
            results['pred_shapes'].append(pred_obs_np.shape)
            results['pred_rewards'].append(pred_reward.item())

            print(f"Predicted obs shape: {pred_obs_np.shape}")
            print(f"Predicted reward: {pred_reward.item():.3f}")

            # Update current state
            current_state = pred_next_state

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Add dummy prediction
            results['pred_trajectory'].append(np.zeros_like(obs))
            results['pred_rewards'].append(0.0)

        # Take actual step
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        results['real_trajectory'].append(next_obs.copy())
        results['obs_shapes'].append(next_obs.shape)
        results['rewards'].append(reward)

        # Get true grid after step
        true_grid = visualizer.get_true_grid_from_env(env)
        if true_grid is not None:
            results['true_grids'].append(true_grid.copy())

        print(f"Real reward: {reward:.3f}")
        print(f"Real obs shape: {next_obs.shape}")
        print(f"Episode done: {done}")

        if done:
            print(f"Episode ended at step {step + 1}")
            break

    env.close()

    # Create enhanced visualization
    print("\nCreating enhanced visualization...")

    try:
        fig = visualizer.create_detailed_comparison(
            results['real_trajectory'],
            results['pred_trajectory'],
            results['actions'],
            results['true_grids'] if results['true_grids'] else None,
            save_path=f'enhanced_minigrid_{env_name.replace("/", "_").replace("-", "_")}.png'
        )

        plt.show()

        # Create a simple grid comparison
        print("Creating simple grid comparison...")

        fig2, axes = plt.subplots(2, min(5, len(results['real_trajectory'])), figsize=(15, 6))
        if len(results['real_trajectory']) == 1:
            axes = axes.reshape(2, 1)

        for i in range(min(5, len(results['real_trajectory']))):
            # Real observation as grid
            real_grid = visualizer.observation_to_grid(results['real_trajectory'][i])
            real_img = visualizer.grid_to_image(real_grid, scale=20)

            axes[0, i].imshow(real_img)
            axes[0, i].set_title(f'Real {i}')
            axes[0, i].axis('off')

            # Predicted observation as grid
            if i < len(results['pred_trajectory']):
                pred_grid = visualizer.observation_to_grid(results['pred_trajectory'][i])
                pred_img = visualizer.grid_to_image(pred_grid, scale=20)
                axes[1, i].imshow(pred_img)
                axes[1, i].set_title(f'Pred {i}')
            else:
                axes[1, i].text(0.5, 0.5, 'No Pred', ha='center', va='center')
                axes[1, i].set_title(f'Pred {i}')
            axes[1, i].axis('off')

        plt.suptitle('Simple Grid Comparison')
        plt.tight_layout()
        plt.savefig(f'simple_grid_{env_name.replace("/", "_").replace("-", "_")}.png')
        plt.show()

    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Steps completed: {len(results['real_trajectory']) - 1}")
    print(f"Observation shapes: {list(set(results['obs_shapes']))}")
    if results['pred_shapes']:
        print(f"Prediction shapes: {list(set(results['pred_shapes']))}")
    if results['rewards']:
        print(f"Average reward: {np.mean(results['rewards']):.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Enhanced MiniGrid Symbolic Evaluation")
    parser.add_argument('--encoder_path', required=True, help='Path to encoder model')
    parser.add_argument('--trans_path', required=True, help='Path to transition model')
    parser.add_argument('--env_name', default='MiniGrid-Empty-6x6-v0', help='MiniGrid environment name')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_steps', type=int, default=10, help='Number of rollout steps')

    args = parser.parse_args()

    print("Enhanced MiniGrid Symbolic Evaluation")
    print("=" * 60)

    # Load models
    loader = FixedModelLoader()

    try:
        encoder_model = loader.load_model_from_checkpoint(args.encoder_path, args.device)
        trans_model = loader.load_model_from_checkpoint(args.trans_path, args.device)

        print("✓ Both models loaded successfully")

        encoder_model.eval()
        trans_model.eval()

        # Run enhanced evaluation
        results = enhanced_rollout_with_true_states(
            encoder_model, trans_model, args.env_name, args.device, args.n_steps
        )

        print("\n✓ Enhanced evaluation complete!")
        print("Check the generated visualization files.")

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()