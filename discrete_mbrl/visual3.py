# model_rollout_viewer.py
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import argparse

from env_helpers import make_env
from model_construction import construct_ae_model, construct_trans_model
from training_helpers import get_obs_target_size, batch_obs_resize



class ModelRolloutViewer:
    def __init__(self, args, encoder_model, trans_model, env, n_steps=30):
        self.args = args
        self.encoder_model = encoder_model
        self.trans_model = trans_model
        self.env = env
        self.n_steps = n_steps

        # Storage for rollout
        self.observations = []
        self.actions = []
        self.rewards = []
        self.real_observations = []
        self.imagined_observations = []

        # GUI elements
        self.fig = None
        self.axes = None
        self.current_step = 0
        self.is_playing = False
        self.show_imagined = True

    def collect_rollout(self):
        """Collect a rollout using the transition model"""
        print("Collecting rollout...")

        # Reset environment
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        self.observations = [obs]
        self.real_observations = [obs]
        self.actions = []
        self.rewards = []

        # Get initial latent state
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.args.device)
            z = self.encoder_model.encode(obs_tensor)

            # Reshape for continuous models
            if hasattr(self.encoder_model, 'latent_dim'):
                z = z.reshape(z.shape[0], self.encoder_model.latent_dim)

        # Collect imagined rollout
        self.imagined_observations = [obs]

        for step in range(self.n_steps):
            # Get action from random policy
            action = self.env.action_space.sample()
            self.actions.append(action)

            # Real environment step
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            self.real_observations.append(obs)
            self.rewards.append(reward)

            # Imagined step using transition model
            with torch.no_grad():
                action_tensor = torch.tensor([action]).to(self.args.device)
                z, reward_pred, gamma_pred = self.trans_model(z, action_tensor)

                # Decode to get imagined observation
                imagined_obs = self.encoder_model.decode(z).cpu().numpy()[0]
                self.imagined_observations.append(imagined_obs)

            if done:
                print(f"Episode ended at step {step + 1}")
                break

        print(f"Collected {len(self.real_observations)} observations")

    def setup_gui(self):
        """Setup the matplotlib GUI"""
        self.fig = plt.figure(figsize=(15, 8))

        # Create grid layout
        gs = self.fig.add_gridspec(3, 3, height_ratios=[5, 5, 1], width_ratios=[1, 1, 1])

        # Main display axes
        self.ax_real = self.fig.add_subplot(gs[0, :2])
        self.ax_imagined = self.fig.add_subplot(gs[1, :2])

        # Info panel
        self.ax_info = self.fig.add_subplot(gs[:2, 2])
        self.ax_info.axis('off')

        # Control panel
        self.ax_controls = self.fig.add_subplot(gs[2, :])

        # Setup images
        self.im_real = self.ax_real.imshow(self._process_obs(self.real_observations[0]))
        self.ax_real.set_title("Real Environment", fontsize=14)
        self.ax_real.axis('off')

        self.im_imagined = self.ax_imagined.imshow(self._process_obs(self.imagined_observations[0]))
        self.ax_imagined.set_title("Model Imagination", fontsize=14)
        self.ax_imagined.axis('off')

        # Setup controls
        self.setup_controls()

        # Update display
        self.update_display()

    def setup_controls(self):
        """Setup GUI controls"""
        # Step slider
        ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
        self.slider = Slider(ax_slider, 'Step', 0, len(self.real_observations) - 1,
                             valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        # Play/Pause button
        ax_play = plt.axes([0.75, 0.04, 0.08, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.on_play_clicked)

        # Reset button
        ax_reset = plt.axes([0.84, 0.04, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.on_reset_clicked)

        # New rollout button
        ax_new = plt.axes([0.05, 0.04, 0.12, 0.04])
        self.btn_new = Button(ax_new, 'New Rollout')
        self.btn_new.on_clicked(self.on_new_rollout_clicked)

    def _process_obs(self, obs):
        """Process observation for display"""
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Handle different formats
        if len(obs.shape) == 3:
            if obs.shape[0] <= 3:  # CHW format
                obs = obs.transpose(1, 2, 0)

        # Clip to valid range
        return obs.clip(0, 1)

    def update_display(self):
        """Update the display with current step"""
        step = int(self.current_step)

        # Update images
        self.im_real.set_data(self._process_obs(self.real_observations[step]))
        self.im_imagined.set_data(self._process_obs(self.imagined_observations[step]))

        # Update info panel
        self.ax_info.clear()
        self.ax_info.axis('off')

        info_text = f"Step: {step}/{len(self.real_observations) - 1}\n\n"

        if step > 0:
            info_text += f"Action: {self.actions[step - 1]}\n"
            info_text += f"Reward: {self.rewards[step - 1]:.3f}\n\n"

            # Calculate difference between real and imagined
            real_obs = self._process_obs(self.real_observations[step])
            imag_obs = self._process_obs(self.imagined_observations[step])

            if real_obs.shape == imag_obs.shape:
                mse = np.mean((real_obs - imag_obs) ** 2)
                info_text += f"MSE: {mse:.4f}\n"
            else:
                info_text += "MSE: N/A (shape mismatch)\n"

        info_text += f"\nEnvironment: {self.args.env_name}\n"
        info_text += f"Model: {self.args.trans_model_type}"

        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                          fontsize=12, verticalalignment='top', fontfamily='monospace')

        plt.draw()

    def on_slider_change(self, val):
        """Handle slider change"""
        self.current_step = int(val)
        self.update_display()

    def on_play_clicked(self, event):
        """Handle play/pause button"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.label.set_text('Pause')
            self.play_animation()
        else:
            self.btn_play.label.set_text('Play')

    def on_reset_clicked(self, event):
        """Handle reset button"""
        self.current_step = 0
        self.slider.set_val(0)
        self.update_display()

    def on_new_rollout_clicked(self, event):
        """Handle new rollout button"""
        self.collect_rollout()
        self.slider.valmax = len(self.real_observations) - 1
        self.slider.set_val(0)
        self.current_step = 0
        self.update_display()

    def play_animation(self):
        """Play animation"""
        if self.is_playing and self.current_step < len(self.real_observations) - 1:
            self.current_step += 1
            self.slider.set_val(self.current_step)
            self.update_display()
            self.fig.canvas.draw_idle()
            # Schedule next frame
            plt.pause(0.2)
            self.play_animation()
        else:
            self.is_playing = False
            self.btn_play.label.set_text('Play')

    def run(self):
        """Run the GUI"""
        self.collect_rollout()
        self.setup_gui()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-5x5-v0',
                        help='Environment name')
    parser.add_argument('--encoder_path', type=str,
                        default='./models/MiniGrid-Empty-5x5-v0/model_25cb239f0df83f36333fa6dc2e84c913.pt',
                        help='Path to encoder model')
    parser.add_argument('--trans_path', type=str,
                        default='./models/MiniGrid-Empty-5x5-v0/model_aa6d126ffa2897f2d6e76068df7ab6b4.pt',
                        help='Path to transition model')
    parser.add_argument('--n_steps', type=int, default=30,
                        help='Number of steps to roll out')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # Model configuration arguments
    parser.add_argument('--ae_model_type', type=str, default='discrete')
    parser.add_argument('--ae_model_version', type=int, default=0)
    parser.add_argument('--trans_model_type', type=str, default='discrete')
    parser.add_argument('--trans_model_version', type=int, default=0)
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--env_max_steps', type=int, default=None)

    # Add ALL missing arguments that model_construction expects
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--comet_ml', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--ae_model_hash', type=str, default='')
    parser.add_argument('--trans_model_hash', type=str, default='')
    parser.add_argument('--n_embeddings', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--load_checkpoint', type=str, default='')
    parser.add_argument('--checkpoint_dir', type=str, default='./model_checkpoints')
    parser.add_argument('--experiment_name', type=str, default='model_viewer')
    parser.add_argument('--run_id', type=str, default='')
    parser.add_argument('--tags', type=str, nargs='*', default=[])

    # Additional model-specific arguments
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--load_prob', type=float, default=0.5)
    parser.add_argument('--n_latent_embeds', type=int, default=32)
    parser.add_argument('--codebook_size', type=int, default=512)
    parser.add_argument('--commitment_cost', type=float, default=0.25)
    parser.add_argument('--vq_beta', type=float, default=0.25)

    args = parser.parse_args()

    # Load environment
    print(f"Creating environment: {args.env_name}")
    env = make_env(args.env_name, max_steps=args.env_max_steps)

    # Get sample observation for model construction
    sample_obs = env.reset()
    if isinstance(sample_obs, tuple):
        sample_obs = sample_obs[0]

    # First, inspect the checkpoints
    print(f"\nInspecting checkpoints...")
    encoder_checkpoint = torch.load(args.encoder_path, map_location=args.device)
    trans_checkpoint = torch.load(args.trans_path, map_location=args.device)

    # Check what's in the checkpoints
    if isinstance(encoder_checkpoint, dict):
        print(f"Encoder checkpoint keys: {list(encoder_checkpoint.keys())}")
        if 'args' in encoder_checkpoint:
            saved_args = encoder_checkpoint['args']
            # Update our args with the saved configuration
            for attr in dir(saved_args):
                if not attr.startswith('_'):
                    setattr(args, attr, getattr(saved_args, attr))
            print(f"Loaded configuration from encoder checkpoint")

    if isinstance(trans_checkpoint, dict):
        print(f"Trans checkpoint keys: {list(trans_checkpoint.keys())}")

    # Now construct models with the loaded configuration
    print("\nConstructing models...")
    encoder_model, _ = construct_ae_model(sample_obs.shape, args)
    encoder_model = encoder_model.to(args.device)

    trans_model, _ = construct_trans_model(encoder_model, args, env.action_space)
    trans_model = trans_model.to(args.device)

    # Load the weights
    print("\nLoading model weights...")
    if isinstance(encoder_checkpoint, dict):
        if 'model_state_dict' in encoder_checkpoint:
            encoder_model.load_state_dict(encoder_checkpoint['model_state_dict'])
        elif 'state_dict' in encoder_checkpoint:
            encoder_model.load_state_dict(encoder_checkpoint['state_dict'])
        else:
            print("Warning: Could not find state dict in encoder checkpoint")
    else:
        encoder_model.load_state_dict(encoder_checkpoint)

    if isinstance(trans_checkpoint, dict):
        if 'model_state_dict' in trans_checkpoint:
            trans_model.load_state_dict(trans_checkpoint['model_state_dict'])
        elif 'state_dict' in trans_checkpoint:
            trans_model.load_state_dict(trans_checkpoint['state_dict'])
        else:
            print("Warning: Could not find state dict in trans checkpoint")
    else:
        trans_model.load_state_dict(trans_checkpoint)

    encoder_model.eval()
    trans_model.eval()
    print("Models loaded successfully!")

    # Create and run viewer
    viewer = ModelRolloutViewer(args, encoder_model, trans_model, env, n_steps=args.n_steps)
    viewer.run()


if __name__ == "__main__":
    main()

