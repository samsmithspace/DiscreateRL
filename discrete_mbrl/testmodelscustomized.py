#!/usr/bin/env python3
"""
Simplified MiniGrid Model GUI Test - bypasses complex argument processing
"""

import sys
import os
import argparse
from argparse import Namespace
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Add the parent directory to path to import discrete_mbrl modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_helpers import make_env, check_env_name
from model_construction import construct_ae_model, construct_trans_model


def create_simple_args(env_name, ae_model_type, latent_dim, device='cpu', **kwargs):
    """Create a simple args object without complex processing."""
    args = Namespace()

    # Basic settings
    args.env_name = check_env_name(env_name)
    args.ae_model_type = ae_model_type
    args.trans_model_type = kwargs.get('trans_model_type', 'continuous')
    args.device = device
    args.load = True

    # Model architecture - USE EXACT SAME DEFAULTS AS TRAINING
    args.latent_dim = latent_dim
    args.embedding_dim = kwargs.get('embedding_dim', 64)  # Default from training_helpers.py
    args.filter_size = kwargs.get('filter_size', 8)  # Default from training_helpers.py
    args.codebook_size = kwargs.get('codebook_size', 16)  # Default from training_helpers.py
    args.ae_model_version = kwargs.get('ae_model_version', '2')  # Default from training_helpers.py
    args.trans_model_version = kwargs.get('trans_model_version', '1')  # Default from training_helpers.py
    args.trans_hidden = kwargs.get('trans_hidden', 256)  # Default from training_helpers.py
    args.trans_depth = kwargs.get('trans_depth', 3)  # Default from training_helpers.py
    args.stochastic = kwargs.get('stochastic', 'simple')  # Default from training_helpers.py

    # Hash-relevant parameters that must match training
    args.extra_info = kwargs.get('extra_info', None)  # Important for hash
    args.repr_sparsity = kwargs.get('repr_sparsity', 0)  # Important for hash
    args.sparsity_type = kwargs.get('sparsity_type', 'random')  # Important for hash
    args.vq_trans_1d_conv = kwargs.get('vq_trans_1d_conv', False)  # Important for hash

    # Disable logging
    args.wandb = False
    args.comet_ml = False

    return args


class SimpleModelGUI:
    def __init__(self, env_name, ae_model_type, latent_dim, device='cpu', **kwargs):
        print(f"Initializing GUI with: {env_name}, {ae_model_type}, latent_dim={latent_dim}")

        self.args = create_simple_args(env_name, ae_model_type, latent_dim, device, **kwargs)

        # Debug: Print all parameters that affect model hash
        print("\n=== Model Parameters (for hash generation) ===")
        print(f"env_name: {self.args.env_name}")
        print(f"ae_model_type: {self.args.ae_model_type}")
        print(f"trans_model_type: {self.args.trans_model_type}")
        print(f"latent_dim: {self.args.latent_dim}")
        print(f"embedding_dim: {self.args.embedding_dim}")
        print(f"filter_size: {self.args.filter_size}")
        print(f"codebook_size: {self.args.codebook_size}")
        print(f"ae_model_version: {self.args.ae_model_version}")
        print(f"trans_model_version: {self.args.trans_model_version}")
        print(f"trans_hidden: {self.args.trans_hidden}")
        print(f"trans_depth: {self.args.trans_depth}")
        print(f"stochastic: {self.args.stochastic}")
        print(f"extra_info: {self.args.extra_info}")
        print(f"repr_sparsity: {self.args.repr_sparsity}")
        print(f"sparsity_type: {self.args.sparsity_type}")
        print(f"vq_trans_1d_conv: {self.args.vq_trans_1d_conv}")
        print("=" * 50)

        self.setup_models()
        self.setup_environments()
        self.setup_gui()

        # State tracking
        self.real_obs = None
        self.model_obs = None
        self.model_state = None
        self.step_count = 0

        # Initialize environments
        self.reset_environments()

    def setup_models(self):
        """Load the trained encoder and transition models."""
        print("\n=== Loading Models ===")

        try:
            # Create a dummy observation to get the shape
            temp_env = make_env(self.args.env_name)
            temp_obs = temp_env.reset()
            if isinstance(temp_obs, tuple):
                temp_obs = temp_obs[0]
            temp_env.close()

            print(f"Observation shape: {temp_obs.shape}")

            # Load encoder model with hash debugging
            print(f"\n--- Loading Encoder ---")
            print(f"Model type: {self.args.ae_model_type}, latent_dim={self.args.latent_dim}")

            # Import the hash function to debug
            from model_construction import make_model_hash, AE_MODEL_VARS, MODEL_VARS

            # Generate and display encoder hash
            encoder_hash = make_model_hash(self.args, model_vars=AE_MODEL_VARS, exp_type='encoder')
            print(f"Encoder hash: {encoder_hash}")

            # Check if encoder file exists
            encoder_path = f'./models/{self.args.env_name}/model_{encoder_hash}.pt'
            encoder_path = encoder_path.replace(':', '-')
            print(f"Looking for encoder at: {encoder_path}")
            print(f"Encoder file exists: {os.path.exists(encoder_path)}")

            self.encoder_model = construct_ae_model(temp_obs.shape, self.args)[0]
            self.encoder_model = self.encoder_model.to(self.args.device)
            self.encoder_model.eval()

            # Load transition model with hash debugging
            print(f"\n--- Loading Transition Model ---")
            temp_env = make_env(self.args.env_name)

            # Generate and display transition hash
            trans_hash = make_model_hash(self.args, model_vars=MODEL_VARS, exp_type='trans_model')
            print(f"Transition hash: {trans_hash}")

            # Check if transition file exists
            trans_path = f'./models/{self.args.env_name}/model_{trans_hash}.pt'
            trans_path = trans_path.replace(':', '-')
            print(f"Looking for transition model at: {trans_path}")
            print(f"Transition file exists: {os.path.exists(trans_path)}")

            self.trans_model = construct_trans_model(
                self.encoder_model, self.args, temp_env.action_space)[0]
            self.trans_model = self.trans_model.to(self.args.device)
            self.trans_model.eval()
            temp_env.close()

            # List available model files for comparison
            models_dir = f'./models/{self.args.env_name}'
            models_dir = models_dir.replace(':', '-')
            if os.path.exists(models_dir):
                print(f"\n--- Available model files in {models_dir} ---")
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                for f in model_files:
                    print(f"  {f}")
                if not model_files:
                    print("  No .pt model files found!")
            else:
                print(f"\n--- Models directory does not exist: {models_dir} ---")

            print("✓ Model loading completed!")

        except Exception as e:
            print(f"✗ Error loading models: {e}")
            print("Make sure you have trained models with the same parameters!")
            import traceback
            traceback.print_exc()
            raise

    def setup_environments(self):
        """Setup real environment."""
        self.real_env = make_env(self.args.env_name)
        print(f"✓ Environment created: {self.args.env_name}")

    def setup_gui(self):
        """Create the GUI interface."""
        self.root = tk.Tk()
        self.root.title(f"MiniGrid: Real vs Model ({self.args.env_name})")
        self.root.geometry("700x500")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)

        # Title
        title = ttk.Label(main_frame, text="MiniGrid: Real vs Model Prediction",
                          font=("Arial", 14, "bold"))
        title.pack(pady=(0, 10))

        # Images frame
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(pady=10)

        # Real environment
        real_frame = ttk.LabelFrame(images_frame, text="Real Environment", padding="5")
        real_frame.pack(side='left', padx=(0, 10))

        self.real_canvas = tk.Canvas(real_frame, width=200, height=200, bg="white")
        self.real_canvas.pack()

        # Model prediction
        model_frame = ttk.LabelFrame(images_frame, text="Model Prediction", padding="5")
        model_frame.pack(side='right', padx=(10, 0))

        self.model_canvas = tk.Canvas(model_frame, width=200, height=200, bg="white")
        self.model_canvas.pack()

        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(pady=10)

        self.step_label = ttk.Label(info_frame, text="Steps: 0", font=("Arial", 12))
        self.step_label.pack()

        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.pack(pady=10, fill='x')

        controls_text = (
            "Arrow Keys: ↑=Forward, ←=Turn Left, →=Turn Right, ↓=Stay\n"
            "R: Reset  |  Q: Quit"
        )
        ttk.Label(controls_frame, text=controls_text, justify='center').pack()

        # Action mapping for MiniGrid
        self.action_map = {
            'Up': 2,  # Move forward
            'Left': 0,  # Turn left
            'Right': 1,  # Turn right
            'Down': 6  # Done/Stay
        }

        # Bind keyboard events
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.focus_set()

    def preprocess_obs(self, obs):
        """Preprocess observation for model input."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)  # Add batch dimension
        return obs

    def obs_to_display_image(self, obs, size=(200, 200)):
        """Convert observation to displayable PIL Image."""
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Handle different observation formats
        if len(obs.shape) == 4:  # Batch dimension
            obs = obs[0]
        if len(obs.shape) == 3 and obs.shape[0] <= 3:  # CHW format
            obs = obs.transpose(1, 2, 0)  # Convert to HWC

        # Ensure values are in [0, 1] range
        if obs.max() > 1.0:
            obs = obs / 255.0
        obs = np.clip(obs, 0, 1)

        # Convert to PIL Image
        if len(obs.shape) == 2:  # Grayscale
            obs = np.stack([obs] * 3, axis=-1)  # Convert to RGB

        img = Image.fromarray((obs * 255).astype(np.uint8))
        img = img.resize(size, Image.NEAREST)  # Use nearest neighbor for pixel art
        return img

    def reset_environments(self):
        """Reset both real and model environments."""
        # Reset real environment
        reset_result = self.real_env.reset()
        if isinstance(reset_result, tuple):
            self.real_obs, _ = reset_result
        else:
            self.real_obs = reset_result

        # Reset model state to match real environment
        self.sync_model_with_real()

        self.step_count = 0
        self.update_display()
        print("Environments reset!")

    def sync_model_with_real(self):
        """Sync model state with real environment state."""
        with torch.no_grad():
            obs_tensor = self.preprocess_obs(self.real_obs).to(self.args.device)

            # Encode real observation to get model state
            self.model_state = self.encoder_model.encode(obs_tensor)
            if self.args.trans_model_type == 'continuous':
                self.model_state = self.model_state.reshape(self.model_state.shape[0], -1)

            # Decode to get model observation
            model_obs_tensor = self.encoder_model.decode(self.model_state)
            self.model_obs = model_obs_tensor.cpu().numpy()[0]

    def step_real_environment(self, action):
        """Take a step in the real environment."""
        step_result = self.real_env.step(action)

        if len(step_result) == 4:
            self.real_obs, reward, done, info = step_result
        else:
            self.real_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        return reward, done, info

    def step_model_prediction(self, action):
        """Get model's prediction for the next state."""
        if self.model_state is None:
            return

        with torch.no_grad():
            action_tensor = torch.tensor([action], dtype=torch.long).to(self.args.device)

            # Predict next state using transition model
            next_state_pred = self.trans_model(self.model_state, action_tensor)[0]

            # Decode predicted state to observation
            next_obs_pred = self.encoder_model.decode(next_state_pred)

            # Update model state and observation
            self.model_state = next_state_pred
            self.model_obs = next_obs_pred.cpu().numpy()[0]

    def update_display(self):
        """Update the GUI display with current observations."""
        try:
            # Update real environment image
            if self.real_obs is not None:
                real_img = self.obs_to_display_image(self.real_obs)
                self.real_photo = ImageTk.PhotoImage(real_img)
                self.real_canvas.delete("all")
                self.real_canvas.create_image(100, 100, image=self.real_photo)

            # Update model prediction image
            if self.model_obs is not None:
                model_img = self.obs_to_display_image(self.model_obs)
                self.model_photo = ImageTk.PhotoImage(model_img)
                self.model_canvas.delete("all")
                self.model_canvas.create_image(100, 100, image=self.model_photo)

            # Update step counter
            self.step_label.config(text=f"Steps: {self.step_count}")

        except Exception as e:
            print(f"Error updating display: {e}")

    def on_key_press(self, event):
        """Handle keyboard input."""
        key = event.keysym

        if key == 'q' or key == 'Q':
            self.quit_app()
        elif key == 'r' or key == 'R':
            self.reset_environments()
        elif key in self.action_map:
            self.take_action(self.action_map[key])

    def take_action(self, action):
        """Execute an action in both environments."""
        try:
            # Step real environment
            reward, done, info = self.step_real_environment(action)

            # Step model prediction
            self.step_model_prediction(action)

            self.step_count += 1
            print(f"Action {action}, Step {self.step_count}, Reward: {reward:.2f}")

            # Update display
            self.update_display()

            # Check if episode is done
            if done:
                print(f"Episode finished! Steps taken: {self.step_count}")
                self.root.after(2000, self.reset_environments)

        except Exception as e:
            print(f"Error taking action: {e}")

    def quit_app(self):
        """Clean up and quit the application."""
        try:
            self.real_env.close()
        except:
            pass
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the GUI application."""
        print("🎮 Starting GUI...")
        print("Use arrow keys to control the agent!")
        print("Press 'R' to reset, 'Q' to quit")

        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nQuitting...")
        finally:
            self.quit_app()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple MiniGrid Model GUI")
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-6x6-v0')
    parser.add_argument('--ae_model_type', type=str, default='ae')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    print(f"🚀 Launching GUI:")
    print(f"  Environment: {args.env_name}")
    print(f"  Model: {args.ae_model_type}")
    print(f"  Latent Dim: {args.latent_dim}")
    print(f"  Device: {args.device}")

    try:
        app = SimpleModelGUI(args.env_name, args.ae_model_type, args.latent_dim, args.device)
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()