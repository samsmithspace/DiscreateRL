#!/usr/bin/env python3
"""
Visual Game Environment Viewer - Force real graphical display
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.distributions import Categorical

# Add your project paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from env_helpers import make_env, preprocess_obs
from model_construction import construct_ae_model
from shared.models import mlp


def setup_display():
    """Setup display environment variables for rendering"""
    import pygame

    # Initialize pygame
    pygame.init()
    pygame.display.init()

    # Set environment variables for better compatibility
    os.environ['SDL_VIDEODRIVER'] = 'windib' if os.name == 'nt' else 'x11'

    print("✓ Pygame initialized successfully")
    return True


def create_visual_env(env_name, render_mode='human'):
    """Create environment with explicit rendering mode"""
    print(f"🌍 Creating visual environment: {env_name}")

    # Try different ways to create environment with rendering
    try:
        # Method 1: Use your make_env function (most compatible)
        env = make_env(env_name)
        print("✓ Created environment with make_env")
        return env
    except Exception as e1:
        print(f"make_env failed: {e1}")

        try:
            # Method 2: Direct gym.make
            import gym
            # Extract the actual minigrid env name
            if 'crossing' in env_name:
                gym_name = 'MiniGrid-SimpleCrossingS9N1-v0'
            elif 'empty' in env_name:
                gym_name = 'MiniGrid-Empty-6x6-v0'
            else:
                gym_name = 'MiniGrid-Empty-6x6-v0'  # Default

            env = gym.make(gym_name)
            print(f"✓ Created environment: {gym_name}")
            return env
        except Exception as e2:
            print(f"Direct gym.make failed: {e2}")
            return None


def load_models_simple(model_path, env_name):
    """Simplified model loading"""
    print(f"📦 Loading model: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Get basic info
    if 'avg_reward' in checkpoint:
        print(f"  Model reward: {checkpoint['avg_reward']:.4f}")

    # Create environment to get dimensions
    temp_env = make_env(env_name)
    reset_result = temp_env.reset()
    if isinstance(reset_result, tuple):
        sample_obs, _ = reset_result
    else:
        sample_obs = reset_result
    sample_obs = preprocess_obs([sample_obs])
    act_dim = temp_env.action_space.n
    temp_env.close()

    # Get args with defaults
    if 'args' in checkpoint:
        args_dict = checkpoint['args']

        class Args:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        args = Args(**args_dict)
    else:
        class Args:
            def __init__(self):
                self.env_name = env_name
                self.ae_model_type = 'vae'
                self.rl_activation = 'relu'
                self.policy_hidden = [256, 256]
                self.critic_hidden = [256, 256]
                self.codebook_size = 512
                self.embedding_dim = 64
                self.load = True

        args = Args()

    # Ensure lists
    if isinstance(args.policy_hidden, str):
        args.policy_hidden = [int(x) for x in args.policy_hidden.split(',')]
    if isinstance(args.critic_hidden, str):
        args.critic_hidden = [int(x) for x in args.critic_hidden.split(',')]

    # Load autoencoder
    ae_model, _ = construct_ae_model(sample_obs.shape[1:], args, latent_activation=True, load=True)
    ae_model.load_state_dict(checkpoint['ae_model_state_dict'])
    ae_model.eval()

    # Load policy
    mlp_kwargs = {'activation': args.rl_activation, 'discrete_input': args.ae_model_type == 'vqvae'}
    if args.ae_model_type == 'vqvae':
        mlp_kwargs['n_embeds'] = args.codebook_size
        mlp_kwargs['embed_dim'] = args.embedding_dim
        input_dim = args.embedding_dim * ae_model.n_latent_embeds
    else:
        input_dim = ae_model.latent_dim

    policy = mlp([input_dim] + args.policy_hidden + [act_dim], **mlp_kwargs)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()

    print("✓ Models loaded successfully")
    return ae_model, policy


def force_render_environment(env):
    """Try multiple methods to force rendering"""
    methods = [
        lambda: env.render(),  # New API - no mode parameter
        lambda: env.unwrapped.render(),  # Try unwrapped
        lambda: getattr(env, 'window', None) and env.window.switch_to(),  # Direct window access
        lambda: env.render(mode='human') if hasattr(env.render,
                                                    '__code__') and 'mode' in env.render.__code__.co_varnames else env.render(),
        # Conditional mode
    ]

    for i, method in enumerate(methods):
        try:
            result = method()
            if i == 0:  # First method worked
                print(f"✓ Using new MiniGrid rendering API")
            return True
        except Exception as e:
            if i == 0:  # Only print detailed error for first attempt
                print(f"⚠️  Method {i + 1}: {e}")
            continue

    print("❌ All rendering methods failed")
    return False


def play_visual_episode(env, ae_model, policy, max_steps=1000, step_delay=0.2):
    """Play episode with visual rendering"""
    print("\n🎮 Starting visual episode...")
    print("🎯 Look for a pygame window that should open!")
    print("Press Ctrl+C to stop early")

    # Reset environment
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result

    obs_tensor = torch.from_numpy(obs).float()
    total_reward = 0
    step_count = 0

    # Initial render
    render_working = force_render_environment(env)
    if not render_working:
        print("❌ Could not initialize rendering - you may only see console output")
    else:
        print("✓ Rendering initialized - game window should be visible!")

    try:
        for step in range(max_steps):
            # Render current state
            if render_working:
                force_render_environment(env)

            # Get action from policy
            with torch.no_grad():
                state = ae_model.encode(obs_tensor.unsqueeze(0), return_one_hot=True)
                act_logits = policy(state)
                act_dist = Categorical(logits=act_logits)
                action = act_dist.sample().item()
                action_probs = torch.softmax(act_logits, dim=-1).squeeze()

            print(f"Step {step + 1:3d}: Action={action} (confidence={action_probs[action]:.1%})", end=" ")

            # Take action
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            obs_tensor = torch.from_numpy(next_obs).float()
            total_reward += reward
            step_count += 1

            print(f"Reward={reward:.3f}, Total={total_reward:.4f}")

            # Pause to see the action
            time.sleep(step_delay)

            if done:
                # Final render
                if render_working:
                    force_render_environment(env)

                print(f"\n🎯 Episode Complete!")
                print(f"   Final Reward: {total_reward:.4f}")
                print(f"   Steps Taken: {step_count}")

                # Keep window open briefly
                time.sleep(2.0)
                break

    except KeyboardInterrupt:
        print(f"\n⏹ Stopped by user after {step_count} steps")
        print(f"   Reward: {total_reward:.4f}")

    return total_reward, step_count


def test_rendering_setup():
    """Test if rendering can work at all"""
    print("🔍 Testing rendering setup...")

    try:
        import pygame
        print("✓ Pygame available")

        # Test basic pygame
        pygame.init()
        screen = pygame.display.set_mode((200, 200))
        pygame.display.set_caption("Test Window")
        screen.fill((0, 255, 0))  # Green
        pygame.display.flip()
        print("✓ Pygame window created successfully")
        time.sleep(1)
        pygame.quit()

        return True
    except Exception as e:
        print(f"❌ Rendering test failed: {e}")
        return False


def main():
    """Main function to run visual game viewer"""
    parser = argparse.ArgumentParser(description='Visual Game Environment Viewer')
    parser.add_argument('--env_name', type=str, default='minigrid-crossing-stochastic')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--speed', type=float, default=0.3, help='Delay between steps')
    parser.add_argument('--test_render', action='store_true', help='Test rendering first')

    args = parser.parse_args()

    print("🎮 Visual MiniGrid Game Viewer")
    print("=" * 50)

    # Test rendering if requested
    if args.test_render:
        if not test_rendering_setup():
            print("❌ Rendering test failed - visual mode may not work")
            return
        print("✓ Rendering test passed!")

    # Setup display
    try:
        setup_display()
    except Exception as e:
        print(f"⚠️  Display setup warning: {e}")

    # Find model
    if args.model_path:
        model_path = args.model_path
    else:
        import glob
        pattern = f"./model_free/models/minigrid-crossing-stochastic/final_model_reward_0.9980.pt"
        files = glob.glob(pattern)
        if not files:
            print(f"❌ No models found matching: {pattern}")
            return
        model_path = files[0]
        print(f"✓ Auto-found model: {model_path}")

    # Load models
    try:
        ae_model, policy = load_models_simple(model_path, args.env_name)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    # Create visual environment
    env = create_visual_env(args.env_name)
    if env is None:
        print("❌ Failed to create environment")
        return

    print(f"✓ Environment created: {type(env).__name__}")

    try:
        # Play episodes
        for episode in range(args.episodes):
            if args.episodes > 1:
                print(f"\n🎬 Episode {episode + 1}/{args.episodes}")

            total_reward, steps = play_visual_episode(env, ae_model, policy, step_delay=args.speed)

            if args.episodes > 1 and episode < args.episodes - 1:
                input("Press Enter for next episode...")

    except Exception as e:
        print(f"❌ Error during gameplay: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            env.close()
        except:
            pass

        # Cleanup pygame
        try:
            import pygame
            pygame.quit()
        except:
            pass

        print("\n✅ Done!")


if __name__ == "__main__":
    main()