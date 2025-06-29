#!/usr/bin/env python3
"""
MiniGrid Viewer - Fixed display version
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
    """Setup pygame display with error handling"""
    try:
        # Disable PyCharm display interference
        os.environ['PYCHARM_DISPLAY_PORT'] = ''
        os.environ['PYCHARM_MATPLOTLIB_BACKEND'] = ''
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
        os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'

        if os.name == 'nt':
            os.environ['SDL_VIDEODRIVER'] = 'windib'
        else:
            os.environ['SDL_VIDEODRIVER'] = 'x11'
            if 'DISPLAY' not in os.environ:
                os.environ['DISPLAY'] = ':0'

        import pygame
        pygame.init()
        pygame.display.init()

        # Test if display works
        try:
            test_surface = pygame.display.set_mode((1, 1))
            pygame.display.flip()
            pygame.display.quit()
            pygame.display.init()
        except Exception as e:
            print(f"⚠️ Display test failed: {e}")
            return False

        print("✓ Display setup completed")
        return True

    except Exception as e:
        print(f"⚠️ Display setup failed: {e}")
        return False


def create_game_window():
    """Create our main game display window"""
    import pygame
    screen = pygame.display.set_mode((900, 700))
    pygame.display.set_caption("🎮 MiniGrid Game Viewer - Live Action!")
    return screen


def get_minigrid_rendering(env):
    """Get the MiniGrid rendering as a numpy array"""
    try:
        # Method 1: Try to get rgb_array mode
        if hasattr(env.unwrapped, 'render'):
            try:
                rgb_array = env.unwrapped.render(mode='rgb_array')
                if rgb_array is not None:
                    return rgb_array
            except:
                pass

            try:
                original_mode = getattr(env.unwrapped, 'render_mode', None)
                env.unwrapped.render_mode = 'rgb_array'
                rgb_array = env.unwrapped.render()
                env.unwrapped.render_mode = original_mode
                if rgb_array is not None:
                    return rgb_array
            except:
                pass

        # Method 2: Generate our own rendering
        return generate_simple_grid_view(env)

    except Exception as e:
        print(f"⚠️ Rendering capture failed: {e}")
        return generate_fallback_view()


def generate_simple_grid_view(env):
    """Generate a simple visualization of the grid state"""
    try:
        if hasattr(env.unwrapped, 'grid'):
            grid = env.unwrapped.grid
            agent_pos = getattr(env.unwrapped, 'agent_pos', (0, 0))
            agent_dir = getattr(env.unwrapped, 'agent_dir', 0)

            height, width = grid.height, grid.width
            cell_size = 40
            img = np.ones((height * cell_size, width * cell_size, 3), dtype=np.uint8) * 255

            colors = {
                'wall': [100, 100, 100],
                'floor': [240, 240, 240],
                'goal': [0, 255, 0],
                'lava': [255, 0, 0],
                'door': [0, 0, 255],
                'key': [255, 255, 0],
            }

            # Draw grid
            for y in range(height):
                for x in range(width):
                    cell = grid.get(x, y)
                    if cell is not None:
                        obj_type = cell.type if hasattr(cell, 'type') else 'floor'
                        color = colors.get(obj_type, [200, 200, 200])
                    else:
                        color = colors['floor']

                    y1, y2 = y * cell_size, (y + 1) * cell_size
                    x1, x2 = x * cell_size, (x + 1) * cell_size
                    img[y1:y2, x1:x2] = color

            # Draw agent
            if agent_pos:
                ax, ay = agent_pos
                if 0 <= ax < width and 0 <= ay < height:
                    y1, y2 = ay * cell_size + 5, (ay + 1) * cell_size - 5
                    x1, x2 = ax * cell_size + 5, (ax + 1) * cell_size - 5
                    img[y1:y2, x1:x2] = [255, 0, 255]  # Magenta agent

                    # Draw direction indicator
                    center_y, center_x = ay * cell_size + cell_size // 2, ax * cell_size + cell_size // 2
                    directions = [(0, -10), (10, 0), (0, 10), (-10, 0)]
                    if 0 <= agent_dir < 4:
                        dx, dy = directions[agent_dir]
                        end_x, end_y = center_x + dx, center_y + dy
                        if 0 <= end_y < img.shape[0] and 0 <= end_x < img.shape[1]:
                            img[end_y - 2:end_y + 2, end_x - 2:end_x + 2] = [0, 0, 0]

            return img

    except Exception as e:
        print(f"⚠️ Simple grid generation failed: {e}")
        pass

    return generate_fallback_view()


def generate_fallback_view():
    """Generate a fallback view when all else fails"""
    return np.full((200, 200, 3), [100, 150, 200], dtype=np.uint8)


def draw_game_display(screen, env, step_info, reward_info):
    """Draw the complete game display"""
    import pygame

    screen.fill((30, 30, 30))

    # Get MiniGrid rendering
    game_img = get_minigrid_rendering(env)

    if game_img is not None:
        try:
            if game_img.dtype != np.uint8:
                game_img = (game_img * 255).astype(np.uint8)

            game_surface = pygame.surfarray.make_surface(game_img.swapaxes(0, 1))
            target_size = (400, 400)
            game_surface = pygame.transform.scale(game_surface, target_size)
            screen.blit(game_surface, (50, 50))

        except Exception as e:
            print(f"⚠️ Surface conversion failed: {e}")
            pygame.draw.rect(screen, (100, 100, 200), (50, 50, 400, 400))
            pygame.draw.circle(screen, (255, 255, 255), (250, 250), 50)
    else:
        pygame.draw.rect(screen, (100, 100, 200), (50, 50, 400, 400))
        pygame.draw.circle(screen, (255, 255, 255), (250, 250), 50)

    # Draw text information
    try:
        font_large = pygame.font.Font(None, 48)
        font_medium = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 24)

        title = font_large.render("🎮 MiniGrid Live Game", True, (255, 255, 255))
        screen.blit(title, (500, 50))

        if step_info:
            step_text = font_medium.render(step_info, True, (255, 255, 255))
            screen.blit(step_text, (500, 120))

        if reward_info:
            reward_text = font_medium.render(reward_info, True, (255, 255, 255))
            screen.blit(reward_text, (500, 170))

        instructions = [
            "🎯 Game View (Left Side)",
            "📊 Live game state captured from MiniGrid",
            "🤖 AI is playing automatically",
            "⌨️ Press ESC or close window to stop",
            "",
            "🔍 What you're seeing:",
            "• Gray = Walls",
            "• Light Gray = Floor",
            "• Green = Goal",
            "• Red = Lava/Danger",
            "• Magenta = Agent (AI player)",
            "• Black dot = Agent direction"
        ]

        y_offset = 220
        for instruction in instructions:
            if instruction:
                text = font_small.render(instruction, True, (200, 200, 200))
                screen.blit(text, (500, y_offset))
            y_offset += 25

    except Exception as e:
        print(f"⚠️ Text rendering failed: {e}")
        pygame.draw.rect(screen, (255, 255, 255), (500, 50, 300, 30))

    pygame.display.flip()


def play_episode_with_display(env, screen, ae_model, policy, max_steps=30, step_delay=1.0):
    """Play episode with visual display"""
    print("\n🎮 Starting episode with display...")

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result

    obs_tensor = torch.from_numpy(obs).float()
    total_reward = 0
    step_count = 0

    draw_game_display(screen, env, "Starting game...", "Reward: 0.0000")
    time.sleep(2)

    try:
        for step in range(max_steps):
            step_info = f"Step {step + 1}/{max_steps}"
            print(f"\n--- {step_info} ---")

            # Get action from policy
            with torch.no_grad():
                state = ae_model.encode(obs_tensor.unsqueeze(0), return_one_hot=True)
                act_logits = policy(state)
                act_dist = Categorical(logits=act_logits)
                action = act_dist.sample().item()
                action_probs = torch.softmax(act_logits, dim=-1).squeeze()

            action_names = ["Turn Left", "Turn Right", "Move Forward", "Pick Up", "Drop", "Toggle", "Done"]
            action_name = action_names[action] if action < len(action_names) else f"Action {action}"

            print(f"🎯 Action: {action_name} ({action}) - Confidence: {action_probs[action]:.1%}")

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

            print(f"📊 Reward: {reward:.3f}, Total: {total_reward:.4f}, Done: {done}")

            # Update display
            step_display = f"Step {step + 1}: {action_name}"
            reward_display = f"Reward: {reward:.3f} | Total: {total_reward:.4f}"
            draw_game_display(screen, env, step_display, reward_display)

            # Handle events and pause
            import pygame
            start_time = time.time()
            while time.time() - start_time < step_delay:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        print("\n⏹ Stopped by user")
                        return total_reward, step_count
                time.sleep(0.05)

            if done:
                final_step = f"COMPLETE! Final Score: {total_reward:.4f}"
                final_reward = f"Episode finished in {step_count} steps"
                draw_game_display(screen, env, final_step, final_reward)

                print(f"\n🎯 Episode Complete!")
                print(f"   Final Reward: {total_reward:.4f}")
                print(f"   Steps Taken: {step_count}")

                print("   Keeping final result visible for 5 seconds...")
                time.sleep(5)
                break

    except KeyboardInterrupt:
        print(f"\n⏹ Stopped by user after {step_count} steps")

    return total_reward, step_count


def play_text_episode(env, ae_model, policy, max_steps=30, step_delay=1.0):
    """Play episode with text output only"""
    print("\n🎮 Starting text-only episode...")

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result

    obs_tensor = torch.from_numpy(obs).float()
    total_reward = 0
    step_count = 0

    try:
        for step in range(max_steps):
            step_info = f"Step {step + 1}/{max_steps}"
            print(f"\n--- {step_info} ---")

            with torch.no_grad():
                state = ae_model.encode(obs_tensor.unsqueeze(0), return_one_hot=True)
                act_logits = policy(state)
                act_dist = Categorical(logits=act_logits)
                action = act_dist.sample().item()
                action_probs = torch.softmax(act_logits, dim=-1).squeeze()

            action_names = ["Turn Left", "Turn Right", "Move Forward", "Pick Up", "Drop", "Toggle", "Done"]
            action_name = action_names[action] if action < len(action_names) else f"Action {action}"

            print(f"🎯 Action: {action_name} ({action}) - Confidence: {action_probs[action]:.1%}")

            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            obs_tensor = torch.from_numpy(next_obs).float()
            total_reward += reward
            step_count += 1

            print(f"📊 Reward: {reward:.3f}, Total: {total_reward:.4f}, Done: {done}")

            if hasattr(env.unwrapped, 'agent_pos'):
                agent_pos = env.unwrapped.agent_pos
                agent_dir = getattr(env.unwrapped, 'agent_dir', 0)
                directions = ["North", "East", "South", "West"]
                dir_name = directions[agent_dir] if 0 <= agent_dir < 4 else f"Dir {agent_dir}"
                print(f"📍 Agent Position: {agent_pos}, Facing: {dir_name}")

            if step_delay > 0:
                time.sleep(step_delay)

            if done:
                print(f"\n🎯 Episode Complete!")
                print(f"   Final Reward: {total_reward:.4f}")
                print(f"   Steps Taken: {step_count}")
                break

    except KeyboardInterrupt:
        print(f"\n⏹ Stopped by user after {step_count} steps")

    return total_reward, step_count


def load_models_simple(model_path, env_name):
    """Load models"""
    print(f"📦 Loading model: {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if 'avg_reward' in checkpoint:
        print(f"  Model reward: {checkpoint['avg_reward']:.4f}")

    temp_env = make_env(env_name)
    reset_result = temp_env.reset()
    if isinstance(reset_result, tuple):
        sample_obs, _ = reset_result
    else:
        sample_obs = reset_result
    sample_obs = preprocess_obs([sample_obs])
    act_dim = temp_env.action_space.n
    temp_env.close()

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

    if isinstance(args.policy_hidden, str):
        args.policy_hidden = [int(x) for x in args.policy_hidden.split(',')]
    if isinstance(args.critic_hidden, str):
        args.critic_hidden = [int(x) for x in args.critic_hidden.split(',')]

    ae_model, _ = construct_ae_model(sample_obs.shape[1:], args, latent_activation=True, load=True)
    ae_model.load_state_dict(checkpoint['ae_model_state_dict'])
    ae_model.eval()

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


def main():
    """Main function with both display and text options"""
    parser = argparse.ArgumentParser(description='MiniGrid Viewer')
    parser.add_argument('--env_name', type=str, default='minigrid-crossing-stochastic')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--speed', type=float, default=1.0, help='Delay between steps')
    parser.add_argument('--steps', type=int, default=25, help='Max steps to run')
    parser.add_argument('--no_display', action='store_true', help='Run in text-only mode')

    args = parser.parse_args()

    print("🎮 MiniGrid Viewer")
    print("=" * 50)

    # Find model
    if args.model_path:
        model_path = args.model_path
    else:
        import glob
        pattern = f"./model_free/models/{args.env_name}/final_model_*.pt"
        files = glob.glob(pattern)
        if not files:
            pattern = f"./model_free/models/*/final_model_*.pt"
            files = glob.glob(pattern)

        if not files:
            print(f"❌ No models found")
            return

        model_path = files[0]

    # Load models
    try:
        ae_model, policy = load_models_simple(model_path, args.env_name)
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return

    # Create environment
    try:
        env = make_env(args.env_name)
        print("✓ Environment created")

        # Decide on display mode
        use_display = not args.no_display
        screen = None

        if use_display:
            print("🖥️ Attempting to setup display...")
            if setup_display():
                try:
                    screen = create_game_window()
                    print("✓ Display window created")
                except Exception as e:
                    print(f"⚠️ Window creation failed: {e}")
                    print("   Falling back to text mode...")
                    use_display = False
            else:
                print("   Falling back to text mode...")
                use_display = False

        if use_display and screen:
            print("\n🎯 Starting game with visual display...")
            total_reward, steps = play_episode_with_display(
                env, screen, ae_model, policy,
                max_steps=args.steps,
                step_delay=args.speed
            )
        else:
            print("\n🎯 Starting game in text mode...")
            total_reward, steps = play_text_episode(
                env, ae_model, policy,
                max_steps=args.steps,
                step_delay=args.speed
            )

        print(f"\n📊 Final Results:")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Steps Taken: {steps}")

    except Exception as e:
        print(f"❌ Error during gameplay: {e}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            env.close()
        except:
            pass

        try:
            import pygame
            pygame.quit()
        except:
            pass

        print("\n✅ Game completed!")


if __name__ == "__main__":
    main()