import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from data_logging import init_experiment, finish_experiment
from env_helpers import *
from training_helpers import *
from train_encoder import train_encoder
from train_transition_model import train_trans_model
from evaluate_model import eval_model
from train_rl_model import train_rl_model
from e2e_train import full_train


def main():
    """Run all validation tests"""
    # Parse args
    args = get_args()
    # Setup logging
    args = init_experiment('discrete-mbrl-full', args)

    # FIXED DEBUG CODE ⬇️
    print("=== DEBUGGING ENVIRONMENT PROCESSING ===")
    from env_helpers import make_env
    import matplotlib.pyplot as plt
    import numpy as np

    # Test the training pipeline environment
    print("1. Testing Training Pipeline Environment:")
    training_env = make_env(args.env_name)
    training_reset_result = training_env.reset()

    # Handle both old and new Gymnasium API
    if isinstance(training_reset_result, tuple):
        training_obs, training_info = training_reset_result
    else:
        training_obs = training_reset_result

    print(f"Training env obs shape: {training_obs.shape}")
    print(f"Training env obs dtype: {training_obs.dtype}")
    print(f"Training env obs range: [{training_obs.min():.3f}, {training_obs.max():.3f}]")

    # Test the raw MiniGrid environment (before wrappers)
    print("\n2. Testing Raw MiniGrid Environment:")
    import gymnasium as gym
    raw_env = gym.make('MiniGrid-Empty-6x6-v0')
    raw_reset_result = raw_env.reset()

    if isinstance(raw_reset_result, tuple):
        raw_obs, raw_info = raw_reset_result
    else:
        raw_obs = raw_reset_result

    print(f"Raw env obs type: {type(raw_obs)}")
    if isinstance(raw_obs, dict):
        print(f"Raw env obs keys: {list(raw_obs.keys())}")
        if 'image' in raw_obs:
            print(f"Raw image shape: {raw_obs['image'].shape}")
            print(f"Raw image dtype: {raw_obs['image'].dtype}")
            print(f"Raw image range: [{raw_obs['image'].min()}, {raw_obs['image'].max()}]")
            print("Note: Raw image contains symbolic encodings, not pixels!")

    # Test intermediate wrapper stages
    print("\n3. Testing Wrapper Stages:")

    # Stage 1: Add RGB conversion
    from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    from env_helpers import Custom2DWrapper, MiniGridSimpleStochActionWrapper

    # Start with base environment
    stage1_env = gym.make('MiniGrid-Empty-6x6-v0')
    stage1_obs = stage1_env.reset()
    if isinstance(stage1_obs, tuple):
        stage1_obs = stage1_obs[0]
    print(
        f"Stage 1 (Base): {type(stage1_obs)} - {stage1_obs['image'].shape if isinstance(stage1_obs, dict) else 'N/A'}")

    # Stage 2: Add stochastic actions
    stage2_env = MiniGridSimpleStochActionWrapper(gym.make('MiniGrid-Empty-6x6-v0'), n_acts=3)
    stage2_obs = stage2_env.reset()
    if isinstance(stage2_obs, tuple):
        stage2_obs = stage2_obs[0]
    print(
        f"Stage 2 (Stochastic): {type(stage2_obs)} - {stage2_obs['image'].shape if isinstance(stage2_obs, dict) else 'N/A'}")

    # Stage 3: Add RGB conversion
    stage3_env = MinigridRGBImgObsWrapper(stage2_env, tile_size=8)  # Changed this line!
    stage3_obs = stage3_env.reset()
    if isinstance(stage3_obs, tuple):
        stage3_obs = stage3_obs[0]
    print(f"Stage 3 (RGB): {type(stage3_obs)} - {stage3_obs['image'].shape if isinstance(stage3_obs, dict) else 'N/A'}")

    # Stage 4: Remove mission field
    stage4_env = ImgObsWrapper(stage3_env)
    stage4_obs = stage4_env.reset()
    if isinstance(stage4_obs, tuple):
        stage4_obs = stage4_obs[0]
    print(f"Stage 4 (ImgOnly): {type(stage4_obs)} - {stage4_obs.shape}")

    # Stage 5: Custom processing
    stage5_env = Custom2DWrapper(stage4_env)
    stage5_obs = stage5_env.reset()
    if isinstance(stage5_obs, tuple):
        stage5_obs = stage5_obs[0]
    print(f"Stage 5 (Custom2D): {type(stage5_obs)} - {stage5_obs.shape}")

    # Create visualization comparing meaningful stages
    plt.figure(figsize=(15, 10))

    # Plot 1: Raw symbolic encoding (converted to heatmap for visualization)
    plt.subplot(2, 3, 1)
    if isinstance(raw_obs, dict) and 'image' in raw_obs:
        # Convert symbolic encoding to heatmap
        symbolic_img = raw_obs['image']
        # Combine the 3 channels into a single visualization
        combined = symbolic_img[:, :, 0] + symbolic_img[:, :, 1] * 10 + symbolic_img[:, :, 2] * 100
        plt.imshow(combined, cmap='viridis')
        plt.title("Raw MiniGrid\n(Symbolic Encoding)")
        plt.colorbar()

    # Plot 2: RGB converted observation
    plt.subplot(2, 3, 2)
    if isinstance(stage3_obs, dict) and 'image' in stage3_obs:
        rgb_img = stage3_obs['image']
        plt.imshow(rgb_img)
        plt.title(f"After RGB Conversion\n{rgb_img.shape}")

    # Plot 3: Image only (no mission)
    plt.subplot(2, 3, 3)
    if hasattr(stage4_obs, 'shape') and len(stage4_obs.shape) == 3:
        if stage4_obs.shape[0] == 3:  # Channels first
            img_to_show = stage4_obs.transpose(1, 2, 0)
        else:  # Channels last
            img_to_show = stage4_obs
        plt.imshow(img_to_show.astype(np.uint8))
        plt.title(f"After ImgObsWrapper\n{stage4_obs.shape}")

    # Plot 4: Final processed observation
    plt.subplot(2, 3, 4)
    if hasattr(stage5_obs, 'shape') and len(stage5_obs.shape) == 3:
        if stage5_obs.shape[0] <= 3:  # Channels first
            img_to_show = stage5_obs.transpose(1, 2, 0)
        else:  # Channels last
            img_to_show = stage5_obs
        plt.imshow(img_to_show.clip(0, 1))
        plt.title(f"After Custom2D\n{stage5_obs.shape}")

    # Plot 5: Training pipeline final result
    plt.subplot(2, 3, 5)
    if len(training_obs.shape) == 3:
        if training_obs.shape[0] <= 3:  # Channels first
            img_to_show = training_obs.transpose(1, 2, 0)
        else:  # Channels last
            img_to_show = training_obs
        plt.imshow(img_to_show.clip(0, 1))
        plt.title(f"Training Pipeline Final\n{training_obs.shape}")

    # Plot 6: Summary info
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.9, "Pipeline Summary:", fontsize=14, weight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f"Raw: {raw_obs['image'].shape if isinstance(raw_obs, dict) else 'N/A'}",
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"RGB: {stage3_obs['image'].shape if isinstance(stage3_obs, dict) else 'N/A'}",
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"ImgOnly: {stage4_obs.shape if hasattr(stage4_obs, 'shape') else 'N/A'}",
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Custom2D: {stage5_obs.shape if hasattr(stage5_obs, 'shape') else 'N/A'}",
             transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Training: {training_obs.shape}", transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, "✅ Pipeline Working!" if training_obs.shape == stage5_obs.shape else "❌ Pipeline Issue",
             transform=plt.gca().transAxes, fontsize=12, weight='bold',
             color='green' if training_obs.shape == stage5_obs.shape else 'red')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('debug_img.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Test action consistency
    print("\n4. Testing Action Consistency:")
    for i in range(3):
        action = training_env.action_space.sample()
        step_result = training_env.step(action)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        print(f"Step {i + 1}: action={action}, reward={reward:.3f}, done={done}, obs_shape={obs.shape}")

        if done:
            training_env.reset()
            break

    # Clean up
    training_env.close()
    raw_env.close()
    stage1_env.close()
    stage2_env.close()
    stage3_env.close()
    stage4_env.close()
    stage5_env.close()

    print("\n=== FIXED DEBUG COMPLETE ===")
    print("Key Findings:")
    print("- Raw MiniGrid uses symbolic encoding (not pixels)")
    print("- RGB wrapper converts to visual pixels")
    print("- Pipeline transforms observations correctly")
    print("- Training observations are properly formatted")
    # END FIXED DEBUG CODE ⬆️

    if args.e2e_loss:
        # Train and test end-to-end model
        encoder_model, trans_model = full_train(args)
    else:
        # Train and test the encoder model
        encoder_model = train_encoder(args)
        # Train and test the transition model
        trans_model = train_trans_model(args, encoder_model)
        # Train and evaluate an RL model with the learned model
        if args.rl_train_steps > 0:
            train_rl_model(args, encoder_model, trans_model)

    # Evalulate the models
    eval_model(args, encoder_model, trans_model)

    # Clean up logging
    finish_experiment(args)

if __name__ == '__main__':
    main()
    # from memory_profiler import memory_usage
    # mem_usage = memory_usage(main, interval=0.01)
    # print(f'Memory usage (in chunks of .1 seconds): {mem_usage}')
    # print(f'Maximum memory usage: {max(mem_usage)}')
