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
    # Parse args
    args = get_args()
    # Setup logging
    args = init_experiment('discrete-mbrl-full', args)

    # ADD DEBUG CODE HERE ⬇️
    print("=== DEBUGGING ENVIRONMENT PROCESSING ===")
    from env_helpers import make_env
    import matplotlib.pyplot as plt

    env = make_env(args.env_name)
    reset_result = env.reset()

    # Handle both old and new Gymnasium API
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result

    print(f"Training env obs shape: {obs.shape}")
    print(f"Training env obs dtype: {obs.dtype}")
    print(f"Training env obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Visualize what the training pipeline sees
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if len(obs.shape) == 3:
        if obs.shape[0] <= 3:  # Channels first
            plt.imshow(obs.transpose(1, 2, 0))
        else:
            plt.imshow(obs)
    else:
        plt.imshow(obs.reshape(-1, obs.shape[-1]) if len(obs.shape) > 1 else obs.reshape(1, -1))
    plt.title("Training Pipeline Observation")

    # Compare with raw environment
    import gymnasium as gym
    raw_env = gym.make('MiniGrid-Empty-6x6-v0')
    raw_reset_result = raw_env.reset()

    # Handle both old and new Gymnasium API for raw env too
    if isinstance(raw_reset_result, tuple):
        raw_obs, raw_info = raw_reset_result
    else:
        raw_obs = raw_reset_result

    if isinstance(raw_obs, dict) and 'image' in raw_obs:
        plt.subplot(1, 2, 2)
        plt.imshow(raw_obs['image'])
        plt.title("Raw MiniGrid Observation")

    plt.tight_layout()
    plt.savefig('debug_env_comparison.png')
    plt.show()
    print("=== END DEBUG ===")
    # END DEBUG CODE ⬆️

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
