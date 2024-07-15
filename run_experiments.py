import random
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import numpy as np
import pandas as pd
import plotly.express as px
from q_values import QValues
from debugging_aids import print_policy
from frozen_lake_mod import FrozenLakeMod

# TODO: switch this to True after you have confirmed success on the simple experiments
RUN_FULL_EXPERIMENTS = True   

def get_environment(name: str, display_gui: bool):
    render_mode = "human" if display_gui else None

    if name == 'taxi':
        env = gym.make("Taxi-v3", render_mode=render_mode)

        # The environment comes with a TimeLimit wrapper,
        #   but we want to control that limit, so we unwrap,
        #   then rewrap with our own TimeLimit wrapper
        env = env.env  # unwrap
        env = TimeLimit(env, max_episode_steps=1000) # rewrap
    elif name == 'lake-icecleats':
        env = FrozenLakeMod(desc=None, map_name="4x4", is_slippery=False, render_mode=render_mode)
    elif name == 'lake-slippery':
        env = FrozenLakeMod(desc=None, map_name="4x4", is_slippery=True, render_mode=render_mode) 
    else:
        raise ValueError(f"Invalid environment name: {name}")

    env.name = name

    return env


def init_q_values(env: gym.Env, discount_factor: float):
    """Creates QValues of the appropriate size for this environment"""

    print(env.observation_space)
    if isinstance(env.observation_space, gym.spaces.Discrete):
        num_obs = int(env.observation_space.n)
    elif isinstance(env.observation_space, gym.spaces.Box):
        assert(len(env.observation_space.shape) == 1)
        assert(env.observation_space.dtype == np.int64)
        num_obs = int(env.observation_space.shape[0])

    assert(isinstance(env.action_space, gym.spaces.Discrete))
    qs = QValues(num_obs, int(env.action_space.n), discount_factor)
    return qs


def learn(env: gym.Env, q_vals: QValues, num_steps: int, learning_rate: float, epsilon: float):
    """ Modifies q_vals using Q-learning, taking num_steps time steps
        with an epsilon-greedy policy
    """

    observation, info = env.reset()
    q_vals.set_learning_rate(learning_rate)

    # Loops over the number of steps
    for _ in range(num_steps):
        if random.random() <= epsilon:
            # The line below picks a random action (excluding no-op actions)
            action = env.action_space.sample(info.get('action_mask', None))
        else:
            # The line below picks the best action according to the q-values
            action = q_vals.best_action(observation)

        old_obs = observation

        # Takes the action and gets information back from the environment
        observation, reward, terminated, truncated, info = env.step(action)

        if not truncated:
            # Update the Q-values based on the result from the last action
            q_vals.update(old_obs, action, reward, observation)

        if terminated or truncated:
            observation, info = env.reset()


def evaluate(env: gym.Env, q_vals: QValues, num_steps: int) -> float:
    """ Uses q_vals to pick the best actions for num_steps steps, returning average discounted reward per episode
    
        q_vals is not modified
    """

    sum_episode_reward = 0.0
    num_episodes = 0

    observation, info = env.reset()

    episode_discounted_reward = 0.0
    current_discount = 1.0

    # Loops over the number of steps
    for _ in range(num_steps):
        # The line below picks the best action according to the q-values
        action = q_vals.best_action(observation)

        # Takes the action and gets information back from the environment
        observation, reward, terminated, truncated, info = env.step(action)

        episode_discounted_reward = episode_discounted_reward + current_discount * reward
        current_discount *= q_vals.discount_factor
        
        if terminated or truncated:
            num_episodes += 1
            sum_episode_reward += episode_discounted_reward
            episode_discounted_reward = 0.0
            current_discount = 1.0

            observation, info = env.reset()

    if not terminated and not truncated:
        num_episodes += 1
        sum_episode_reward += episode_discounted_reward

    return sum_episode_reward / num_episodes
    

def run_experiment(env_name: str, initial_learning_rate: float = 0.3, learning_rate_decay: float = 0.95,
                   epsilon: float = 0.3, discount_factor: float = 0.9,
                   batch_size: int = 10000, num_batches: int = 1000,
                   gui_progress: bool = True, verbose: bool = True):
    """ Run Q-learning experiment using one set of parameters
    
        env_name is the environment name (e.g., 'lake-slippery')
        
        initial_learning_rate, learning_rate_decay, epsilon, and discount_factor
        are the parameters we have discussed in class
        
        batch_size is how many time steps of learning to conduct before evaluating the policy
        num_batches is how many batches to run in this experiment
         
        gui_progress is set to True to show the GUI of the agent's policy before
         any learning and at the end of the experiment
          
        verbose is set to True to print extra information during the experiment 
    """

    EVAL_TIME_STEPS = 10000
    GUI_TIME_STEPS = {'lake-icecleats': 25,
                'lake-slippery': 50,
                'taxi': 100
                }

    if verbose:
        print(f"Beginning experiment with {env_name=}, {initial_learning_rate=}, {learning_rate_decay=}, {epsilon=}, {discount_factor=}")


    env = get_environment(env_name, False)
    q_vals = init_q_values(env, discount_factor)

    learning_rate = initial_learning_rate


    if gui_progress:
        # Show the final policy in the GUI
        gui_env = get_environment(env_name, True)
        evaluate(gui_env, q_vals, GUI_TIME_STEPS[env_name])
        gui_env.close()

    results = []
    for batch_num in range(1, num_batches+1):
        # call the learn method with appropriate parameters
        learn(env, q_vals, batch_size, learning_rate, epsilon)

        # update learning_rate according to learning_rate_decay
        learning_rate *= learning_rate_decay

        # Evaluate the quality of the current policy
        discounted_reward = evaluate(env, q_vals, EVAL_TIME_STEPS)
        results.append(discounted_reward)

        if verbose:
            print(f'After batch {batch_num},\t discounted reward was {discounted_reward:.2f}')

            ## Note: printing the policy here could help with debugging
            ## print_policy(env_name, q_vals)

    
    env.close()

    if gui_progress:
        # Show the final policy in the GUI
        gui_env = get_environment(env_name, True)
        evaluate(gui_env, q_vals, GUI_TIME_STEPS[env_name])
        gui_env.close()

    return results




def main():
    num_batches = {'lake-icecleats': 100,
                'lake-slippery': 800,
                'taxi': 100
                }

    if RUN_FULL_EXPERIMENTS:
        initial_alpha_list = [0.3, 0.03]
        alpha_decay_list = [1.0, 0.99, 0.9]
        show_gui = False
    else:
        initial_alpha_list = [0.3]
        alpha_decay_list = [0.99]
        show_gui = True

    for env_name in ['lake-icecleats', 'lake-slippery', 'taxi']:
        for discount_factor in [0.95]:
            results = {}

            for epsilon in [0.4]:
                for initial_alpha in initial_alpha_list:
                     for alpha_decay in alpha_decay_list:  

                        param_name = f'{initial_alpha=}, {alpha_decay=}'

                        results[param_name] = run_experiment(env_name, initial_alpha, alpha_decay, epsilon, discount_factor,
                                                            batch_size=1000, num_batches=num_batches[env_name], gui_progress=show_gui)
                                                            

            
            df = pd.DataFrame(results)
            px.line(df, title=f'{env_name} (gamma={discount_factor})').show()


if __name__ == '__main__':
    main()
