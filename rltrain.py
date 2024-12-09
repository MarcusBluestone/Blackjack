#std
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import json

#lib
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

#pkg
from env import BJEnv
from eval import PredSB3
from actorcritic import CustomActorCriticPolicy

class CustomEvalCallback(BaseCallback):
    
    def __init__(self, env: BJEnv, eval_freq = 100, total_rollouts = 25, directory_loc = None):
        super(CustomEvalCallback, self).__init__()
        self.env = env
        self.directory_loc = directory_loc
        self.total_rollouts = total_rollouts
        self.rollouts = 0
        self.eval_freq = eval_freq
        self.all_evaluations = []

        self.best_evaluation = -float('inf')
        self.best_model = None
        self.n_calls = 0

    def _on_training_start(self):
        print("Getting hyped!!")
        self._collect_data()
        # self.total_rollouts = math.ceil(self.total)

    def _on_step(self):
        return True
    
    def _on_rollout_start(self) -> None:
        pass

    def _collect_data(self):
        all_scores, overall_stats = self.env.evaluate(PredSB3(name='', env=self.env, model=self.model).predict, num_r = 600, num_score = 150)
        median = np.median(all_scores)
        mean = np.mean(all_scores)
        self.all_evaluations.append((median, mean))
        print(f"Rollout #{self.rollouts}:\n\tMean = {mean}\n\tMedian = {median}")
        if median >= self.best_evaluation:
            self.best_evaluation = median
            self.best_model = self.model
        return mean, median
        
    def _on_rollout_end(self):
        self.rollouts += 1
        if self.rollouts % self.eval_freq == 0:
            self._collect_data()

    
    def _on_training_end(self):
        means, medians = zip(*self.all_evaluations)
        x_choices = range(0, self.total_rollouts+1, self.eval_freq)
        plt.plot(x_choices, means, label = f"Mean {i}")
        plt.plot(x_choices, medians, label =f"Median {i}")
        plt.legend()
        plt.title("Model Performance vs. Rollout # ")
        plt.ylabel("Evaluation Score")
        plt.xlabel("Rollout Number")
        with open(f'{self.directory_loc}/training_results.json', 'w') as f:
            json.dump({"x_choices": list(x_choices), "means": list(means), "medians": list(medians)}, f)
        plt.savefig(f'{self.directory_loc}/training_graph.png')
        self.best_model.save(f"{self.directory_loc}/best_model.zip")


        
if __name__ == "__main__":
    """
    total_rollouts: number of rollouts
    steps_per_rollout: number of steps in each rollout
    eval_freq: number of rollouts b4 we save the evaluation data [we also save the 0th step]
    """
    with open('experiments.json', 'r') as f:
        experiments = json.load(f)['experiments']
    all_exp_dir = f"test_results/{datetime.datetime.now()}"
    os.mkdir(all_exp_dir)

    for i, experiment in enumerate(experiments): 
        print(f"Beginning experiment {i+1} w/ parameters:\n{experiment.items()}")
        #Hyper-parameters
        total_rollouts = experiment['total_rollouts']
        steps_per_rollout = experiment['steps_per_rollout']
        eval_freq = experiment['eval_freq']
        observation_type = experiment['observation_type']
        render_mode = None
        net_arch = experiment['arch'] #network architecture

        #Setup
        env = BJEnv(observation_type=observation_type, render_mode=render_mode)
        total_timesteps = total_rollouts * steps_per_rollout
        print(f"Total Rollouts: {total_rollouts}")
        print(f"Total Datapoints To Collect: {total_rollouts // eval_freq + 1}")

        #Save Info.txt
        total_str = ""
        directory_loc = f'{all_exp_dir}/{observation_type},{str(net_arch)}'
        os.mkdir(directory_loc)
        total_str += f"Parameters: {total_rollouts=}, {steps_per_rollout=}, {eval_freq=}, {observation_type=}\n"
        total_str += f"Total Datapoints To Collect: {total_rollouts // eval_freq + 1}\n"
        with open(f'{directory_loc}/info.txt', 'w') as f:
            f.write(total_str)
        
        #Callback Function
        callback = CustomEvalCallback(env=env, eval_freq = eval_freq, total_rollouts=total_rollouts, directory_loc=directory_loc)

        #Training   
        if net_arch:
            model = PPO(CustomActorCriticPolicy, env=env, verbose=0, n_steps = steps_per_rollout)
        else: #use default 
            model = PPO("MlpPolicy", env=env, verbose=0, n_steps = steps_per_rollout)
        # print(model.policy.mlp_extractor)
        print(model.policy)

        model.learn(total_timesteps=total_timesteps, callback=callback)



