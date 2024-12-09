#std
import random as random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json

#lib
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from scipy import stats

#pkg
from env import BJEnv
from dp import DPModel

"""
Input x is np array of the observation of the model OR true state -- depends on which model
Note that env is defined in main... 
"""

class Predictor:
    def __init__(self, name: str, env: BJEnv):
        self.name = name
        self.env = env

    def predict(self, _):
        pass

class PredRandom(Predictor):
    """
    Totally random
    """
    def predict(self, _): 
        return int(random.random() * 2) 
    
class PredDP(Predictor):
    """
    Apply DP
    """
    def predict(self, _): 
        action_map = {"HIT":0, "STAY":1}
        dp_state = self.env.get_value_state(self.env._played)
        action_str = dpmodel.value(dp_state)[1]
        return action_map[action_str]
    
class PredStrat17(Predictor):
    """
    Stay on 17 and above; Hit otherwise
    """

    def predict(self, _):
        if self.env._get_sum(1) >= 17:
            return 1
        else:
            return 0
        
class PredSB3(Predictor):

    def __init__(self, name: str, env: BJEnv, model: BasePolicy):
        self.name = name
        self.env = env
        self.model = model
    
    def predict(self, _):
        return self.model.predict(self.env._get_obs())[0]
    
if __name__ == "__main__":
    #Hyper-parameters
    num_runs = 1000 #2500
    num_score = 300 #300

    #Enviornment Creation
    default_env = BJEnv(observation_type="TRUE")

    #Models
    directory = 'final5'
    dpmodel = DPModel()
    ppo_values2 = PPO.load(f"test_results/{directory}/VALUES2,None/best_model.zip")
    ppo_values2_lin = PPO.load(f"test_results/{directory}/VALUES2,[1]/best_model.zip")

    ppo_values = PPO.load(f"test_results/{directory}/VALUES,None/best_model.zip")
    ppo_values_lin = PPO.load(f'test_results/{directory}/VALUES,[1]/best_model.zip')

    ppo_true_lin = PPO.load(f'test_results/{directory}/TRUE,[1]/best_model.zip')
    ppo_true = PPO.load(f'test_results/{directory}/TRUE,None/best_model.zip')

    high_low_lin = PPO.load(f'/Users/marcusbluestone/Desktop/Blackjack/test_results/{directory}/HIGH_LOW,[1]/best_model.zip')
    high_low = PPO.load(f'/Users/marcusbluestone/Desktop/Blackjack/test_results/{directory}/HIGH_LOW,None/best_model.zip')

    options: list[Predictor] = [PredRandom('random', default_env),
                                PredDP('DP', default_env),

                                PredSB3('ppo_values2', BJEnv(observation_type="VALUES"), ppo_values2),
                                PredSB3('ppo_values2_lin', BJEnv(observation_type="VALUES"), ppo_values2_lin),

                                PredSB3('ppo_values', BJEnv(observation_type="VALUES"), ppo_values),
                                PredSB3('ppo_values_lin', BJEnv(observation_type="VALUES"), ppo_values_lin),

                                PredSB3('ppo_true', BJEnv(observation_type="TRUE"), ppo_true),
                                PredSB3('ppo_true_lin', BJEnv(observation_type="TRUE"), ppo_true_lin),

                                PredSB3('high_low', BJEnv(observation_type = "HIGH_LOW"), high_low),
                                PredSB3('high_low_lin', BJEnv(observation_type = "HIGH_LOW"), high_low_lin),

                                PredStrat17('Strat17', default_env)
                                ]

    #Run Experiments
    info_df = pd.DataFrame() #rows = info; cols = models
    times = {}
    fig,ax = plt.subplots(1, len(options), figsize = (4 * len(options), 7))

    for i, option in enumerate(options):
        print("Trying option: " + option.name)
        start_time = datetime.datetime.now()
        all_scores, overall_stats = option.env.evaluate(option.predict, num_r = num_runs, num_score = num_score)
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

        print("Median:", np.median(all_scores))
        print("Total Time:", elapsed_time)
        
        times[option.name] = elapsed_time
        info_df[option.name] = all_scores

        counts, bins, patches = ax[i].hist(overall_stats, label = option.name, bins = 9)
        for count, bin, patch in zip(counts, bins, patches):
            if count != 0:
                ax[i].text(bin + 0.5 * (bins[1] - bins[0]), count, int(count), ha='center', va='bottom')

        ax[i].set_title(option.name)
        ax[i].set_xlabel("Game Status")
        ax[i].set_ylabel("Number of Games")
        ax[i].set_ylim(0, num_runs * num_score)

    dpmodel.save() #save DP model for future compute...

    #Statistics for Experiments
    info_df.to_csv('evaluations.csv', index=True)

    print(info_df.describe())
    fig.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.savefig('output.jpg')

    print("Comparison:")
    print(stats.ttest_ind(info_df['ppo_values2_lin'], info_df['Strat17'], equal_var=False))

    
    with open('times.json', 'w') as f:
        json.dump(times, f)
