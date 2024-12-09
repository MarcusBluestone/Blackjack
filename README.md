# Blackjack
Capstone Project for MIT Reinforcement Learning (6.7920) Class. 
See `overview.pdf` for the full paper.

## Overview of Files
1. `counts.py` and `dp.py` are scripts for enumarting all possible game states and also calculating the Dynamic Programming solution for Blackjack. The results can be optionally saved into a folder so that the DP can run quicker in the future.
2. `actor_critic.py` contains the code for the custom ActorCritic Networks that were used in this project
3. `env.py` contains the Custom OpenAI Gym environment for blackjack
4. `rltrain.py` is a script used to train the RL models. Experimental parameters are defined in the `experiments.json` file. 
5. `eval.py` is a script used to evaluate the models after they are trained
6. `test.ipynb` and `displays.ipynb` are notebooks used to explore and research the results graphically and analytically. They are a bit disorganized. 

## Setup
```
git clone https://github.com/MarcusBluestone/Blackjack.git
conda env create -f blackjack_conda_env.yml
```

## Running 
1. Choose which experiments you want to set up in `experiments.json`. See current listings for examples for format
2. Run `rltrain.py` to train the models based on the experiments.
3. Run `eval.py` to evaluate the models.
4. Explore the results using `test.ipynb` and `displays.ipynb`
 

