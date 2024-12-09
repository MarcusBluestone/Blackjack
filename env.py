#std
import numpy as np
import random 

#lib
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete, Box

class BJEnv(gym.Env):

    def __init__(self, observation_type='TRUE', render_mode = None):
        """
        Observation Space overview:
        1. 'TRUE': 
            - Keeps track of all drawn cards for player and dealer
            - 52-dim array
            - Each spot is: 0 empty, 1 player, or 2 dealer
            - All nums (23456789TJQKA) for suite 1, all nums for suite 2.... 
        2. 'VALUES':
            - Keeps track similar to what dp.py does
            - Only the the values of cards 
            - 20-dim array [10 for player; 10 for dealer]
            - value in array is count of that value card [note 1 = A]
        3. 'VALUES2':
            - Keeps track similar to what dp.py does
            - Only the the values of cards 
            - 20-dim array [10 for player; 10 for dealer]
            - Stored as floats, not one-hots
        3. 'HIGH_LOW':
            - Keeps track of how many of low, medium, and high 
            - Low (2-6), medium (7-9), high (10-A)
        """
        self.observation_type = observation_type
        assert observation_type in ('TRUE', 'VALUES', 'HIGH_LOW', 'VALUES2')

        if observation_type == 'TRUE': #keeps track of all drawn cards for player and dealer (52 spots; each can be 0, 1, or 2)
            self.observation_space = MultiDiscrete([3 for _ in range(52)])
        elif observation_type == 'VALUES': #all possible values for player and dealer
            self.observation_space = MultiDiscrete([4+1 for _ in range(9)] + [16+1] + [4+1 for _ in range(9)] + [16+1] )
        elif observation_type == 'VALUES2':
            total = [4, 4, 4, 4, 4, 4, 4, 4, 4, 16]
            self.observation_space = Box(low = 0, high = np.array(total + total))
        elif observation_type == "HIGH_LOW": #total number of high cards played, total number of low cards played
           # self.observation_space = MultiDiscrete([20+1, 12+1, 20+1]) #low (2-6), medium (7-9), high (10-A)
            self.observation_space = Box(low=0, high = np.array([21, 13, 21, 21, 13, 21])) #low (2-6), medium (7-9), high (10-A)

        self.action_space = Discrete(2) #0 is hit; 1 is stay
        self.render_mode = render_mode

        #Internal Tracking
        self._fulldeck =  np.array([val+suite for suite in 'CDHS' for val in '23456789TJQKA' ])

    
    def reset(self, seed = None):
        self._played = np.zeros(52) #all nums for suite 1, all nums for suite 2.... 
        self._deck = self._fulldeck.copy()
        np.random.shuffle(self._deck)

        self._draw_and_set(who=2)
        self.hidden_card = self._draw_and_set(who=2, register_in_played=False)
        self._draw_and_set(who=1, n_times=2)
        self.render()
        return self._get_obs(), {}
        
    def step(self, action):
        if len(self._deck) == 48: # Explicit check for initial blackjacks
            reward = 100
            if self._get_sum(1) == 21 and self._get_sum(2) == 21: 
                reward = 0
            elif self._get_sum(1) == 21: #player wins w/ blackjack!
                reward = 1.5
            elif self._get_sum(2) == 21: #dealer wins
                reward = -1
            if reward < 100:
                return self._get_obs(), reward, True, False, {}

        if action == 0: #hit = draw top card
            self._draw_and_set(1)
        elif action == 1: #stay = dealer draws until they reach 17
            while self._get_sum(2) < 17:
                self._draw_and_set(2)
        
        reward, terminated = self._get_reward_terminated(action)
        self.render(action)
        return self._get_obs(), reward, terminated, False, {}
    
    def _get_reward_terminated(self, action):
        """
        Get reward. Note that the action has already been applied
        """
        assert action in (0, 1), f"Got faulty action: {action}"
        
        player_sum = self._get_sum(1)
        dealer_sum = self._get_sum(2)

        if action == 0:
            if player_sum == 21: #blackjack!
                return 1, True
            elif player_sum > 21: #bust!
                return -1, True
            else: #keep playing
                return 0, False
        if action == 1:
            if dealer_sum > 21:
                return 1, True #Player wins!
            if player_sum == dealer_sum:
                return 0, True #tie
            elif dealer_sum > player_sum: #dealer wins
                return -1, True
            else: #player won
                return 1, True
            
        return None
    
    def render(self, action=None):
        if self.render_mode == "ansi":
            with open('saved_data/logs.txt', 'a') as f:
                f.write(self._render_frame(action) + '\n')
        return None
    
    def _render_frame(self, action=None):
        total_str = ""
        def card_str(indices):
            # Helper function to convert indices of played cards to card strings
            return ' '.join(self._fulldeck[i] for i in indices)

        player_indices = np.where(self._played == 1)[0]
        dealer_indices = np.where(self._played == 2)[0]
        if action is None:
            total_str += ("\nInitial Game State:")
        total_str += '\n'

        if action == 0:
            total_str += "Player Hit!\n"
        if action == 1:
            total_str += "Player Stayed!\n"
        
        total_str += f"Player's Cards: {card_str(player_indices)} ({self._get_sum(1)})\n"
        total_str += f"Dealer's Cards: {card_str(dealer_indices)};{self.hidden_card} ({self._get_sum(2)})\n"
        total_str += f"Cards Left in Deck: {len(self._deck)}"

        if action is not None:
            reward, finished = self._get_reward_terminated(action)
            if finished:
                total_str += ('\n' + {1:'Win!', -1:'Lose!', 0:'Tie'}[reward])

        return total_str

    def _get_obs(self):
        if self.observation_type == "TRUE":
            return self._played
        elif self.observation_type in ("VALUES", "VALUES2"):
            return self.get_value_state(self._played)
        elif self.observation_type == "HIGH_LOW":
            return self._get_highlow_state(self._played)
        assert ValueError, "What the heck? Fake observation type!"

    def _get_highlow_state(self, arr):
        arr = arr.reshape((4, 13)) #row is suite; col is card (23456789TJQKA))
        output = []
        for who in (1, 2):
            v = (arr == who)
            output.extend([np.sum(v[:, 0:5]), np.sum(v[:, 5:8]), np.sum(v[:, 8:])])#(sum of 2 thru 6) - (sum of T thru A) 
        return np.array(output) 
    
    def get_value_state(self, arr):
        """
        Gets an np array of cards [2-A x 4]; ouputs a state that it can recognize dealer[10] and player[10]
        """
        arr = arr.reshape((4, 13)) #row is suite; col is card (23456789TJQKA))
        output = np.zeros((2, 10)) #row is player; col = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        for who in (1, 2):
            v = (arr == who)
            output[who-1, -1] = np.sum(v[:, -5:-1]) #TJQK
            output[who-1, 0] = np.sum(v[:, -1]) #A
            output[who-1, 1:-1] = np.sum(v[:, :-5], axis = 0)
        player_state, dealer_state = list(output[0, :]), list(output[1, :]) #returns player, dealer
        return tuple(player_state + dealer_state)
        
    def _get_cl(self, card):
        #gets the location of a card in the played array
        val, suite = card
        suite_map = {'C': 1, 'D': 2, 'H': 3, 'S': 4} 
        card_map = {'T': 1, 'J': 2, 'Q': 3, 'K': 4, 'A': 5} 
        try:
            index_col = int(val) - 2
        except ValueError:

            index_col = 7 + card_map[val]
        index_row = suite_map[suite]
        return (index_row-1) * 13 + index_col
    
    def _draw_and_set(self, who: int, n_times: int = 1, register_in_played=True):
        """who is 1 (player) or 2 (dealer)"""
        for _ in range(n_times):
            chosen_card = self._deck[0]
            self._deck = np.delete(self._deck, 0)
            deck_loc = self._get_cl(chosen_card)
            if register_in_played:
                self._played[deck_loc] = who
        return chosen_card

    def _get_sum(self, who: int): 
        """
        Gets the sum total of player/dealer
            - who is 1 (player) or 2 (dealer)"
        
        """
        assert who in (1,2)
        if who == 2: #this takes care of hidden blackjack card
            hidden_card_loc = self._get_cl(self.hidden_card)
            self._played[hidden_card_loc] = 2

        values = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 1]) #match with fulldeck style
        all_values = np.concatenate([values, values, values, values], axis = 0)
        total = np.sum(all_values * (self._played  == who))
        has_ace = any((self._played[i] == who for i in (12, 25, 38, 51)))
        if total + 10 <= 21 and has_ace: #if you have ace, boost you up; happens at most once
            total += 10

        if who == 2: #return back to original
            self._played[hidden_card_loc] = 0
        return total
    
    def rollout(self, model_func = None):
        if model_func is None:
            model_func = lambda _:int(random.random() * 2)  # noqa: E731

        obs, _ = self.reset()
        while True:
            action = model_func(obs)
            obs, reward, terminated, _, _ = self.step(action)
            if terminated:
                break
        return reward
        
    def evaluate(self, model_func = None, num_r = 100, num_score = 150):
        """
        Keeps Track of 2 Things:
        1. Overall wins/lossses/ties
        2.  Scores a model by running it over num_score games and getting average result 
            We carry out this score num_r times and get the statistics on it
        """
        if self.render_mode == 'ansi':
            with open('saved_data/logs.txt', 'w') as f:
                f.write('')

        overall_stats = []
        all_scores = []

        for _ in range(num_r): #each run
            score_sum = 0
            for _ in range(num_score): #get score
                score = self.rollout(model_func)
                score_sum += score
                overall_stats.append(score)
            all_scores.append(score_sum / num_score)
        return all_scores, overall_stats
    

        
    
if __name__ == "__main__":
    bj = BJEnv("TRUE", render_mode = 'ansi')
    # print(bj.evaluate(None, num_e = 100))

