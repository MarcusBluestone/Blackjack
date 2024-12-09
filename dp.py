#std
import numpy as np
import random
import pickle

#lib
from tqdm import tqdm


class DPModel:
    def __init__(self, num_decks = 1, target = 21):
        self.num_decks = num_decks
        self.target = target
        self.values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #1 is Ace
        self.memo_dealer = {} #maps state to values
        self.memo_player = {} #maps state to values
        self.file_name = f'saved_data/target_{target}_decks_{num_decks}'
        try:
            with open(self.file_name, 'rb') as f:
                    self.memo_dealer, self.memo_player = pickle.load(f).values()
        except (FileNotFoundError, EOFError):
            pass

    def get_sum(self, focus):
        total = np.sum(self.values * focus)
        if total + 10 <= 21 and focus[0] > 0: #if you have ace, boost you up
            total += 10
        return total

    def get_dealer_value(self, state, time = 0):

            """
            No decisions. Just keep recursing w/ draws
            """
            player_state, dealer_state = state[:10], state[10:]

            if state in self.memo_dealer:
                return self.memo_dealer[state]
            
            dealer_sum = self.get_sum(dealer_state)
            player_sum = self.get_sum(player_state)

            if dealer_sum > self.target:
                return 1 #Player wins!
            if dealer_sum >= 17: #no more drawing for dealer [include soft or hard?]
                if player_sum == dealer_sum:
                    return 0 #tie
                elif dealer_sum > player_sum: #dealer wins
                    return -1
                else: 
                    return 1
                
            #otherwise, recurse!

            expected_values = []
            total_options = (52 * self.num_decks) - sum(state)

            for i in range(10): #loop through all possible cards to add
                value_total = 16 if i == 9 else 4
                remaining = value_total * self.num_decks - player_state[i] - dealer_state[i]
                if remaining > 0:
                    next_state = tuple(list(player_state) + [val + (idx == i) for idx, val in enumerate(dealer_state)])
                    prob = remaining/total_options
                    expected_values.append(prob * self.get_dealer_value(next_state, time=time + 1))
                
            self.memo_dealer[state] = sum(expected_values)
            return self.memo_dealer[state]

    def value(self, state, time = 0, verbose=False):
        """state is a 20-dim tuple; 10 dealer counts + 10 player counts"""
        player_state, dealer_state = state[:10], state[10:]

        if time == 0 and verbose:
            print(player_state)
            print(f"Player: {self.get_sum(player_state)}" + (" (SOFT)" if player_state[0] > 0 else ""))
            print(dealer_state)
            print(f"Dealer: {self.get_sum(dealer_state)}")

        if state in self.memo_player:
            return self.memo_player[state]
        
        if self.get_sum(player_state) == self.target: #Blackjack!
            return (1.5, "STAY")
        
        if self.get_sum(player_state) > self.target: #busted
            return (-1, None)
        
        #Hit Value [Only Uses Player State]
        total_options = (52 * self.num_decks) - sum(state)
        expected_values = []
        for i in range(10):  #picks values 1/A ... 10
            value_total = 16 if i == 9 else 4
            remaining = value_total * self.num_decks - player_state[i] - dealer_state[i]
            if remaining > 0: #some left of this card value type to use
                next_state = tuple([val + (idx == i) for idx, val in enumerate(player_state)] + list(dealer_state))
                prob = remaining/total_options
                expected_values.append(prob * self.value(next_state, time + 1)[0])
        hit_value = sum(expected_values)
        
        #Stay Value
        stay_value = self.get_dealer_value(state, time=time)

        best_action = "HIT"
        if hit_value < stay_value:
            best_action = "STAY"

        self.memo_player[state] = (max(hit_value, stay_value), best_action)
        return self.memo_player[state]
    
    def save(self):
        print("Saving DP Model Info...")
        with open (self.file_name, 'wb') as f:
            pickle.dump({"dealer": self.memo_dealer, "player":self.memo_player}, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    dpmodel = DPModel()
    # TESTING (should always hit if < 11)
    end_p = [0, 0, 1, 0, 0, 0, 0, 0, 0] #to test w/o aces
    test_d = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    for _ in tqdm(range(100)):
        random.shuffle(test_d)
        random.shuffle(end_p)
        test_p = [0] + end_p
        current = tuple(test_p + test_d)
        assert dpmodel.value(current, time = 0)[1] == "HIT", "You should be hitting here buddy"
    print("Test #1 Success!")

    #should be hit
    test_p = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1] #to test w/o aces
    test_d = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    current = tuple(test_p + test_d)
    assert dpmodel.value(current, verbose=False)[1] == "HIT"
    print("Test #2 success!")
    dpmodel.save()