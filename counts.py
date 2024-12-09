import numpy as np
import matplotlib.pyplot as plt


def blackjack_count(target_value, num_decks):
    #note that A heart and A dimaond is the same as A space and A diamond -- the suites don't count; only the 
    #count of each number

    #without this simplification, exponential/combinatorial number of possible states; also views T,J,Q,K as the same

    #FULL STATE: Dealer Start + My delt cards [the dealer's later cards in state transitions]
    #over-estimation b/c of Aces? [assuming u can choose if Ace is 1 or 11 -- not actually b/c base cases will catch it]

    total = 0
    visited = set() #maps current to total future

    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    def get_sum(current):
        return np.sum(values * current)

    target = target_value

    def count(current): #current is length 14
        #this doesn't take into account which card the dealer chooses

        nonlocal total
        if current in visited:
            return 0
        if current[0] + current[-1] > 4 * num_decks: #deals w/ aces
            return 0
        if current[-2] > 16 * num_decks: #deals w/ 10's
            return 0
        for num in current[1:-2]: #all other numbrss
            if num > 4 * num_decks:
                return 0
        if get_sum(current) > target: #fix sum and number
            return 0

        total += 1
        future = 0

        for i in range(len(values)):
            next = tuple(val + (idx == i) for idx, val in enumerate(current))
            future += count(next)
        visited.add(current)

        return future

    def count_with_dealer(): #to be called AFTER count
        if visited is None:
            raise LookupError("You should've called count first!")
        full_options = set()
        for hand in visited:
            for i in range(len(values)):
                if hand[i] > 0: #something is there
                    full = tuple([i] + [val - (idx == i) for idx, val in enumerate(hand)])
                    full_options.add(full)
        return full_options
        
    count(tuple(0 for _ in range(len(values))))
    # print("Without Dealer: ", total)
    fo = count_with_dealer()
    # print("With Dealer: ", len(fo))
    return len(fo)



if __name__ == "__main__":
    plt.yticks(np.arange(0, 90000, 10000))

    for target_value in (5, 10, 15, 21, 25, 30):
        size = []
        for i in range(0, 10):
            size.append(blackjack_count(target_value = target_value, num_decks = i))
        print(size)
        plt.plot(size, label = f"Target = {target_value}")
    plt.legend()
    plt.xlabel("Number of Decks Used")
    plt.ylabel("State Space Size")
    plt.title("State Space vs. Num. Decks & Target")

    plt.savefig("Counts.png")

    # dynamic_programming()