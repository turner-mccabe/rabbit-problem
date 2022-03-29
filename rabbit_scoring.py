from numpy import random
import numpy as np
import functools
import time
import pandas as pd

"""
# there is a new best way to evaluate all of these things. 
# We can just run the functions without a rabbit and see which ones reduce the probability the fastest! 

# we could also add probability monitoring to our "basic"algos and compare the raw probability elimination between all of the functions. 
# We just put a new threshold on the functions. instead of running until we find the rabbit, we run until p_sum < threshold

- forget about the state array
- forget about the rabbit
- change all functions to maintain:
    a probability state array: p_state
    a list of the remaining probability sum after each guess: p_sum_list (aka the chance that the rabbit hasn't been found yet)
    a list of the guesses made each turn: g_pos_list

- we can run the algos for a set number of moves, and plot the probabilities against each other to visually compare them. 



We can explicitly score the algorithms by calculating their EV

# scoring function:

# score = summation of probability_reduction * guess_count for each guess_count, 
in other words, if each turn is 'T', and the remaining probability is P_sum:
score(T) = summation[ (P_sum(T-1) - P_sum(T)) * T ]

as T increases, P_sum approaches 0 for any decent strategy. At large values of T, the score will converge on the actual EV
We can test to see what values of T are sufficient to approximate EV for each algo


Once we can score, it becomes easy to answer questions like: what is the best starting position for this algo? 

"""

def score(p_sum_list):
    """ score is the summation of the product of the guess number and the probability of that guess. """
    # probability for guess 1 is always 0.01, so score(t=1) = 0.01 * 1 => 0.01 
    score = 0.01
    for i in range(1,len(p_sum_list)):
        probability_decline = (p_sum_list[i-1] - p_sum_list[i])/100        
        score += (i+1) * probability_decline

    return score


# define timer decorator for timing functions as needed
def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs\n")
        return value
    return wrapper_timer


# apply to a function to make it run multiple times when called
def repeat(_func=None, *, num_times=2):
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            for _ in range(num_times):
                value = func(*args, **kwargs)
            return value
        return wrapper_repeat
    if _func is None:
        return decorator_repeat
    else:
        return decorator_repeat(_func)



def gen_initial_state(hole_count=100):
    """ randomly generates the initial rabbit position"""
    return random.randint(0, hole_count)


def gen_initial_probability(hole_count=100, g_pos=None):
    """ initializes probability array with probability of 1 everywhere except a 0 where the guess was made"""
    p_state = [1]*hole_count
    if g_pos:
        p_state[g_pos] = 0
    return p_state


def find_next_p_state(p_state):
    """ calculates the next layer of probabilities """

    # define the hole count and initialize the next state as an array of zeroes
    hole_count = len(p_state)
    next_state = [0]*hole_count

    # calculate probabilities for the first hole and final hole (50% of the one adjacent)
    next_state[0] = 0.5 * p_state[1]
    next_state[hole_count-1] = 0.5 * p_state[hole_count-2]

    # calculate probabilities for the holes adjacent to the boundary (these get 100% of boundary probability + 50% of adjacent)
    next_state[1] = p_state[0] + 0.5 * p_state[2]
    next_state[hole_count-2] = p_state[hole_count-1] + 0.5 * p_state[hole_count-3]

    # recalculate the inner probabilities (50% of the sum of both adjacents)
    for i in range(2, hole_count-2):
        next_state[i] = 0.5 * (p_state[i-1] + p_state[i+1])

    return next_state

def random_guesses(guess_limit=100, hole_count=100):
    """ algorithm to find the rabbit with completely random guesses. 
    This is basically just practice to set up the logical flow of a rabbit-guessing algorithm """
    p_state_list = []
    # 2. make an initial guess - this one is random
    g_pos = random.randint(0, hole_count)
    # 3. increment the guess_count
    guess_count = 1
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]
    p_state_list = [p_state]
    # This is the key line
    p_state = find_next_p_state(p_state)
    while guess_count < guess_limit:
        # make next guess
        g_pos = random.randint(0, hole_count)
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0

        # append p_sum_list
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        p_state_list.append(p_state)
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)

        # 4. increment the guess count
        guess_count += 1


    return p_state_list, g_pos_list, p_sum_list

# 'smart' solution
def visual_select_best_p(starting_position, guess_limit=100, hole_count=100):
    """ algorithm to reduce the probability as quickly as possible. """
    p_state_list = []
    # 2. make an initial guess - this one is random
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]
    p_state_list = [p_state]
    # This is the key line
    p_state = find_next_p_state(p_state)
    while guess_count < guess_limit:
        # 3. use the aray of probabilities to make a guess. Tie goes to the lowest index value
        max_p = max(p_state)
        # find the index of the maximum probability
        for i in range(len(p_state)):
            if p_state[i]==max_p:
                g_pos = i
                break
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0

        # append p_sum_list
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        p_state_list.append(p_state)
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)

        # 4. increment the guess count
        guess_count += 1

    p_state_list.append(p_state)
    return p_state_list, g_pos_list, p_sum_list


"""
41 is not the best place to guess. There was a small issue with the algorythm causing it to guess 0 on the second guess every time. 
This had a small impact on the scoring of the algo
Now that we fix this issue, we should be able to get better scores. 
"""


# # determine best starting position:
# 39 is best starting position
# i_list = []
# x_list = []
# for i in range(0, 149):
#     p_state_list, g_pos_list, p_sum_list = visual_select_best_p(i, 5000, 150)
#     x = score(p_sum_list)
#     i_list.append(i)
#     x_list.append(x)
#     print(i, x)

# # visualize
# p_state_list, g_pos_list, p_sum_list = visual_select_best_p(39, 5000, 200)
# x = score(p_sum_list)
# print(x)

# # visualize
p_state_list, g_pos_list, p_sum_list = random_guesses(5000, 100)
x = score(p_sum_list)
print(x)

# d = {"x":x_list, "i":i_list}
# df = pd.DataFrame(d)
# # print(df)
# df.to_csv("minima2.csv")

moves_df = pd.DataFrame(p_state_list)
moves_df.to_csv("random_visual.csv")

# d = {"select_best_p_guess":g_pos_list, "select_best_p_performance":p_sum_list}
# moves_df = pd.DataFrame(d)
# moves_df.to_csv("random_visual.csv")

# 'smart' solution
def select_best_p(starting_position, guess_limit=100, hole_count=100):
    """ algorithm to reduce the probability as quickly as possible. """
    # 2. make an initial guess - this one is random
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]

    while guess_count < guess_limit:
        # 3. use the aray of probabilities to make a guess. Tie goes to the lowest index value
        max_p = max(p_state)
        # find the index of the maximum probability
        for i in range(len(p_state)):
            if p_state[i]==max_p:
                g_pos = i
                break
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0
        # append p_sum_list
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)
        # 4. increment the guess count
        guess_count += 1

    return g_pos_list, p_sum_list


def circular_sync_net(starting_position, guess_limit=100, sync_threshold=75, hole_count=100):
    """ algorithm moves to the right until it reaches the end. 
    When it reaches the end, it loops back in a circle. when sync_threshold number of guesses is reached g_pos will stall once """
    # 2. use algo to make an initial guess
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]
    while guess_count < guess_limit:
        # 3. make guess. if we reach sync threshold, we add an additional increment
        g_pos += 1
        if guess_count % sync_threshold == 0:
            g_pos += 1
        if g_pos >= hole_count:
            g_pos = 0
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0
        # append output lists
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)
        # 4. increment the guess count
        guess_count += 1
        
    return g_pos_list, p_sum_list


def never_move(starting_position, guess_limit=100, hole_count=100):
    """ algorithm starts at a certain spot and moves to the right until it reaches the end. 
    When it reaches the end, it loops back in a circle """
    # 2. use algo to make an initial guess
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]

    while guess_count < guess_limit:
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0
        # append output lists
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)
        # 4. increment the guess count
        guess_count += 1
        
    return g_pos_list, p_sum_list


def circular_pause_net(starting_position, guess_limit=100, hole_count=100):
    """ algorithm moves to the right until it reaches the end. It pauses once on every guess. 
    When it reaches the end, it loops back in a circle. """
    # 2. use algo to make an initial guess
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]

    while guess_count < guess_limit:
        # 3. make guess. if we reach sync threshold, we add an additional increment
        if guess_count % 2 == 0:
            g_pos += 1
        if g_pos >= hole_count:
            g_pos = 0
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0
        # append output lists
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)
        # 4. increment the guess count
        guess_count += 1
        
    return g_pos_list, p_sum_list


# this is the best basic algorithm, with an average performance of 74
def double_step_tuned_net(guess_limit=100, hole_count=100):
    """ this is the same as the circular double net except that we are trying to take advantage of the early-round probabilities. 
    our first three guesses have increased chances (1x, 1.5x, 1.25x), compared to (1x, 1x, 1x)
    Results are not very different than the standard circular double net, though I think theoretically they must be better """
    # 2. use algo to make an initial guess. 
    """Start at 96 so that we can guess 98 on the next turn and 2 on the following turn"""
    g_pos = 96
    # 3. increment the guess_count
    guess_count = 1
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]

    while guess_count < guess_limit:
        # 3. make guess. if we reach sync threshold, we add an additional increment
        g_pos += 2
        if g_pos >= hole_count: # don't bother guessing 99 because it is not as likely. instead loop back after guessing 97
            g_pos = 2
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0
        # append output lists
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)
        # 4. increment the guess count
        guess_count += 1
        # 5. check if the guess is correct (using the while loop)
        
    return g_pos_list, p_sum_list


def variable_step_size_net(guess_limit=100, step_size = 2, hole_count=100):
    """ algorithm moves to the right by the step_size until it reaches the end. 
    When it reaches the end, it loops back in a circle. """
    # 2. use algo to make an initial guess
    g_pos = 50
    # 3. increment the guess_count
    guess_count = 1
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]

    while guess_count < guess_limit:
        # 3. make guess. if we reach sync threshold, we add an additional increment
        g_pos += step_size
        if g_pos >= hole_count:
            g_pos = g_pos - hole_count
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0
        # append output lists
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)
        # 4. increment the guess count
        guess_count += 1
        # 5. check if the guess is correct (using the while loop)
        
    return g_pos_list, p_sum_list


# algo idea - we do the double_step_net, but each time we loop back around, we tighten our loop to exclude a few more of the numbers on the edge
# this one is better than random, but slightly worse than the simple double_step net
# more variance because sometimes the rabbit will hide at the edges and you will miss him for a long time.
# However, it's also more likely that you get him in less than 75 moves, compared to the standard double net. 
# this is a slightly more 'agressive' and risky strategy
# reset_threshold=0 is a base case that makes this function identical to double_step_tuned_net() 
def double_step_reset_net(guess_limit=100, reset_threshold = 0, hole_count=100):
    """ Added reset count to tighten search around central values. 
    this reset_count process makes the function slightly worse. """
    # 2. use algo to make an initial guess. 
    """Start at 96 so that we can guess 98 on the next turn and 2 on the following turn"""
    g_pos = 96
    # 3. increment the guess_count
    guess_count = 1
    """reset increment will cause the loop to gradually avoid peripheral values in the long term"""
    reset_increment = 0
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]

    while guess_count < guess_limit:
        # 3. make guess. if we reach sync threshold, we add an additional increment
        g_pos += 2
        if g_pos >= hole_count - reset_increment: # don't bother guessing 99 because it is not as likely. instead loop back after guessing 97
            if reset_increment > reset_threshold:
                reset_increment = 0
            reset_increment += 2
            g_pos = reset_increment # reset loop at 2 because it is 1.25x on turn 3
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0
        # append output lists
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)
        # 4. increment the guess count
        guess_count += 1
        
    return g_pos_list, p_sum_list





# this one ends up being identical to the other
def select_best_1deep_p(starting_position, guess_limit=100, hole_count=100):
    """ algorithm to reduce the probability as quickly as possible. """
    # 2. make an initial guess - this one is random
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)
    # initialize lists for output
    g_pos_list = [g_pos]
    p_sum_list = [sum(p_state)]

    while guess_count < guess_limit:
        # 3. use the aray of probabilities to make a guess. Tie goes to the lowest index value
        max_p = max(p_state)
        # find the indexes of the maximum probability
        candidate_indexes = []
        for i in range(len(p_state)):
            if p_state[i]==max_p:
                candidate_indexes.append(i)

        # tiebreaker logic if there are muliple top choices:
        if len(candidate_indexes) > 1:
            current_max = -1
            for j in range(len(candidate_indexes)):
                temp_p_state = p_state
                temp_p_state[candidate_indexes[j]]
                # find the highest p value for the next state
                temp_next_p = find_next_p_state(temp_p_state)
                temp_max_p = max(temp_next_p)
                if temp_max_p > current_max:
                    # assign the g_pos value that will cause the next pick to have the highest g_pos value:
                    g_pos = candidate_indexes[j]
        
        # simple assignment if there is only one top choice:
        else:
            g_pos = candidate_indexes[0]


        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0
        # append p_sum_list
        g_pos_list.append(g_pos)
        p_sum_list.append(sum(p_state))
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)
        # 4. increment the guess count
        guess_count += 1

    return g_pos_list, p_sum_list



# g_pos_list, p_sum_list = select_best_p(41, 500)
# d = {"select_best_p_guess":g_pos_list, "select_best_p_performance":p_sum_list}
# df = pd.DataFrame(d)


# g_pos_list, p_sum_list = select_best_1deep_p(58, 500)
# df["select_best_1deep_p_guess"] = g_pos_list
# df["select_best_1deep_p_performance"] = p_sum_list

# # g_pos_list, p_sum_list = random_guesses(500)
# # df["random_guesses1_performance"] = p_sum_list

# # g_pos_list, p_sum_list = random_guesses(500)
# # df["random_guesses2_performance"] = p_sum_list

# # g_pos_list, p_sum_list = random_guesses(500)
# # df["random_guesses3_performance"] = p_sum_list

# g_pos_list, p_sum_list = circular_sync_net(2, 500)
# df["circular_sync_guess"] = g_pos_list
# df["circular_sync_performance"] = p_sum_list

# g_pos_list, p_sum_list = never_move(50, 500)
# df["never_move_guess"] = g_pos_list
# df["never_move_performance"] = p_sum_list

# g_pos_list, p_sum_list = circular_pause_net(2, 500)
# df["circular_pause_guess"] = g_pos_list
# df["circular_pause_performance"] = p_sum_list

# g_pos_list, p_sum_list = double_step_tuned_net(500)
# df["2step_guess"] = g_pos_list
# df["2step_performance"] = p_sum_list

# g_pos_list, p_sum_list = variable_step_size_net(500, step_size=4)
# df["4step_guess"] = g_pos_list
# df["4step_performance"] = p_sum_list

# g_pos_list, p_sum_list = variable_step_size_net(500, step_size=6)
# df["6step_guess"] = g_pos_list
# df["6step_performance"] = p_sum_list

# g_pos_list, p_sum_list = double_step_reset_net(500, 6)
# df["reset_2step_guess"] = g_pos_list
# df["reset_2step_performance"] = p_sum_list

# df.to_csv("test.csv")






# # determine best starting position:
# for i in range(0, 99):
#     g_pos_list, p_sum_list = select_best_1deep_p(i, 5000)
#     x = score(p_sum_list)
#     print(i, x)

# g_pos_list, p_sum_list = select_best_1deep_p(41, 5000)
# x = score(p_sum_list)
# print(x)


# # determine best starting position:
# for i in range(0, 99):
#     g_pos_list, p_sum_list = select_best_p(i, 5000)
#     x = score(p_sum_list)
#     print(i, x)

# # 41 is the best starting position
# g_pos_list, p_sum_list = select_best_p(41, 5000)
# x = score(p_sum_list)
# print(x)


# # 41 is the best starting position
# g_pos_list, p_sum_list = random_guesses(5000)
# x = score(p_sum_list)
# print(x)




# # find double step tuned net performance
# g_pos_list, p_sum_list = double_step_tuned_net(5000)
# x = score(p_sum_list)
# print(x)

# # find circular pause net best starting position (best starting position is 1)
# # for i in range(0, 99):
# #     g_pos_list, p_sum_list = circular_pause_net(i, 5000)
# #     x = score(p_sum_list)
# #     print(i, x)

# g_pos_list, p_sum_list = circular_pause_net(1, 5000)
# x = score(p_sum_list)
# print(x)


# # find best reset threshold
# for i in range(0, 40, 2):
#     g_pos_list, p_sum_list = double_step_reset_net(5000, i)
#     x = score(p_sum_list)
#     print(i, x)



# TODO do a depth search where we look at what next g_pos option will allow the biggest future g_pos value



