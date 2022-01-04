from numpy import random
import numpy as np
import functools
import time

"""

Notes on this file: 

First we set up some helper functions to maintain the state array and model rabbit behavior. 
Then we set up the general procedure for rabbit algorithms:  and define an initial one: random guessing 
Then we have other candidate algorithms defined in seperate functions. 
Then we define an evaluation function to run many tests on each algo and summarise the results. 
Then we have commented code for evaluating the algorithms, and notes on how each one performed.



Problem:


State note: this code uses backwards index conventions to the problem statement
    state[t = guess_count-1][x = positons]

the state array is completely unneeded for testing these models and it consumes a lot of memory. 
I might remove it later

We can evaluate the algorithms on the expected time to reach the rabbit

TODO it would be cool to plot the frequency distribution of number of guesses required per trial. 
    This might reveal that some strategies are like a risky method and better if you need to get the rabbit 
    in the first 20 guesses or something, but worse if you need long-term performance


Performance note: we need to store the state history because the problem asks us to, 
    but it will run faster if we rely on three important values:
        1. position of the rabbit
        2. position of the guess
        3. number of guesses performed
    we can store these in an additional array called params
    params = [r_pos, g_pos, guess_count]

The number of holes is an argument so that we can test smaller cases first


Algorithm insight:
In the long run, where is the rabbit most likely to be? Nearer to the boundaries? 
Can we plot the rabbit position density after something like 1000 simulated rabbit moves?
- we do this with function test_p_state()
see: unimpacted_rabbit_probabilities.xlsx for details


Algorithm insight:
The rabbit will always alternate between even and odd holes. 
theoretically, If we were able to know for certain the rabbits sync, then we can cut the possibilities in half.
if we are moving one hole each time, we can be 'in sync' or 'out of sync' with the rabbit
if we move like this, We could have a threshold where it becomes likely that we are 'out of sync' with the rabbit. 
in that case, we could switch our sync - the circular_sync_net() algo tests this idea

So far, moving by two spaces each turn performs the best. Why is this? 
- Compared to moving one space at a time, it is far superior to move by two, in addition to the sync issues.
- If we are on space 41, it is 50% less likely that the rabbit will be on space 42 on the next turn. 

Early-round probability analysis:
before we guess, each hole has a 1 in 100 chance to contain a rabbit. After we make a guess in the middle of the board (41, for example), 
each other hole has a 1 in 99 chance to contain a rabbit. For the rabbit's second location these are the probabilities:
- adjacent holes 42 and 40:             1 in 198 (~half as likely as a standard hole, because one possible origin is eliminated)
- boundary holes 0 and 99:              1 in 198 (~half as likely as a standard hole, because there is only one origin)
- adjacent boundary holes 1 and 98:     3 in 198 (~50% more likely than an average hole because starting positions 0 and 99 MUST go there)
- standard hole (93 of them):           slightly less than 1 in 100 (standard probability)
On the third turn, assuming nothing on the boundary gets checked on the second turn: 
    The boundary holes 0 and 99 are     ~75% of the standard probability. ('standard probability' is ~1 in 100). 
    holes 1 and 98 go back to           standard probability 
    holes 2 and 97 are                  ~125% standard probability. 
On the fourth turn, assuming nothing on the boundary has been checked:
    -boundary holes:                    50% standard probability
    -adjacent boundary holes:           137.5% standard probability
    -second-adjacent boundary holes:    standard probability
    -third-adjacent boundary holes:     112.5% standard probability

If we are guessing the most likely hole each time: 
On the third turn, assuming we guessed 98 last turn (since it is most likely on turn 2):
    The boundary hole 99 is     impossible - 0% of the standard probability. 
    hole 98                     75% standard probability. 
    hole 97 is                  50% standard probability. 
On the fourth turn, assuming nothing on the boundary has been checked:
    -boundary holes:                    50% standard probability
    -adjacent boundary holes:           137.5% standard probability
    -second-adjacent boundary holes:    standard probability
    -third-adjacent boundary holes:     112.5% standard probability


I created double_step_tuned_net() to try to take advantage of these early probability differences


Probability ripple: like a small wave
The impact of guessing ascending even-numbered holes lowers the probability of the overall guessed area, like mowing the lawn or something (probability lawnmower??)
this observation is apparent, but I wonder if we can quantify how much lower probability the area is. 

if we guess in sequence: 20, 22, 24, 26, 28, and 30, let's look at how each square could be occupied. 
In this  chart, G is the guessed square, and the number is the rough multiple of standard probability that the other squares contain
T   20  21  22  23  24  25  26  27  28  29  30  
1   G   1   1   1   1   1   1   1   1   1   1
2   1   .5  G   1   1   1   1   1   1   1   1
3   .75 .5  1   .5  G   1   1   1   1   1   1
4   .75 .88 .5  .5  .75 .5  G   1   1   1   1
5   .95 .63 .7  .63 .5  .35 .75 .5  G   1   1

in the long run, without guesses in the area, these squares will equalize in probability, so it makes sense to guess in this sequence 
so that we maintain a 'probability ripple', forcing increased chances to stay in front of us. 
This explains why taking double steps is better than taking steps of 4, 6, or 8 in the variable_step_size_net()

Algorithm insight:
Doing this 'probability ripple' chart by hand is maybe dumb. Why don't I just write some code that can do this math???

TODO Oh, this is where the matrix algebra comes in. 
and that's the entire point of the state array...
Should we maintain a 'state' array of the probabilities of each square at any given time?
If we do that, then we can simply guess the most likely square at all times. 
Or, we can guess the square that will give us a future square with high probability??? This specific idea is maybe dumb


idea: can we animate the rabbit paths and guessing patterns in some way?


# there will be a deterministic best algorithm (with some symetrical equivalents)
# this will be an explicit list of moves to take against all possible rabbit possibilities
# this game is completely 'solvable'


"""

def gen_initial_state(hole_count=100):
    """ generates the initial rabbit position"""
    # for each trial, we need to maintain rabbit and guess history with a state array
    # state[hole_status, time]
    state = [[0]*hole_count]
    # initialize random rabbit state
    # we will call our rabbit "Fiver" and set his position equal to 5
    r_pos = random.randint(0, hole_count)
    state[0][r_pos] = 5
    return state, r_pos


def write_state(state, r_pos, g_pos, hole_count=100):
    """ adds the state to the history log, after guesses and movement have been performed """
    next_state = [0]*hole_count
    next_state[r_pos] = 5
    next_state[g_pos] = 8
    state.append(next_state) 
    return state


def gen_initial_probability(hole_count=100, g_pos=None):
    """ initializes probability array with probability of 1 everywhere except a 0 where the guess was made"""
    p_state = [1]*hole_count
    if g_pos:
        p_state[g_pos] = 0
    return p_state


def move_rabbit(r_pos, hole_count=100):
    """ randomly moves the rabbit to a valid adjacent hole """
    # handle edge cases
    if r_pos == hole_count - 1:
        return r_pos - 1
    if r_pos == 0:
        return r_pos + 1
    # randomly move up or down. 
    movement_options = [-1, 1]
    return r_pos + random.choice(movement_options)


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



"""
general procedure for rabbit algorithms:

# initial setup:
1. generate initial state and rabbit position
2. use algo to make an initial guess
3. increment the guess_count
3. check if the guess is correct - eval win condition
4. add the initial guess to the initial state array

# iteration:
1. Move the rabbit with the move_rabbit() function
2. TODO unsure here, should we check if there is a win condition when the rabbit moves? - UPDATE: we do not
3. use the algo to make a guess
4. increment the guess count
5. check if the guess is correct
6. use write_state to add the state to the history

What to do when you evaluate the win condition:
    - if guess is correct, then return the guess_count and the state history
    - possibly add a final row to the state array with like an 8 for where the guess and rabbit converged

"""


""" 
naive solution: random guesses

any algo that doesn't beat this is useless. We know that random guesses would have an EV of 100
The expected number of guesses for the first success would be a geometric distribution I think
"""


# @timer
def random_guesses(hole_count):
    """ algorithm to find the rabbit with completely random guesses. 
    This is basically just practice to set up the logical flow of a rabbit-guessing algorithm """
    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. use algo to make an initial guess
    g_pos = random.randint(0, hole_count)
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count

    # iterate until guess is correct:
    while r_pos != g_pos:
        # 1. Move the rabbit with the move_rabbit() function
        r_pos = move_rabbit(r_pos, hole_count)
        # 3. use the algo to make a guess
        g_pos = random.randint(0, hole_count)
        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count



# example of how to run a single trial and print the results. used this for debugging rabbit and guess behavior
# state, r_pos, g_pos, guess_count = random_guesses(hole_count=20)
# print("here is the history of the rabbit search:")
# for i in range(len(state)):
#     print(state[i])

# print("")
# print("Success! Rabbit found in", guess_count, "moves!")
# print("Rabbit found in hole", r_pos, ". guess:", g_pos)



# algo idea - picking the same number every time
def never_move(starting_position, hole_count=100):
    """ algorithm starts at a certain spot and moves to the right until it reaches the end. 
    When it reaches the end, it loops back in a circle """
    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. use algo to make an initial guess
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count

    # iterate until guess is correct:
    while r_pos != g_pos:
        # 1. Move the rabbit with the move_rabbit() function. the guess does not move in this algo
        r_pos = move_rabbit(r_pos, hole_count)
        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count




# algo idea - Circular net
# pick a starting point and move in a single direction. 
# when it reaches the end, it loops back to the first hole. 
# we can test to see what is the optimal net size from 1 to 99 (left and right are equivalent, so there are 100 possible simple nets)

# **** update: Circular net algorithm is terrible because it fails to find rabbit 50% of the time.
# if rabbit starts on an even number and the circular net algo starts on an odd, then it will never find the rabbit

def circular_net(starting_position, hole_count=100):
    """ algorithm starts at a certain spot and moves to the right until it reaches the end. 
    When it reaches the end, it loops back in a circle """
    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. use algo to make an initial guess
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count

    # iterate until guess is correct:
    while r_pos != g_pos:
        # 1. Move the rabbit with the move_rabbit() function
        r_pos = move_rabbit(r_pos, hole_count)
        # 3. use the algo to make a guess
        g_pos += 1
        if g_pos == hole_count:
            g_pos = 0
        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count



# algo idea - Circular net with 'sync' awareness threshold
# if we are moving one hole each time, we can be 'in sync' or 'out of sync' with the rabbit
# we can set a threshold where it becomes likely that we are 'out of sync' with the rabbit. we can test for the best threshold.
# when the threshold is reached, we could switch our sync. 

def circular_sync_net(starting_position, sync_threshold=75, hole_count=100):
    """ algorithm moves to the right until it reaches the end. 
    When it reaches the end, it loops back in a circle. when sync_threshold number of guesses is reached g_pos will stall once """
    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. use algo to make an initial guess
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count
    while r_pos != g_pos:
        r_pos = move_rabbit(r_pos, hole_count)
        # 3. make guess. if we reach sync threshold, we add an additional increment
        g_pos += 1
        if guess_count % sync_threshold == 0:
            g_pos += 1
        if g_pos >= hole_count:
            g_pos = 0
        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count




# algo idea - Simple Net with pausing
# pick a starting point and move in a single direction. pauses on each hole 

def circular_pause_net(starting_position, hole_count=100):
    """ algorithm moves to the right until it reaches the end. It pauses once on every guess. 
    When it reaches the end, it loops back in a circle. """
    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. use algo to make an initial guess
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count
    while r_pos != g_pos:
        r_pos = move_rabbit(r_pos, hole_count)
        # 3. make guess. if we reach sync threshold, we add an additional increment
        if guess_count % 2 == 0:
            g_pos += 1
        if g_pos >= hole_count:
            g_pos = 0
        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count



# algo idea - Simple Net where you move two steps every time
# pick a starting point and move in a single direction by two steps each time
# this net has an average performance of 75, and 
def double_step_net(hole_count=100):
    """ algorithm moves to the right by two until it reaches the end. 
    When it reaches the end, it loops back in a circle. """
    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. use algo to make an initial guess
    g_pos = 1
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count
    while r_pos != g_pos:
        r_pos = move_rabbit(r_pos, hole_count)
        # 3. make guess. 
        g_pos += 2
        if g_pos >= hole_count:
            g_pos = 0
        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count



# algo idea - Simple Net where you move n steps every time. Generalized case of circular_double_net
# Will fail if step_size is an odd number

def variable_step_size_net(step_size = 2, hole_count=100):
    """ algorithm moves to the right by the step_size until it reaches the end. 
    When it reaches the end, it loops back in a circle. """
    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. use algo to make an initial guess
    g_pos = 0
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count
    while r_pos != g_pos:
        r_pos = move_rabbit(r_pos, hole_count)
        # 3. make guess. if we reach sync threshold, we add an additional increment
        g_pos += step_size
        if g_pos >= hole_count:
            g_pos = g_pos - hole_count
        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count

# this is the best algorithm, with an average performance of 74
def double_step_tuned_net(hole_count=100):
    """ this is the same as the circular double net except that we are trying to take advantage of the early-round probabilities. 
    our first three guesses have increased chances (1x, 1.5x, 1.25x), compared to (1x, 1x, 1x)
    Results are not very different than the standard circular double net, though I think theoretically they must be better """
    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. use algo to make an initial guess. 
    """Start at 96 so that we can guess 98 on the next turn and 2 on the following turn"""
    g_pos = 96
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count
    while r_pos != g_pos:
        r_pos = move_rabbit(r_pos, hole_count)
        # 3. make guess. if we reach sync threshold, we add an additional increment
        g_pos += 2
        if g_pos >= hole_count: # don't bother guessing 99 because it is not as likely. instead loop back after guessing 97
            g_pos = 2
        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count


# algo idea - we do the double_step_net, but each time we loop back around, we tighten our loop to exclude a few more of the numbers on the edge
# this one is better than random, but slightly worse than the simple double_step net
# more variance because sometimes the rabbit will hide at the edges and you will miss him for a long time.
# However, it's also more likely that you get him in less than 75 moves, compared to the standard double net. 
# this is a slightly more 'agressive' and risky strategy
# reset_threshold=0 is a base case that makes this function identical to double_step_tuned_net() 
def double_step_reset_net(reset_threshold = 0, hole_count=100):
    """ Added reset count to tighten search around central values. 
    this reset_count process makes the function slightly worse. """
    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. use algo to make an initial guess. 
    """Start at 96 so that we can guess 98 on the next turn and 2 on the following turn"""
    g_pos = 96
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count
    """reset increment will cause the loop to gradually avoid peripheral values in the long term"""
    reset_increment = 0
    while r_pos != g_pos:
        r_pos = move_rabbit(r_pos, hole_count)
        # 3. make guess. if we reach sync threshold, we add an additional increment
        g_pos += 2
        if g_pos >= hole_count - reset_increment: # don't bother guessing 99 because it is not as likely. instead loop back after guessing 97
            if reset_increment > reset_threshold:
                reset_increment = 0
            reset_increment += 2
            g_pos = reset_increment # reset loop at 2 because it is 1.25x on turn 3

        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count



# evaluation: 
# we need to do a large number of trials in order to remove variance and demonstrate which algorythms are the best
# we define an evaluation function to help us do this.

# idea to improve evaluation - UPDATE: tested this idea and it doesn't make a difference
# since the starting position is random, we could do something like 4 trials from each starting position, which might reduce variance for n trials
# we can test how much it reduces variance by doing multiple evaluations of the same algo (using a simple one)

# Note: evaluate the evaluation function... How many trials are needed to converge on the true average?
# we run the evaluation function multiple times to see how consistently the evaluation performs for each rabbit algorithm. 
# if the evaluations are close to each other, then it is likely that it has converged. 

@repeat(num_times=4)
@timer
def eval_algo(rabbit_algo, trials, *args, **kwargs):
    """ evaluates the specified rabbit algorithm over the specified number of trials """

    if trials > 1000:
        print("Evaluating...")

    # TODO we could consider defining the initial array state inside the evaluation function instead of inside the algo
    results = []
    for _ in range(trials):
        guess_count = rabbit_algo(*args, **kwargs)[1]
        # state array is available as the first value of the tuple returned by the rabbit algo
        results.append(guess_count)

    # TODO do something with the results to save them in some way
    print(f"rabbit function {rabbit_algo.__name__!r} evaluated over {trials} trials. average performance:", np.average(results))
    kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
    signature = ", ".join(kwargs_repr)
    print(f"arg values: {signature}")
    # print(f"arguments: {*args} value: {*args.value}")
    print("standard deviation:", np.std(results))
    print("shortest trial:", np.amin(results), ". longest trial:", np.amax(results))

    return results


# TODO make a boxplot of the results to show the distribution


# uncomment the lines for the algorythms you'd like to test. Each one will be tested multiple times, 
# based on the num_times value in the @repeat decorator above the eval function definition


"""random_guesses performance: EV of 100 """
# eval_algo(random_guesses, trials=1000, hole_count=100) 

""" circular_net performance: fails via infinite loop 50% of the time """
# eval_algo(circular_net, trials=1000, starting_position=50, hole_count=100) 

""" average performance : 1630, when starting from position 50"""
""" starting position 10 is much worse, with an average of 52,600 turns """
# eval_algo(never_move, trials = 1000, starting_position=10, hole_count=100) 

""" average performance : 111, with sync_threshold = 75"""
""" average performance : 286, with sync_threshold = 50"""
""" average performance : 102, with sync_threshold = 100"""
""" best performance is with sync threshold of 100. """
# eval_algo(circular_sync_net, starting_position=1, trials=1000, sync_threshold=100, hole_count=100)

""" average performance : 120, with starting_position=1"""
# eval_algo(circular_pause_net, trials=1000, starting_position=1, hole_count=100)

""" 
Success! average performance : 75 ish (starting position doesn't really matter)"""
# eval_algo(double_step_net, trials=5000, hole_count=100)


""" average performance : 75 ish, with step_size = 2 """
""" average performance : 92 ish, with step_size = 4 """
""" average performance : 95 ish, with step_size = 6 """
""" average performance : 92 ish, with step_size = 8 """
# eval_algo(variable_step_size_net, trials=1000, step_size=2, hole_count=100) # exactly equivalent to double_step_net
# eval_algo(variable_step_size_net, trials=1000, step_size=4, hole_count=100)
# eval_algo(variable_step_size_net, trials=1000, step_size=6, hole_count=100)
# eval_algo(variable_step_size_net, trials=1000, step_size=8, hole_count=100)


""" average performance : 74, very small improvement to standard double_step_net. similar standard deviation"""
""" This is the current best algorithm. """
# eval_algo(double_step_tuned_net, trials=5000, hole_count=100)

""" average performance : 76 ish, pretty comparable to standard double_step_net? more variance because 
sometimes the rabbit will hide at the edges and you will miss him for a long time.
However, it's also more likely that you get him in less than 75 moves, compared to the standard double net. 
Standard deviation is about 80 compared to about 70 for the standard double_step_net
this is a slightly more 'agressive' and risky strategy""" 
# eval_algo(double_step_reset_net, trials=3000, hole_count=100)



# the functions below use idea of maintaing the probability state of each hole over time

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

    # adjust probability weights to have an average value of 1
    # TODO do we want to adjust the weighting????? - I think no actually, because:
    # we could use the unweighted values to show which is the best algorithm??
    # this would show us which one can reduce the possibilities in the very best way. 
    # additionally, we would have the p_sum as an indicator of how quickly we are coming to a solution????
    # p_sum = sum(next_state)
    # w_ratio = hole_count/p_sum
    # next_state = [x*w_ratio for x in next_state]

    return next_state


# In the long run, where is the rabbit most likely to be? Nearer to the boundaries? 
# plot the rabbit position density after something like 1000 simulated rabbit moves
def test_p_state(move_count=10, hole_count = 10):
    """ test the probability calculation and find the overall pattern of rabbit probability, if we are not making guesses. """
    p_state = gen_initial_probability(hole_count)
    p_history = []
    p_history.append(p_state)
    for _ in range(move_count):
        p_state = find_next_p_state(p_state)
        p_history.append(p_state)
    return p_history

# # find and print out the rabbit probability behavior, for the case where no guesses are made
# p_history = test_p_state(move_count=1000, hole_count = 100)
# import pandas as pd
# df = pd.DataFrame(p_history)
# df.to_csv("unimpacted_rabbit_probability.csv")




# 'smart' solution
def select_lowest_max_p(starting_position, hole_count):
    """ algorithm to find the rabbit using the iterative probability state.
    Initial guess was originally random but now can be specified 
    then, Guesses the most probable spot. if there is a tie, it selects the lower of the two.  """

    # 1. generate initial state and rabbit position
    state, r_pos = gen_initial_state(hole_count)
    # 2. make an initial guess - this one is random
    g_pos = starting_position
    # 3. increment the guess_count
    guess_count = 1
    # 5. add the initial guess to the initial state array
    state[guess_count-1][g_pos] = 8
    # 4. check if the guess is correct - eval win condition
    if g_pos == r_pos:
        return state, guess_count
    # 6. Set the initial probability state
    p_state = gen_initial_probability(hole_count, g_pos)

    # iterate until guess is correct:
    while r_pos != g_pos:
        # 1. Move the rabbit with the move_rabbit() function
        r_pos = move_rabbit(r_pos, hole_count)
        # 3. use the aray of probabilities to make a guess. Tie goes to the lowest index value
        max_p = max(p_state)
        # find the index of the maximum probability
        for i in range(len(p_state)):
            if p_state[i]==max_p:
                g_pos = i
                break
        # if max_p < 1 or guess_count >= 50:
        #     print(guess_count, g_pos, max_p, sum(p_state))
        # 4. increment the guess count
        guess_count += 1
        # 6. use write_state to add the state to the history
        state = write_state(state, r_pos, g_pos, hole_count)
        
        # set the probability to 0 at the location of the current state
        p_state[g_pos] = 0
        # 7. calculate the new probability array
        p_state = find_next_p_state(p_state)
        # print(p_state)
        # 5. check if the guess is correct (using the while loop)
        
    return state, guess_count



# # example of how to run a single trial and print the results. used this for debugging rabbit and guess behavior
s_pos = 2
state, guess_count = select_lowest_max_p(starting_position=s_pos, hole_count=100)
# state, guess_count = max_p_depth(starting_position=2, hole_count=20)
print("here is the history of the rabbit search:")
# for i in range(len(state)):
#     print(state[i])

print("")
print("starting_position:", s_pos)
print("Success! Rabbit found in", guess_count, "moves!")

# starting positions of 2 or 3 from the border are superior to other starting positions such as: 0 or 1, or in the middle
eval_algo(select_lowest_max_p, trials=1000, starting_position=2, hole_count=100)



# TODO do a depth search where we look at what next g_pos option will allow the biggest future g_pos value


# TODO there is a new best way to evaluate all of these things. 
# We can just run the functions without a rabbit and see which ones reduce the probability the fastest! 

# we could also add probability monitoring to our "basic"algos and compare the raw probability elimination between all of the functions. 
# We just put a new threshold on the functions. instead of running until we find the rabbit, we run until p_sum < threshold

# scoring function:
# score = summation of p_max * guess_count for each guess_count
# TODO to accomodate the scoring function, we need to change the functions to return the arrays of p_max, p_sum and g_count for each algo

# look at the area under the p_sum probability curve


