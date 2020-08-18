import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import numpy as np

def normal_prob(i):
    """
    Calculates the probability of taking a step to the right, left,
    or staying at the same location for a given position i.
    Probabilities are based on the normal distribution at that location
    and the detail balance conditions.
    """
    if (i > 0):
        right = pi(i+1)/(2*pi(i))
        left = 1/2
        same = 1 - right - left
    if (i < 0):
        right = 1/2
        left = pi(i-1)/(2*pi(i))
        same = 1 - right - left
    if (i == 0):
        right = pi(i+1)/(2*pi(i))
        left = pi(i-1)/(2*pi(i))
        same = 1 - right - left
    return right, same, left

mean =  0.0 #numerically determined mean from Louisiana data
std = 1.44 #numerically determined st. dev. from Louisiana data

def pi(i):
    """
    Approximates the integral of the normal distribution function
    with a radius of 1/2 around position i.
    """
    return quad(normal_distribution_function, i-1/2, i+1/2, args=(mean,std,))[0]

def normal_distribution_function(x,mean,std):
    """
    Calculates the pdf of a normal distribution with mean and std
    at location x.
    """
    value = norm.pdf(x,mean,std)
    return value

def get_prob(i):
    """
    Returns the prob_dict at location i.
    """
    return prob_dict[i]

                    
values = range(-55, 56)
#calculates the normal probability for integer values on values
prob_dict = {x:normal_prob(x) for x in values} 

steps = 50 #testing burst lengths from 1 to steps - 1
num_bursts = 1000 #number of trials for each burst length
starting_points = [0, 1, 2] #starting positions of random walk
for start in starting_points:
    expected_values = [] #list of expected values for increasing step sizes
    for k in range(1, steps):
        maxes = [] #list of burst maxes
        for j in range(num_bursts):
            random_walk = [start] #random walk starts at starting_point
            for i in range(k):
                alpha = rand.random()
                #calculates probability of moving right, left, or staying the same
                r, s, l = get_prob(random_walk[i]) 
                if alpha < r:
                    next = random_walk[i] + 1
                elif alpha < (r+s):
                    next = random_walk[i]
                else:
                    next = random_walk[i] - 1
                random_walk.append(next) #appends next step of walk
            maxes.append(max(random_walk)) #appends the burst max to a list of maxes for a given step size
        expected_value = np.mean(maxes) #take an average of the burst maxes
        #appends expected value for a given step size to the accumulating list
        expected_values.append(expected_value) 
    plt.scatter(range(1, steps), expected_values, label='$X_0$ = {0}'.format(start))
plt.legend(loc = "lower right")
plt.xlabel("Number of steps")
plt.ylabel("Expected Maximum Value")
plt.savefig("normal_expected_max_value.png")
plt.show()
