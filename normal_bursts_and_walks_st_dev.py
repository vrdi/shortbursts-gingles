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

#initial position
current = 0
values = range(-55, 56)
#calculates the normal probability for integer values on values
prob_dict = {x:normal_prob(x) for x in values}
#burst parameters
m = [1, 10, 50] #burst length
num_bursts = [20000, 2000, 400] #number of bursts
#total_length = m * num_bursts
total_length = 20000
num_trials = 1000
plt.rc('xtick',labelsize=25)
plt.rc('ytick',labelsize=25)
#bias parameters
walk_probabilities = [.5, .7, .9]
q_values = ['1', '3/7', '1/9']
#unbiased/biased random walks
plt.figure(figsize=(20,15))
for k in range(0, 3):
    p_val = walk_probabilities[k]
    walks = np.zeros((total_length + 1, num_trials)) #array of all trials of maximums of walks
    for trial in range(num_trials): 
        random_walk = [(0, 0)] #holds current position and length of walk
        burst_maxes = [(0, 0)] #holds maximum and length of walk
        maximum = 0
        current = 0
        q_val = 1/p_val - 1 #calculate q parameter from p
        for length_of_walk in range(total_length):
            alpha = rand.random()
            r, s, l = get_prob(random_walk[length_of_walk][0]) #get prob_dict at i
            if alpha < r: #move right
                current += 1
                random_walk.append((current, length_of_walk))
            elif alpha < (r+s): #stay same
                random_walk.append((current, length_of_walk))
            else:
                beta = rand.random()
                if beta < q_val: #move left if probability is less than q
                    current -= 1
                    random_walk.append((current, length_of_walk))
                else: #stay same if probability is greater than q
                    random_walk.append((current, length_of_walk))
            maximum = max(maximum, current) #calculate new maximum
            burst_maxes.append((maximum, length_of_walk))
        y, x = zip(*burst_maxes) #extract maximums
        walks[:, trial] = y #walk of maximums
    values = np.zeros((total_length + 1, 3))
    for i in range(len(walks)):
        std_dev = np.std(walks[i, :]) #calculate standard deviation of set of maximums at each step of walk
        mean = np.mean(walks[i, :]) #calculate mean of set of maximums at each step of walk
        #create an interval of 1 standard deviation around the mean
        values[i, 0] = mean
        values[i, 1] = mean - std_dev
        values[i, 2] = mean + std_dev
    plt.plot(x, values[:, 0], 'k-') #plot means
    #fill radius of 1 standard deviation
    plt.fill_between(x, values[:, 1], values[:, 2], label='q = {0}'.format(q_values[k]))


#short burst walks
for k in range(len(num_bursts)):
    walks = np.zeros((total_length + 1, num_trials)) #array of all trials of maximums of walks
    for trial in range(num_trials):
        t = 0 #step counter
        burst_max = 0 #current overall burst max of a given walk 
        burst_maxes = [(0, 0)] #holds the current burst max and length of walk
        random_walk = [(0, 0)] #holds the position and length of walk
        for j in range(num_bursts[k]): #loop for the number of bursts
            current = burst_max #set the current position of the walk to be the overall burst max
            for i in range(m[k] - 1): #loop for a burst (first m[k] - 1 steps)
                t+=1
                alpha = rand.random()
                r, s, l = get_prob(current)
                if alpha < r: #move right
                    current += 1
                elif alpha < (r+l): #move left
                    current -= 1
                random_walk.append((current, t)) #append new position and step to walk (stay same is current unchanged)
                burst_max = max(burst_max, current) #set the new burst max
                burst_maxes.append((burst_max, t)) #append the new burst max and step
            t+=1 #last step of burst (m[k] step) to set position to be burst max
            alpha = rand.random()
            r, s, l = get_prob(current)
            if alpha < r: #move right
                current += 1
            elif alpha < (r+l): #move left
                current -= 1
            burst_max = max(burst_max, current) #set new burst max
            random_walk.append((burst_max, t)) #random walk at end of burst is burst max and step
            burst_maxes.append((burst_max, t)) #append new burst max and step
        y, x = zip(*burst_maxes) #extract walk of maxes
        walks[:, trial] = y #walk of maxes
    values = np.zeros((total_length + 1, 3))
    for i in range(len(walks)):
        std_dev = np.std(walks[i, :]) #calculate standard deviation of set of maximums at each step of walk
        mean = np.mean(walks[i, :]) #calculate mean of set of maximums at each step of walk
        #create an interval of 1 standard deviation around the mean
        values[i, 0] = mean
        values[i, 1] = mean - std_dev
        values[i, 2] = mean + std_dev
    plt.plot(x, values[:, 0], 'k-') #plot means
    #fill radius of 1 standard deviation
    plt.fill_between(x, values[:, 1], values[:, 2],  label='burst, {0} steps'.format(m[k])) 
plt.xlim([0, total_length*1.05])
plt.ylim([0, 20])
plt.legend(loc='lower right',  prop={'size': 20})
plt.xlabel("Step", fontsize = 25)
plt.ylabel("Max Position", fontsize = 25)
plt.savefig("normal_st_dev.png")
plt.show()

