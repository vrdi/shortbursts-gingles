import random as rand
import matplotlib.pyplot as plt

#initial position
current = 0
# burst parameters
m = [1, 10, 100] #burst length
num_bursts = [1000, 100, 10] #number of bursts
#total_length = m * num_bursts
total_length = 1000
#bias parameters
walk_probabilities = [.5, .51, .6, .7, .8]
#unbiased/biased random walks
for k in walk_probabilities:
    random_walk = [] #holds current position and length of walk
    current = 0 #start walk at 0
    for j in range(total_length):
        alpha = rand.random()
        if (alpha < k): #move right if probability is less than walk_probability
            current += 1 
        else: #move left otherwise
            current -= 1
        random_walk.append((current, j)) #append new position and length of walk
    x, y = zip(*random_walk) #extract random walk
    plt.plot(x, y, label='p = {0}'.format(k))

#short burst random walk
for r in range(len(num_bursts)):
    t = 0 #step counter
    burst_max = 0 #overall burst max for a given walk
    random_walk = []
    for j in range(num_bursts[r]): #loop for the number of bursts
        current = burst_max #set the current position of the walk to be the overall burst max
        for i in range(m[r]): #loop for burst
            t+=1
            alpha = rand.random()
            if (alpha < 0.5): #move to right with probability 0.5
                current += 1
            else: #move to left with probability 0.5
                current -= 1
            random_walk.append((current, t)) #append position and length of walk
            burst_max = max(burst_max, current) #set new overall burst max
    x, y = zip(*random_walk) #extract random walk
    plt.plot(x, y, label='burst, {0} steps'.format(m[r]))
ax = plt.gca()
ax.spines['left'].set_position(('data', 0)) #center y-axis
plt.xlim([-total_length, total_length])
plt.ylim([1, total_length * 1.05])
plt.legend(loc='upper left')
plt.xlabel("Position")
plt.ylabel("Step").set_position(('data', .3))
plt.title("Random walks and unbiased short bursts")
plt.savefig("random_walks_unbiased_bursts.png")
plt.show()