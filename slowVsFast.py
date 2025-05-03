import numpy as np
import matplotlib.pyplot as plt
import random

T = 100
gamma = 3 #exponential decay factor
'''
gamma=1 -> Dynamic always better
gamma=5 -> Static leads
'''

time = np.arange(T)
delta = 0.8 #discount factor
skill_vals = [0,1,2]

def simulate_skill_path(init_prob, final_prob, T=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    skill_values = [0]  # start at Basic skill level
    for t in range(1, T):
        transition_prob = final_prob * (1 - np.exp((-t+1)/gamma)) + init_prob * np.exp((-t+1)/gamma) 
        current   = skill_values[-1]
        weights   = list(transition_prob[current])
        new_level = np.random.choice(skill_vals, p=weights)
        skill_values.append(new_level)
    return skill_values

def plot_values(label , skill_path):
    discounted_values = [delta**t * skill_vals[skill] for t, skill in enumerate(skill_path)]
    cumulative_values = np.cumsum(discounted_values)
    plt.plot(time, cumulative_values, label=label)


static_prob =  np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6],
])


init_prob = np.array([
    [0.8, 0.2, 0.0],  
    [0.6, 0.3, 0.1],  
    [0.4, 0.4, 0.2]  
])

final_prob = np.array([
    [0.2, 0.6, 0.2],  
    [0.1, 0.4, 0.5], 
    [0.0, 0.1, 0.9]   
])


skill_path = simulate_skill_path(static_prob , static_prob, T, seed=42)
plot_values("Static Agent",skill_path)

skill_path = simulate_skill_path(init_prob , final_prob, T, seed=42)
plot_values("Dynamic Agent",skill_path)


title=f"Cumulative Discounted Value for \n Different Employees Gamma= {gamma:.0f}"
plt.title(title)
plt.xlabel("Time")
plt.ylabel("Cumulative Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

