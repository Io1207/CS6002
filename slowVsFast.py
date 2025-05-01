import matplotlib.pyplot as plt
import numpy as np

# Time settings
T = 10  # number of periods
delta = 0.9  # discount factor
time = np.arange(T)

# Skill growth paths for two employees
# Fast learner: Basic → Intermediate → Advanced quickly
fast_growth=[3]*T
i=0
for i in range(T//10):
    fast_growth[i]=1
for i in range(i,T//5):
    fast_growth[i]=2

# Slow learner: Stays at Basic longer
slow_growth = [3]*T
i=0
for i in range(T//5):
    fast_growth[i]=1
for i in range(i,T//2):
    fast_growth[i]=2

# Value function for skill levels
def skill_value(level):
    return {1: 1, 2: 2, 3: 3}[level]

# Compute discounted values
def discounted_values(skill_path):
    return np.array([delta**t * skill_value(skill_path[t]) for t in time])

# Compute cumulative discounted value
def cumulative_discounted(values):
    return np.cumsum(values)

# Get values
fast_values = discounted_values(fast_growth)
slow_values = discounted_values(slow_growth)

fast_cum = cumulative_discounted(fast_values)
slow_cum = cumulative_discounted(slow_values)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(time, fast_values, label='Fast Learner (Per Period Value)')
plt.plot(time, slow_values, linestyle='dotted', label='Slow Learner (Per Period Value)')
plt.plot(time, fast_cum,linestyle='--', label='Fast Learner (Cumulative Value)')
plt.plot(time, slow_cum,linestyle='--', label='Slow Learner (Cumulative Value)')
plt.xticks(time)
plt.xlabel('Time Period')
plt.ylabel('Value')
plt.title('Comparing Dynamic Skill Value: Fast vs Slow Learner')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
