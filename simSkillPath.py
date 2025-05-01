import numpy as np
import matplotlib.pyplot as plt
import random

T = 100
time = np.arange(T)

skill_values = {1: 1, 2: 2, 3: 3}

transition_probs = [
    {},
    {1: 0.6, 2: 0.3, 3: 0.1},
    {1: 0.1, 2: 0.6, 3: 0.3},
    {1: 0.1, 2: 0.3, 3: 0.6},
]

def simulate_skill_path(promo_probs, T=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    skill_values = [1]  # start at Basic
    for t in range(1, T):
        current = skill_values[-1]
        skills   = list(transition_probs[current].keys())
        weights  = list(transition_probs[current].values())
        new_level = random.choices(skills, weights=weights, k=1)[0]
        skill_values.append(new_level)
    return skill_values

skill_path = simulate_skill_path(transition_probs, T, seed=42)

discount_factors = {
    "Short-term": 0.8,
    "Medium-term": 0.9,
    "Long-term": 0.97
}

discount_factors={str(k*0.05):0.05*k for k in range(10,20)}

plt.figure(figsize=(12, 6))
for label, delta in discount_factors.items():
    discounted_values = [delta**t * skill_values[skill] for t, skill in enumerate(skill_path)]
    cumulative_values = np.cumsum(discounted_values)
    plt.plot(time, cumulative_values, label=label)

plt.plot(time, [skill_values[s] for s in skill_path], 'k--', alpha=0.4, label="Skill Level (Raw Value)")

plt.title("Cumulative Discounted Value under Different Company Orientations")
plt.xlabel("Time")
plt.ylabel("Cumulative Value")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()
