import numpy as np
import matplotlib.pyplot as plt

T = 10
time = np.arange(T)

skill_values = {1: 1, 2: 2, 3: 3}

transition_probs = {
    1: 0.4,
    2: 0.3,
    3: 0.3 
}

def simulate_skill_path(promo_probs, T=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    skills = [1]  # start at Basic
    for t in range(1, T):
        current = skills[-1]
        if current < 3 and np.random.rand() < promo_probs[current]:
            skills.append(current + 1)
        else:
            skills.append(current)
    return skills

skill_path = simulate_skill_path(transition_probs, T, seed=42)

discount_factors = {
    "Short-term": 0.5,
    "Medium-term": 0.8,
    "Long-term": 0.97
}

plt.figure(figsize=(10, 6))
for label, delta in discount_factors.items():
    discounted_values = [delta**t * skill_values[skill] for t, skill in enumerate(skill_path)]
    cumulative_values = np.cumsum(discounted_values)
    plt.plot(time, cumulative_values, label=label)

plt.plot(time, [skill_values[s] for s in skill_path], 'k--', alpha=0.4, label="Skill Level (Raw Value)")

plt.title("Cumulative Discounted Value under Different Company Orientations")
plt.xlabel("Time")
plt.ylabel("Cumulative Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
