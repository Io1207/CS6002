import numpy as np
import matplotlib.pyplot as plt

n_agents = 3
m_skills = 3
T = 50
b_t = 300 
alpha = np.array([0.5, 0.3, 0.2])  # skill importance weights
gamma = 0.9  # discount factor

# Initial true skill levels (advanced=3, intermediate=2, basic=1)
true_skills = np.array([
    [3, 2, 2],
    [2, 3, 2],
    [1, 2, 1]
], dtype=float)

# Skill growth (stochastic)
np.random.seed(42)
skill_growth_std = 0.1
true_skills_over_time = np.zeros((T, n_agents, m_skills))
true_skills_over_time[0] = true_skills

for t in range(1, T):
    noise = np.random.normal(0, skill_growth_std, size=(n_agents, m_skills))
    true_skills_over_time[t] = np.clip(true_skills_over_time[t-1] + noise, 1, 3)

# Reported skills: all truthful except agent 2 (index 1) misreports
reported_skills_over_time = true_skills_over_time.copy()
reported_skills_over_time[:, 1, :] -= 0.5  # underreports
reported_skills_over_time[:, 0, :] -= 0.5  # overreports

# Compute skill scores
def compute_skill_score(skills):
    return skills @ alpha

true_values = np.zeros((T, n_agents))
reported_values = np.zeros((T, n_agents))
allocations = np.zeros((T, n_agents))

for t in range(T):
    scores = compute_skill_score(reported_skills_over_time[t])

    # Optimal allocation: give bonus proportional to reported skill scores
    proportions = scores / np.sum(scores)
    allocations[t] = proportions * b_t

    # Compute true and reported values
    for i in range(n_agents):
        true_score = compute_skill_score(true_skills_over_time[t][i])
        reported_score = scores[i]
        true_values[t, i] = true_score * allocations[t, i]
        reported_values[t, i] = reported_score * allocations[t, i]

# Discounted cumulative values
discounts = np.array([gamma**t for t in range(T)])
true_util = np.sum(true_values * discounts[:, None], axis=0)
reported_util = np.sum(reported_values * discounts[:, None], axis=0)



# Plot
plt.figure(figsize=(10, 5))
bar_width = 0.35
x = np.arange(n_agents)
plt.bar(x - bar_width/2, true_util, width=bar_width, label='True Utility')
plt.bar(x + bar_width/2, reported_util, width=bar_width, label='Reported Utility')
plt.xticks(x, [f"Agent {i+1}" for i in x])
plt.ylabel("Discounted Total Utility")
plt.title("Effect of Misreporting in DPM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
