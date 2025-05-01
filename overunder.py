import numpy as np
import matplotlib.pyplot as plt

n_agents = 3
m_skills = 3
T = 20
b_t = 300
alpha = np.array([0.5, 0.3, 0.2]) 
gamma = 0.1

true_skills = np.array([
    [3, 2, 2],
    [2, 3, 2],
    [1, 1, 1]
], dtype=float)

transition_matrix=np.array([[0.4,0.5,0.05],[0.2,0.5,0.4],[0.2,0.7,0.1]],dtype=float)

np.random.seed(42)
skill_growth_std = 0.1
true_skills_over_time = np.zeros((T, n_agents, m_skills))
true_skills_over_time[0] = true_skills

for t in range(1, T):
    noise = np.random.normal(0, skill_growth_std, size=(n_agents, m_skills))
    true_skills_over_time[t] = np.clip(true_skills_over_time[t-1] @ transition_matrix, 1, 3)


reported_skills_over_time = true_skills_over_time.copy()
reported_skills_over_time[:, 1, :]=np.clip(reported_skills_over_time[:, 1, :] -.5, 1, 3) #under reporting
reported_skills_over_time[:, 0, :]=np.clip(reported_skills_over_time[:, 1, :] + .5, 1, 3) #over reporting

# def compute_skill_score(skills, prior=np.array([2, 2, 2])):
#     concave_score = np.log1p(skills) @ alpha
#     penalty = np.sum((skills - prior)**2, axis=0) 
#     return concave_score - 0.1 * penalty

def compute_skill_score(skills):
    return skills @ alpha

true_values = np.zeros((T, n_agents))
reported_values = np.zeros((T, n_agents))
allocations = np.zeros((T, n_agents))

for k in range(9):
    gamma=0.1*(k+1)
    for t in range(T):
        for i in range(n_agents):
            scores = compute_skill_score(reported_skills_over_time[t][i])
            proportions = scores / np.sum(scores)
            allocations[t] = proportions * b_t
            true_score = compute_skill_score(true_skills_over_time[t][i])
            reported_score = scores
            true_values[t, i] = true_score 
            reported_values[t, i] = reported_score 

    discounts = np.array([gamma**t for t in range(T)])
    true_util = np.sum(true_values * discounts[:, None], axis=0)
    reported_util = np.sum(reported_values * discounts[:, None], axis=0)

    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    x = np.arange(n_agents)
    plt.bar(x - bar_width/2, true_util, width=bar_width, label='True Utility')
    plt.bar(x + bar_width/2, reported_util, width=bar_width, label='Reported Utility')
    plt.xticks(x, [f"Agent {i+1}" for i in x])
    plt.ylabel("Discounted Total Utility")
    title="Effect of Misreporting in DPM, Gamma"+str(gamma)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    name=str(gamma)+".png"
    plt.savefig(name)
    plt.show()
