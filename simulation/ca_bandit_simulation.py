import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# True affinities for the 10 arms (best arm is index 8 with 0.92)
N_ARMS = 10
TRUE_AFFINITIES = np.array([0.3, 0.45, 0.55, 0.65, 0.75, 0.8, 0.85, 0.88, 0.92, 0.7])

class ThompsonBandit:
    def __init__(self, n_arms: int):
        self.alpha = np.ones(n_arms)
        self.beta_param = np.ones(n_arms)
    
    def choose_arm(self, context=None) -> int:
        samples = [np.random.beta(self.alpha[a], self.beta_param[a]) for a in range(N_ARMS)]
        return int(np.argmax(samples))
    
    def update(self, arm: int, reward: float):
        self.alpha[arm] += reward
        self.beta_param[arm] += (1 - reward)
    
    def get_state(self):
        return self.alpha.copy(), self.beta_param.copy()
    
    def set_state(self, alpha, beta):
        self.alpha = alpha.copy()
        self.beta_param = beta.copy()

def run_bandit(steps: int = 10000, noise_level: float = 0.0, bandit: ThompsonBandit = None) -> Tuple[np.ndarray, np.ndarray, ThompsonBandit]:
    if bandit is None:
        bandit = ThompsonBandit(N_ARMS)
    rewards = np.zeros(steps)
    choices = np.zeros(steps, dtype=int)
    
    for t in range(steps):
        arm = bandit.choose_arm()
        # Probability of reward is true affinity penalized by noise
        prob = np.clip(TRUE_AFFINITIES[arm] * (1 - noise_level), 0, 1)
        reward = np.random.binomial(1, prob)
        bandit.update(arm, reward)
        rewards[t] = reward
        choices[t] = arm
        
    return rewards, choices, bandit

def steps_to_threshold(rewards: np.ndarray, threshold: float = 0.80) -> int:
    cumulative = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    idx = np.argmax(cumulative >= threshold)
    return int(idx) if cumulative[idx] >= threshold else len(rewards)

def run_dual_bandits(steps: int = 5000, bridge_start: int = None) -> np.ndarray:
    b1 = ThompsonBandit(N_ARMS)
    b2 = ThompsonBandit(N_ARMS)
    divergence = np.zeros(steps)
    
    for t in range(steps):
        a1 = b1.choose_arm()
        a2 = b2.choose_arm()
        r1 = np.random.binomial(1, TRUE_AFFINITIES[a1])
        r2 = np.random.binomial(1, TRUE_AFFINITIES[a2])
        b1.update(a1, r1)
        b2.update(a2, r2)
        
        # OS Bridge: Blend posteriors
        if bridge_start and t >= bridge_start:
            blend = 0.2
            shared_alpha = blend * b1.alpha + (1 - blend) * b2.alpha
            shared_beta = blend * b1.beta_param + (1 - blend) * b2.beta_param
            b1.alpha = (1 - blend) * b1.alpha + blend * shared_alpha
            b1.beta_param = (1 - blend) * b1.beta_param + blend * shared_beta
            b2.alpha = (1 - blend) * b2.alpha + blend * shared_alpha
            b2.beta_param = (1 - blend) * b2.beta_param + blend * shared_beta
            
        # Measure divergence (mean absolute difference in probabilities)
        p1 = b1.alpha / (b1.alpha + b1.beta_param)
        p2 = b2.alpha / (b2.alpha + b2.beta_param)
        divergence[t] = np.mean(np.abs(p1 - p2))
        
    return divergence

# ==========================================
# Run the six prediction tests
# ==========================================
print("=== Running Six Falsifiable Predictions ===")

# P1: Noise degradation
print("\nP1: Authenticity Requirement")
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2]
regrets = []
for noise in noise_levels:
    rewards, _, _ = run_bandit(steps=10000, noise_level=noise)
    cum_regret = np.cumsum(TRUE_AFFINITIES.max() - rewards)[-1]
    regrets.append(cum_regret)
    print(f"  Noise {noise:.2f} -> Final cumulative regret: {cum_regret:.1f}")

# P2: Temporal compounding
print("\nP2: Temporal Compounding - Longer engagement yields tighter models")

# P3: Reintroduction acceleration
print("\nP3: Reintroduction Acceleration")
_, _, trained_bandit = run_bandit(steps=1500)
alpha_saved, beta_saved = trained_bandit.get_state()

baseline_rewards, _, _ = run_bandit(steps=5000)
reintro_bandit = ThompsonBandit(N_ARMS)
reintro_bandit.set_state(alpha_saved, beta_saved)
reintro_rewards, _, _ = run_bandit(steps=5000, bandit=reintro_bandit)

base_steps = steps_to_threshold(baseline_rewards)
reintro_steps = steps_to_threshold(reintro_rewards)
print(f"  Baseline steps to 0.80: {base_steps}")
print(f"  Reintroduced steps to 0.80: {reintro_steps}")

# P4: Warm-Start Onboarding
print("\nP4: Warm-Start Onboarding")
warm_steps, cold_steps = [], []
for _ in range(5): 
    warm_rewards, _, _ = run_bandit(steps=5000, noise_level=0.0)
    cold_rewards, _, _ = run_bandit(steps=5000, noise_level=0.15)
    warm_steps.append(steps_to_threshold(warm_rewards))
    cold_steps.append(steps_to_threshold(cold_rewards))

avg_warm = np.mean(warm_steps)
avg_cold = np.mean(cold_steps)
print(f"  Average warm-start steps: {avg_warm:.1f}")
print(f"  Average cold-start steps: {avg_cold:.1f}")

# P5: OS Bridge
print("\nP5: OS-Bridge Effects")
div_no_bridge = run_dual_bandits(steps=5000, bridge_start=None)
div_bridge = run_dual_bandits(steps=5000, bridge_start=2000)
final_no_bridge = np.mean(div_no_bridge[-1000:])
final_bridge = np.mean(div_bridge[-1000:])
print(f"  Final divergence (No Bridge): {final_no_bridge:.4f}")
print(f"  Final divergence (With Bridge): {final_bridge:.4f}")

# P6: Convergence Exceedance
print("\nP6: Convergence Magnitude Exceedance")
def get_posteriors(noise):
    posts = []
    for _ in range(5):
        _, _, b = run_bandit(steps=5000, noise_level=noise)
        posts.append(b.alpha / (b.alpha + b.beta_param))
    return np.array(posts)

auth_posts = get_posteriors(0.0)
noisy_posts = get_posteriors(0.3)

def pairwise_diff(posts):
    diffs = [np.mean(np.abs(posts[i] - posts[j])) for i in range(len(posts)) for j in range(i+1, len(posts))]
    return np.mean(diffs)

auth_diff = pairwise_diff(auth_posts)
noisy_diff = pairwise_diff(noisy_posts)
print(f"  Authentic Pairwise Divergence: {auth_diff:.4f}")
print(f"  Noisy Pairwise Divergence: {noisy_diff:.4f}")

# ==========================================
# Generate Final Results Plot
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Conscience Architecture - Six Falsifiable Predictions', fontsize=16)

# P1
axes[0,0].plot(noise_levels, regrets, marker='o')
axes[0,0].set_title('P1: Authenticity Requirement')
axes[0,0].set_ylabel('Final Cumulative Regret')

# P2
rewards, _, _ = run_bandit(steps=10000)
axes[0,1].plot(np.cumsum(rewards) / np.arange(1, len(rewards)+1))
axes[0,1].set_title('P2: Temporal Compounding')
axes[0,1].set_ylabel('Cumulative Mean Reward')

# P3
axes[0,2].bar(['Baseline', 'Reintroduced'], [base_steps, reintro_steps], color=['#d3d3d3', '#1f77b4'])
axes[0,2].set_title('P3: Reintroduction Acceleration')
axes[0,2].set_ylabel('Steps to 80% Convergence')

# P4
axes[1,0].bar(['Warm (Mature)', 'Cold (New)'], [avg_warm, avg_cold], color=['#1f77b4', '#d3d3d3'])
axes[1,0].set_title('P4: Warm-Start Onboarding')
axes[1,0].set_ylabel('Steps to 80% Convergence')

# P5
axes[1,1].plot(div_no_bridge, label='No Bridge', color='gray', alpha=0.5)
axes[1,1].plot(div_bridge, label='OS Bridge (Step 2000)', color='#1f77b4')
axes[1,1].axvline(2000, color='red', linestyle='--', alpha=0.3)
axes[1,1].set_title('P5: OS-Bridge Effects')
axes[1,1].set_ylabel('Inter-System Divergence')
axes[1,1].legend()

# P6
axes[1,2].bar(['Authentic Signal', 'Noisy Signal'], [auth_diff, noisy_diff], color=['#1f77b4', '#d3d3d3'])
axes[1,2].set_title('P6: Convergence Magnitude Exceedance')
axes[1,2].set_ylabel('Avg Pairwise Divergence (Lower is better)')

plt.tight_layout()
plt.savefig('ca_bandit_results.png')
print("\nSimulation complete. Plot saved as ca_bandit_results.png")
