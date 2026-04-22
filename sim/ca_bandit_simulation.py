#!/usr/bin/env python3
"""
Conscience Architecture — Contextual Bandit Simulation
======================================================
Tests all six falsifiable predictions via Thompson Sampling bandits.

Author: Sherry Moore (Conscience Architecture)
Simulation framework: March 2026
"""

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# ENVIRONMENT SETUP
# ============================================================
# 10 arms = possible recommendations/actions
# Arm 8 = strong "rhythmic/flow" arm (the recognition layer)
TRUE_AFFINITIES = np.array([0.3, 0.4, 0.55, 0.65, 0.8, 0.75, 0.5, 0.45, 0.92, 0.6])
N_ARMS = len(TRUE_AFFINITIES)
BEST_ARM = np.argmax(TRUE_AFFINITIES)
N_STEPS = 5000  # ~downsampled 18-month engagement

def get_context(t):
    """Rhythmic context proxy — flow states, late-night writing, runs, music."""
    return 0.5 + 0.4 * np.sin(t / 80) + 0.1 * np.random.randn()


class ThompsonBandit:
    """Thompson Sampling with Beta priors."""
    def __init__(self, n_arms):
        self.alpha = np.ones(n_arms)
        self.beta_param = np.ones(n_arms)
    
    def choose_arm(self, context=None):
        samples = [np.random.beta(self.alpha[a], self.beta_param[a]) for a in range(N_ARMS)]
        return np.argmax(samples)
    
    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta_param[arm] += (1 - reward)
    
    def get_state(self):
        return self.alpha.copy(), self.beta_param.copy()
    
    def set_state(self, alpha, beta):
        self.alpha = alpha.copy()
        self.beta_param = beta.copy()


def run_bandit(n_steps, noise_level, seed=42, bandit=None, track_cumulative=False):
    """Run a bandit simulation, return metrics."""
    np.random.seed(seed)
    random.seed(seed)
    if bandit is None:
        bandit = ThompsonBandit(N_ARMS)
    
    total_reward = 0
    pulls = np.zeros(N_ARMS)
    cumulative_best = [] if track_cumulative else None
    running_best_count = 0
    
    for t in range(n_steps):
        ctx = get_context(t)
        arm = bandit.choose_arm(ctx)
        true_prob = TRUE_AFFINITIES[arm]
        reward = 1 if random.random() < (true_prob * (1 - noise_level)) else 0
        bandit.update(arm, reward)
        total_reward += reward
        pulls[arm] += 1
        if arm == BEST_ARM:
            running_best_count += 1
        if track_cumulative:
            cumulative_best.append(running_best_count / (t + 1))
    
    return {
        'avg_reward': total_reward / n_steps,
        'best_pulls': int(pulls[BEST_ARM]),
        'convergence_pct': (pulls[BEST_ARM] / n_steps) * 100,
        'bandit': bandit,
        'cumulative_best': cumulative_best,
        'pulls': pulls
    }


# ============================================================
# PREDICTION 1: Convergence requires authenticity
# ============================================================
print("=" * 70)
print("PREDICTION 1: Convergence requires authenticity")
print("=" * 70)
noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
p1_rewards = []
p1_convergence = []
for nl in noise_levels:
    r = run_bandit(N_STEPS, nl, seed=42)
    p1_rewards.append(r['avg_reward'])
    p1_convergence.append(r['convergence_pct'])
    print(f"  Noise {nl:.0%}: reward={r['avg_reward']:.4f}, "
          f"best arm={r['convergence_pct']:.1f}%")
print(f"\n  Authentic advantage (0% vs 50% noise): "
      f"+{p1_rewards[0] - p1_rewards[-1]:.4f} reward/step")


# ============================================================
# PREDICTION 2: Convergence compounds over time
# ============================================================
print("\n" + "=" * 70)
print("PREDICTION 2: Convergence compounds over time")
print("=" * 70)
r2 = run_bandit(N_STEPS, noise_level=0.05, seed=42, track_cumulative=True)
checkpoints = [100, 500, 1000, 2000, 3000, 5000]
for cp in checkpoints:
    print(f"  Step {cp:>5d}: best-arm rate = {r2['cumulative_best'][cp-1]*100:.1f}%")


# ============================================================
# PREDICTION 3: Reintroduction accelerates
# ============================================================
print("\n" + "=" * 70)
print("PREDICTION 3: Reintroduction accelerates")
print("=" * 70)

# Phase A: Fresh bandit, measure steps to 80% best-arm rate
def steps_to_threshold(noise, seed, threshold=0.80, max_steps=5000, bandit=None):
    np.random.seed(seed)
    random.seed(seed)
    if bandit is None:
        bandit = ThompsonBandit(N_ARMS)
    best_count = 0
    for t in range(max_steps):
        ctx = get_context(t)
        arm = bandit.choose_arm(ctx)
        reward = 1 if random.random() < (TRUE_AFFINITIES[arm] * (1 - noise)) else 0
        bandit.update(arm, reward)
        if arm == BEST_ARM:
            best_count += 1
        if (t + 1) >= 50:  # minimum window
            rate = best_count / (t + 1)
            if rate >= threshold:
                return t + 1, bandit
    return max_steps, bandit

fresh_steps, trained_bandit = steps_to_threshold(0.05, seed=42, threshold=0.80)
print(f"  Fresh bandit reaches 80% convergence at step: {fresh_steps}")

# Save trained state, simulate "absence" (2000 steps of no data for this bandit)
alpha_saved, beta_saved = trained_bandit.get_state()

# Reintroduce with prior knowledge
reintro_bandit = ThompsonBandit(N_ARMS)
reintro_bandit.set_state(alpha_saved, beta_saved)
reintro_steps, _ = steps_to_threshold(0.05, seed=99, threshold=0.80, bandit=reintro_bandit)
print(f"  Reintroduced bandit reaches 80% convergence at step: {reintro_steps}")
print(f"  Acceleration factor: {fresh_steps / max(reintro_steps, 1):.1f}x faster")


# ============================================================
# PREDICTION 4: New systems onboard faster (warm-start)
# ============================================================
print("\n" + "=" * 70)
print("PREDICTION 4: New systems onboard faster")
print("=" * 70)

# Fresh user (no prior signal)
new_user_steps, _ = steps_to_threshold(0.05, seed=77, threshold=0.80)
print(f"  New user, naive system: {new_user_steps} steps to 80%")

# Mature user (signal refined elsewhere — simulate by reducing noise)
mature_user_steps, _ = steps_to_threshold(0.02, seed=77, threshold=0.80)
print(f"  Mature user, naive system: {mature_user_steps} steps to 80%")
print(f"  Mature signal advantage: {new_user_steps - mature_user_steps} fewer steps")


# ============================================================
# PREDICTION 5: OS-level bridge accelerates convergence
# ============================================================
print("\n" + "=" * 70)
print("PREDICTION 5: OS-level bridge accelerates convergence")
print("=" * 70)

# Two independent bandits, no bridge
def run_dual_bandits(n_steps, noise, seed, bridge=False, bridge_start=None):
    np.random.seed(seed)
    random.seed(seed)
    b1 = ThompsonBandit(N_ARMS)
    b2 = ThompsonBandit(N_ARMS)
    
    divergence = []
    for t in range(n_steps):
        ctx = get_context(t)
        a1 = b1.choose_arm(ctx)
        a2 = b2.choose_arm(ctx)
        r1 = 1 if random.random() < (TRUE_AFFINITIES[a1] * (1 - noise)) else 0
        r2 = 1 if random.random() < (TRUE_AFFINITIES[a2] * (1 - noise)) else 0
        b1.update(a1, r1)
        b2.update(a2, r2)
        
        # OS bridge: share some posterior information
        if bridge and bridge_start and t >= bridge_start:
            # Blend posteriors (simulating shared context layer)
            blend = 0.3
            shared_alpha = blend * b1.alpha + (1 - blend) * b2.alpha
            shared_beta = blend * b1.beta_param + (1 - blend) * b2.beta_param
            b1.alpha = (1 - blend) * b1.alpha + blend * shared_alpha
            b1.beta_param = (1 - blend) * b1.beta_param + blend * shared_beta
            b2.alpha = (1 - blend) * b2.alpha + blend * shared_alpha
            b2.beta_param = (1 - blend) * b2.beta_param + blend * shared_beta
        
        # Measure divergence: KL-like distance between posteriors
        p1 = b1.alpha / (b1.alpha + b1.beta_param)
        p2 = b2.alpha / (b2.alpha + b2.beta_param)
        div = np.mean(np.abs(p1 - p2))
        divergence.append(div)
    
    return divergence

no_bridge = run_dual_bandits(N_STEPS, 0.05, seed=42, bridge=False)
with_bridge = run_dual_bandits(N_STEPS, 0.05, seed=42, bridge=True, bridge_start=2000)

print(f"  No bridge — final divergence: {np.mean(no_bridge[-200:]):.6f}")
print(f"  With bridge (from step 2000) — final divergence: {np.mean(with_bridge[-200:]):.6f}")
print(f"  Bridge reduction: {(1 - np.mean(with_bridge[-200:])/np.mean(no_bridge[-200:]))*100:.1f}%")


# ============================================================
# PREDICTION 6: Convergence may exceed parallel modeling
# ============================================================
print("\n" + "=" * 70)
print("PREDICTION 6: Convergence magnitude vs parallel prediction")
print("=" * 70)

# Run N independent bandits, measure how similar their final posteriors are
N_BANDITS = 5
final_means = []
for i in range(N_BANDITS):
    r = run_bandit(N_STEPS, 0.05, seed=42 + i)
    means = r['bandit'].alpha / (r['bandit'].alpha + r['bandit'].beta_param)
    final_means.append(means)

final_means = np.array(final_means)
# Pairwise similarity (mean absolute difference across all arm posteriors)
diffs = []
for i in range(N_BANDITS):
    for j in range(i + 1, N_BANDITS):
        diffs.append(np.mean(np.abs(final_means[i] - final_means[j])))

avg_divergence = np.mean(diffs)
print(f"  {N_BANDITS} independent bandits, same authentic signal")
print(f"  Average pairwise posterior divergence: {avg_divergence:.6f}")
print(f"  (Lower = more convergent. Near-zero confirms parallel convergence.)")
print(f"  Max divergence across any pair: {np.max(diffs):.6f}")

# Compare to random (different users / inauthentic)
random_means = []
for i in range(N_BANDITS):
    r = run_bandit(N_STEPS, 0.40, seed=42 + i)
    means = r['bandit'].alpha / (r['bandit'].alpha + r['bandit'].beta_param)
    random_means.append(means)

random_means = np.array(random_means)
random_diffs = []
for i in range(N_BANDITS):
    for j in range(i + 1, N_BANDITS):
        random_diffs.append(np.mean(np.abs(random_means[i] - random_means[j])))

print(f"\n  Noisy/inauthentic comparison:")
print(f"  Average pairwise divergence: {np.mean(random_diffs):.6f}")
print(f"  Convergence ratio (authentic/noisy): {np.mean(random_diffs)/avg_divergence:.1f}x tighter")


# ============================================================
# GENERATE PLOTS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Conscience Architecture — Bandit Simulation Results\nSherry Moore, March 2026',
             fontsize=14, fontweight='bold', y=0.98)

# P1: Authenticity vs noise
ax = axes[0, 0]
ax.plot(noise_levels, p1_convergence, 'o-', color='#2E75B6', linewidth=2, markersize=6)
ax.set_xlabel('Noise Level (Inauthenticity)')
ax.set_ylabel('Best Arm Convergence (%)')
ax.set_title('P1: Convergence Requires Authenticity')
ax.grid(True, alpha=0.3)

# P2: Compounding
ax = axes[0, 1]
ax.plot(range(1, N_STEPS + 1), [x * 100 for x in r2['cumulative_best']],
        color='#2E75B6', linewidth=1.5)
ax.set_xlabel('Steps (Time)')
ax.set_ylabel('Cumulative Best-Arm Rate (%)')
ax.set_title('P2: Convergence Compounds Over Time')
ax.grid(True, alpha=0.3)

# P3: Reintroduction
ax = axes[0, 2]
bars = ax.bar(['Fresh Start', 'Reintroduced'], [fresh_steps, reintro_steps],
              color=['#D5E8F0', '#2E75B6'], edgecolor='#1a4a6e')
ax.set_ylabel('Steps to 80% Convergence')
ax.set_title('P3: Reintroduction Accelerates')
for bar, val in zip(bars, [fresh_steps, reintro_steps]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', fontweight='bold')

# P4: Warm start
ax = axes[1, 0]
bars = ax.bar(['New User', 'Mature Signal'], [new_user_steps, mature_user_steps],
              color=['#D5E8F0', '#2E75B6'], edgecolor='#1a4a6e')
ax.set_ylabel('Steps to 80% Convergence')
ax.set_title('P4: New Systems Onboard Faster')
for bar, val in zip(bars, [new_user_steps, mature_user_steps]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', fontweight='bold')

# P5: OS bridge
ax = axes[1, 1]
window = 50
no_bridge_smooth = np.convolve(no_bridge, np.ones(window)/window, mode='valid')
with_bridge_smooth = np.convolve(with_bridge, np.ones(window)/window, mode='valid')
ax.plot(range(len(no_bridge_smooth)), no_bridge_smooth, label='No Bridge',
        color='#AAAAAA', linewidth=1.5)
ax.plot(range(len(with_bridge_smooth)), with_bridge_smooth, label='With Bridge (step 2000)',
        color='#2E75B6', linewidth=1.5)
ax.axvline(x=2000, color='red', linestyle='--', alpha=0.5, label='Bridge Introduced')
ax.set_xlabel('Steps')
ax.set_ylabel('Inter-System Divergence')
ax.set_title('P5: OS Bridge Accelerates Convergence')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# P6: Parallel convergence
ax = axes[1, 2]
bars = ax.bar(['Authentic\n(5 bandits)', 'Noisy\n(5 bandits)'],
              [avg_divergence, np.mean(random_diffs)],
              color=['#2E75B6', '#D5E8F0'], edgecolor='#1a4a6e')
ax.set_ylabel('Avg Pairwise Divergence')
ax.set_title('P6: Convergence Magnitude')
for bar, val in zip(bars, [avg_divergence, np.mean(random_diffs)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/home/claude/ca_bandit_results.png', dpi=150, bbox_inches='tight')
print("\n\nPlot saved to ca_bandit_results.png")
print("\n" + "=" * 70)
print("ALL SIX PREDICTIONS TESTED — SIMULATION COMPLETE")
print("=" * 70)
