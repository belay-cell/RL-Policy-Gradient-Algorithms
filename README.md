# RL Policy Gradient Algorithms

**Implementation of REINFORCE, A2C, and DDPG** using PyTorch and Gymnasium for classic control tasks.

> Professional refactoring of KAIST CS377 Reinforcement Learning coursework into production-ready code.

## 🎯 Overview

This repository contains modular, well-documented implementations of three fundamental policy gradient algorithms:

- **REINFORCE** (Vanilla + Baseline): Monte Carlo policy gradient
- **A2C** (Advantage Actor-Critic): Synchronous actor-critic with online learning  
- **DDPG** (Deep Deterministic Policy Gradient): Off-policy for continuous control

### Environments

| Environment | Action Space | Obs Space | Algorithms |
|------------|--------------|-----------|------------|
| **CartPole-v1** | Discrete (2) | Continuous (4D) | REINFORCE, A2C |
| **Pendulum-v1** | Continuous [-2, 2] | Continuous (3D) | DDPG |

## 📈 Results (From Actual Training)

### Training Performance

| Algorithm | Environment | Episodes to Solve | Final Avg Reward |
|-----------|-------------|-------------------|------------------|
| **REINFORCE (Vanilla)** | CartPole-v1 | **262** | **450.80** |
| **REINFORCE (Baseline)** | CartPole-v1 | **495** | **461.60** |
| **A2C** | CartPole-v1 | **699** | **453.50** |
| **DDPG** | Pendulum-v1 | **172** | **-195.39** |

### Key Insights

✅ **REINFORCE vanilla solved fastest** (262 episodes) - simple but effective  
✅ **Baseline version** reduced variance significantly  
✅ **A2C** enables online learning without episode waits  
✅ **DDPG** efficiently handles continuous actions with replay buffer

## 🛠️ Installation

```bash
git clone https://github.com/belay-cell/RL-Policy-Gradient-Algorithms.git
cd RL-Policy-Gradient-Algorithms
pip install -r requirements.txt
```

## 🚀 Usage

### Train REINFORCE on CartPole

```python
import gymnasium as gym
from models import SoftmaxPolicy
from reinforce import REINFORCE

env = gym.make("CartPole-v1")
policy = SoftmaxPolicy(env.observation_space.shape[0], env.action_space.n)
agent = REINFORCE(policy, lr=1e-3, solve_criteria=450, episode_limit=1000)
total_rewards = agent.train(env)
```

### Train DDPG on Pendulum

```python
import gymnasium as gym
from models import DDPGActor, DDPGCritic
from utils import ReplayBuffer
from ddpg import DDPG

env = gym.make("Pendulum-v1")
actor = DDPGActor(3, 1, action_range=2.0)
critic = DDPGCritic(3, 1)
buffer = ReplayBuffer(10000)
agent = DDPG(buffer, actor, critic, actor_lr=1e-4, critic_lr=1e-3)
total_rewards = agent.train(env)
```

## 📚 Project Structure

```
RL-Policy-Gradient-Algorithms/
├── models.py          # Neural network architectures
├── reinforce.py       # REINFORCE (vanilla + baseline)
├── a2c.py             # Advantage Actor-Critic  
├── ddpg.py            # Deep Deterministic Policy Gradient
├── utils.py           # Replay buffer utilities
├── requirements.txt   # Dependencies
├── README.md         # This file
└── LICENSE           # MIT License
```

## 🏗️ Architecture

### Neural Networks

**SoftmaxPolicy** (Discrete Actions - CartPole)
```
state (4D) → FC(128) → ReLU → FC(64) → ReLU → FC(2) → Softmax → action_probs
```

**Critic** (Value Function - A2C)
```
state → FC(128) → ReLU → FC(64) → ReLU → FC(1) → value
```

**DDPGActor** (Continuous Actions - Pendulum)
```
state (3D) → FC(128) → ReLU → FC(64) → ReLU → FC(1) → Tanh → action * 2.0
```

**DDPGCritic** (State-Action Value)
```
concat(state, action) → FC(128) → ReLU → FC(64) → ReLU → FC(1) → Q-value
```

## 🔬 Algorithm Details

### REINFORCE

**Policy Gradient Theorem:**
```
∇θ J(θ) = 𝔼[∇θ log πθ(a|s) · G_t]
```

**With Baseline:**
```
θ ← θ + α ∑_t (G_t - baseline) ∇θ log πθ(a_t|s_t)
```

### A2C

**Advantage Function:**
```
A(s_t, a_t) = R_{t+1} + γV(s_{t+1}) - V(s_t)
```

**Update Rules:**
```
Actor:  θπ ← θπ + α ∇θπ[A · log π(a|s)]
Critic: θv ← θv - α ∇θv[V(s) - (R + γV(s'))]^2
```

### DDPG

**Key Features:**
- Deterministic policy: `a = μ(s)`
- Experience replay (capacity: 10,000)
- Target networks with soft updates (τ=0.005)
- Gaussian exploration noise

## 📊 Hyperparameters

| Parameter | REINFORCE | A2C | DDPG |
|-----------|-----------|-----|------|
| Learning Rate | 1e-3 | Actor:1e-3, Critic:1e-3 | Actor:1e-4, Critic:1e-3 |
| Discount (γ) | 0.99 | 0.99 | 0.99 |
| Batch Size | Full episode | 1 (online) | 32 |
| Replay Buffer | - | - | 10,000 |
| Soft Update (τ) | - | - | 0.005 |
| Solve Criteria | 450 | 450 | -200 |

## 🧪 Technical Highlights

- ✅ **Type Hints**: Full Python type annotations
- ✅ **Modular Design**: Separate files for each algorithm
- ✅ **Reproducible**: Fixed seed (54321) for consistent results
- ✅ **Efficient**: Vectorized PyTorch operations
- ✅ **Clean Code**: PEP 8 compliant
- ✅ **Progress Tracking**: tqdm progress bars with live stats

## 📝 Key Learnings

1. **Variance Matters**: Baseline in REINFORCE dramatically stabilizes training
2. **Online vs Batch**: A2C updates every step vs REINFORCE waits for full episodes
3. **Exploration**: DDPG needs Gaussian noise since policy is deterministic
4. **Stability**: Target networks in DDPG prevent Q-value divergence

## 🔗 References

- Sutton & Barto. *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 13
- Mnih et al. *Asynchronous Methods for Deep Reinforcement Learning* (2016)
- Lillicrap et al. *Continuous Control with Deep Reinforcement Learning* (2015)

## 💼 Portfolio Showcase

**This project demonstrates:**

- ✅ Deep RL algorithm implementation from scratch
- ✅ PyTorch proficiency for neural network development  
- ✅ Clean code architecture and software engineering best practices
- ✅ Technical documentation and reproducible research
- ✅ MLOps readiness for production deployment

> *Originally developed for KAIST CS377 Reinforcement Learning (Student ID: 20220934), refactored into professional-grade codebase.*

## 🛡️ License

MIT License - See [LICENSE](LICENSE)

## 🚀 Author

**Belay Zeleke**  
GitHub: [@belay-cell](https://github.com/belay-cell)  
Interested in MLOps, Reinforcement Learning, and Production ML Systems

---

**Built with PyTorch 🔥 | Trained on Gymnasium Environments 🎮**
