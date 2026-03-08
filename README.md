# Double Inverted Pendulum - Reinforcement Learning

A 2D physics-based reinforcement learning environment built from scratch using pymunk and pygame. A PPO agent is trained to balance two poles on a moving cart simultaneously.

---

### Environment Design

The environment simulates a double inverted pendulum system using pymunk for physics and pygame for visualization.

**Physics Setup:**

- A cart body is constrained to a horizontal track using a `pymunk.GrooveJoint`
- Pole 1 is attached to the cart via a `pymunk.PivotJoint` at the cart's center
- Pole 2 is attached to the tip of Pole 1 via another `pymunk.PivotJoint`
- Gravity is set to -980 units (approximating real-world physics)
- The simulation advances at a fixed timestep of 1/60 seconds per step

**Observation Space (6 values):**

- Cart position normalized by track limit
- Cart velocity normalized by max velocity
- Pole 1 angle (radians, wrapped to [-π, π])
- Pole 1 angular velocity
- Pole 2 angle relative to Pole 1 (radians, wrapped to [-π, π])
- Pole 2 angular velocity

**Action Space:**

- Single continuous value in [-1.0, 1.0]
- Scaled by force magnitude (500 units) and applied horizontally to the cart

**Episode Termination:**

- Either pole angle exceeds 60% of π radians (fallen too far)
- Cart reaches 95% of the track boundary
- Maximum of 1000 steps reached

---

### Reward Function Design

Two reward functions are implemented, selectable via the `reward_type` parameter.

**Baseline Reward:**

```
reward = cos(θ1) + cos(θ2)
```

- `cos(θ1)`: Cosine of Pole 1 angle. Maximum value of 1.0 when perfectly upright, decreasing as the pole tilts
- `cos(θ2)`: Cosine of Pole 2 angle. Same principle for the second pole
- Total maximum reward per step is 2.0 when both poles are perfectly vertical
- This reward is sparse and only signals the upright goal with no guidance on how to achieve stability

**Shaped Reward:**

```
reward = cos(θ1) + cos(θ2) - 0.1 * |cart_x| - 0.01 * (|ω1| + |ω2|) - 0.001 * action²
```

- `cos(θ1) + cos(θ2)`: Core upright bonus, same as baseline
- `-0.1 * |cart_x|`: Center penalty — discourages the cart from drifting to the edges of the track, keeping it recoverable
- `-0.01 * (|ω1| + |ω2|)`: Velocity penalty — penalizes high angular velocities, encouraging smooth and stable pole control rather than frantic oscillation
- `-0.001 * action²`: Action penalty — discourages applying excessive force, promoting energy-efficient and smooth control policies

The shaped reward significantly accelerates learning by providing continuous, meaningful feedback at every timestep rather than only rewarding the binary outcome of staying upright.

---

### How to Run

**Prerequisites:**

- Docker Desktop installed and running
- Git

**Step 1: Clone the repository**

```bash
git clone <your-repo-url>
cd double-pendulum-rl
```

**Step 2: Build the Docker image**

```bash
docker-compose build
```

**Step 3: Train with shaped reward**

```bash
docker-compose run train
```

**Step 4: Train with baseline reward**

```bash
docker-compose run app python train.py --reward_type baseline --timesteps 200000 --save_path models/ppo_baseline.zip
```

**Step 5: Evaluate the trained agent**

```bash
docker-compose run evaluate
```

**Step 6: Generate the reward comparison plot**

```bash
docker-compose run app python plot_results.py
```

**Step 7: Generate GIFs**

```bash
docker-compose run app python generate_gifs.py --initial_model models/ppo_initial.zip --final_model models/ppo_model.zip
```

---

### Project Structure

```
double-pendulum-rl/
├── environment.py        # Custom DoublePendulumEnv gym environment
├── train.py              # PPO training script
├── evaluate.py           # Evaluation and visualization script
├── generate_gifs.py      # GIF recording script
├── plot_results.py       # Learning curve comparison plot
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker service definitions
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable documentation
├── models/               # Saved model weights
├── logs/                 # Training metrics CSV files
├── media/                # GIF recordings
└── reward_comparison.png # Learning curve comparison plot
```

---

### Dependencies

| Package           | Version |
| ----------------- | ------- |
| pygame            | 2.5.2   |
| pymunk            | 6.6.0   |
| stable-baselines3 | 2.3.2   |
| torch             | 2.2.2   |
| pandas            | 2.2.2   |
| matplotlib        | 3.9.0   |
| gymnasium         | 0.29.1  |
| imageio           | 2.34.1  |
| numpy             | 1.26.4  |
