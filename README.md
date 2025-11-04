# MS Thesis: Agile Locomotion & Navigation of Quadrupedal Robots using Learning-based Approaches

**Author:** Fabeha Raheel  
**Supervisor:** Dr. Umar Shahbaz Khan  
**Guidance & Evaluation Committee:** Dr. Hamid Jabbar, Dr. Tahir Habib Nawaz, Dr. Umer Izhar  
**Institution:** National University of Sciences & Technology (NUST)  

---

## 📘 Abstract

Quadrupedal robots have become a popular research avenue in recent years due to their potential to revolutionize various fields, including exploration, search-and-rescue operations and assistance in urban environments. The aim of quadrupedal research is to leverage bio-inspired locomotion capabilities in robots for agile locomotion and navigation in complex, unstructured terrains. However, achieving agile navigation with quadrupedal robots is exceptionally challenging due to the highly non-linear and complex nature of the problem. 

This research aims to develop a **fully-learned, model-free agile locomotion and navigation framework** for low-cost quadrupedal robots such as the Unitree Go1. The proposed approach uses **Deep Reinforcement Learning** to train the robot to execute advanced locomotion maneuvers such as walking, jumping, climbing, and leaping, based on limited perception of the environment and noisy state information.  
The framework is developed and trained in simulation using **NVIDIA Isaac Gym**, leveraging techniques such as **Domain Randomization** and **Curriculum Learning** to minimize the reality gap and ensure robust performance. The overarching goal is to demonstrate that **highly agile locomotion behaviors**, traditionally dependent on sophisticated control pipelines and expensive sensory systems, can instead be achieved through **end-to-end model-free learning** using affordable hardware and minimal sensory input.

---

## 🚀 Research Objective

To develop a **monolithic, end-to-end state-to-action policy** that enables low-cost quadrupedal robots with limited perception capabilities and noisy state information to learn and execute a diverse range of agile locomotion skills, such as:
- Walking
- Climbing
- Jumping
- Leaping

This policy enables **zero-shot generalization** to unseen robot platforms and terrains — demonstrating successful transfer from **Unitree A1 (training)** to **Unitree Go1 (testing)**.

---

## 🧠 Methodology Overview

### 🔹 Learning Approach
- **Deep Reinforcement Learning (DRL)** using the [RSL RL](https://github.com/leggedrobotics/rsl_rl) framework  
- **Teacher–Student Learning (Privileged Learning):**  
  - *Teacher Policy:* trained with privileged terrain and state information  
  - *Student Policy:* trained via **Behavior Cloning** and **DAGGER** (Dataset Aggregation) using only depth perception
- **Domain Randomization** and **Curriculum Learning** for robustness and transferability  

### 🔹 Policy Architecture
- **Depth Encoder:** encodes raw terrain depth images  
- **GRU Network:** captures temporal dependencies  
- **Actor Network:** outputs motor actions (no critic used during deployment)  
- **State Estimator Network:** predicts unobservable privileged states such as linear velocity, friction, and motor strength  

### 🔹 Simulation Environment
- **Simulator:** NVIDIA Isaac Gym (Preview 3 / 4)
- **Programming Language:** Python  
- **Frameworks:** PyTorch, RSL RL, ROS 2  

### 🔹 Privileged Information (Teacher Policy)
- Terrain scandots  
- Base linear velocity  
- Physical parameters (mass, friction coefficients, motor strengths)

---

## 🌍 Terrain & Navigation Setup

- **Navigation:** waypoint-based with adaptive heading adjustment based on obstacle geometry  
- **Terrain Representation:** depth images (for student), scandots (for teacher)  
- **Procedurally Generated Obstacles:**
  | Parameter | Range |
  |------------|--------|
  | Obstacle Height | [-0.45, 1.2] m |
  | Gap Size | [0.02, 0.08] m |
  | Stepping Stone Distance | [0.02, 0.08] m |
  | Max Slope Inclination | 1.5 rad |

---

## 🧩 Repository Structure
```
Directory structure:
└── fabeha-raheel-agile_locomotion/
    ├── README.md
    ├── install.sh
    ├── LICENSE
    ├── legged_gym/
    │   ├── LICENSE
    │   ├── requirements.txt
    │   ├── setup.py
    │   ├── legged_gym/
    │   │   ├── __init__.py
    │   │   ├── envs/
    │   │   │   ├── __init__.py
    │   │   │   ├── a1/
    │   │   │   │   ├── a1_config.py
    │   │   │   │   └── a1_parkour_config.py
    │   │   │   ├── base/
    │   │   │   │   ├── base_config.py
    │   │   │   │   ├── base_task.py
    │   │   │   │   └── legged_robot_config.py
    │   │   │   └── go1/
    │   │   │       └── go1_config.py
    │   │   ├── scripts/
    │   │   │   ├── evaluate.py
    │   │   │   ├── fetch.py
    │   │   │   ├── play.py
    │   │   │   ├── save_jit.py
    │   │   │   ├── train.py
    │   │   │   ├── visualize.py
    │   │   │   └── legged_gym/
    │   │   │       └── envs/
    │   │   │           ├── a1/
    │   │   │           │   └── a1_config.py
    │   │   │           └── base/
    │   │   │               └── legged_robot_config.py
    │   │   ├── tests/
    │   │   │   └── test_env.py
    │   │   └── utils/
    │   │       └── ...
    │   ├── licenses/
    │   └── resources/
    │       ├── actuator_nets/
    │       │   └── anydrive_v3_lstm.pt
    │       └── robots/
    │           ├── a1/
    │           ├── anymal_b/
    │           ├── anymal_c/
    │           └── cassie/
    └── rsl_rl/
        └── ...
```


---

## ⚙️ Installation & Setup

### 1️⃣ Create Conda Environment
```bash
conda create -n quad_env python=3.8
conda activate quad_env
```
### 2️⃣ Install Dependencies
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
### 3️⃣ Clone Repository
```bash
git clone https://github.com/fabeha-raheel/agile_locomotion.git
cd agile_locomotion
```
### 4️⃣ Install Isaac Gym

Download Isaac Gym binaries from NVIDIA Developer
.
Originally trained with Preview 3, compatible with Preview 4.
```bash
cd isaacgym/python && pip install -e .
```
### 5️⃣ Install Local Packages
```bash
cd ~/agile_locomotion/rsl_rl && pip install -e .
cd ~/agile_locomotion/legged_gym && pip install -e .
```
### 6️⃣ Install Additional Python Dependencies
```bash
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

## 🧪 Running the Code
### ▶️ Play Base (Teacher) Policy
```bash
python play.py --exptid xxx-xx
```

### ▶️ Play Distilled (Student) Policy
```bash
python play.py --exptid yyy-yy --delay --use_camera
```

## 📊 Results & Evaluation

The learned skills were tested by deploying the **student policy** obtained after knowledge distillation.  

| **Evaluation Parameter** | **Details** |
|---------------------------|-------------|
| **Environment** | 5×5 patch testbed |
| **Difficulty Levels** | 1 (Easy) → 5 (Hard) |
| **Trials per Course** | 20 |
| **Metrics** | Success rate, traversal completion, stability |

---

### 🎥 Demonstration Videos

- [Unitree Go1 Testing – Part 1](https://youtu.be/muhaUtQNDQw)  
- [Unitree Go1 Testing – Part 2](https://youtu.be/O6sVChRo0nw)  
- [Unitree A1 Testing](https://youtu.be/IDnxZjtDjd0)

---

## 📚 Key Techniques Summary

| **Technique** | **Purpose** |
|----------------|-------------|
| **Deep RL (PPO)** | Train locomotion policy end-to-end |
| **Teacher–Student Learning** | Distill privileged policy to deployable one |
| **DAGGER** | Iterative knowledge transfer |
| **Domain Randomization** | Improve sim-to-real transfer |
| **Curriculum Learning** | Gradual difficulty scaling |

---

## 🧾 Citation

If you find this work useful, please cite:

```bibtex
@thesis{raheel2025agilelocomotion,
  author       = {Fabeha Raheel},
  title        = {Agile Locomotion & Navigation of Quadrupedal Robots using Learning-based Approaches},
  school       = {National University of Sciences & Technology (NUST)},
  year         = {2025},
  supervisor   = {Dr. Umar Shahbaz Khan}
}
```

## 🌟 Acknowledgments

This research was carried out under the supervision of **Dr. Umar Shahbaz Khan** at the **National University of Sciences & Technology (NUST)**.  
Special thanks to **Dr. Hamid Jabbar**, **Dr. Tahir Habib Nawaz** and **Dr. Umer Izhar** for their valuable feedback during evaluations.

---

## 🧩 Keywords

`Quadrupedal Locomotion` · `Deep Reinforcement Learning` · `Agile Robotics` · `Privileged Learning` · `Teacher-Student Policy` · `Isaac Gym` · `Domain Randomization` · `Curriculum Learning` · `Knowledge Distillation` · `Imitation Learning` · `Unitree Go1` · `Unitree A1`

