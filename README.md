# Group Project Setup Guide

## Project Content
- Gymnasium v1.2.2
- Part1 Sample Code
- Part2 Sample Code
- Part3 Sample Code
  
## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Navigate to the Gymnasium directory
cd group_project/Gymnasium

# 4. Install Gymnasium in editable mode
pip install -e .

# 5. Install additional dependencies
pip install "gymnasium[classic_control]"
pip install matplotlib
```

---

## ✅ Verification

Run the following command to verify that the installation is successful:

```bash
% pip list
```

Sample Output from MacOS:

```
Package              Version Editable project location
-------------------- ------- --------------------------------------------
cloudpickle          3.1.2
Farama-Notifications 0.0.4
gymnasium            1.2.2   ./group_project/Gymnasium
numpy                2.3.5
pip                  24.3.1
typing_extensions    4.15.0
```

If your output matches the above (or is similar), your environment is correctly configured.

---

## 🚀 Running the Project

### **Part 1: Mountain Car**
Train and test the reinforcement learning agent:

```bash
# Train the agent
python mountain_car.py --train --episodes 5000

# Render and visualize performance
python mountain_car.py --render --episodes 10
```

### **Part 2: Frozen Lake**
此部分實作了 Tabular Q-Learning 演算法，並結合 Epsilon-Greedy 策略與 雙重參數衰減 (Dual Parameter Decay) 機制來優化收斂過程。程式碼採用 OOP (物件導向) 架構設計，具備高模組化與可維護性。

若要使用視覺化功能 (--render)、請務必先安裝pygame
```bash
pip install "gymnasium[toy-text]" pygame
```

訓練模型，會自動儲存模型至 frozen_lake8x8.pkl 並繪製訓練曲線圖。
```bash
# 預設training(15,000 episodes)
python frozen_lake.py --train

# 可自訂回合數
python frozen_lake.py --train --episodes 10000
```

讀取訓練好的模型，實際演示 Agent 在冰湖上的行走過程。
```bash
#實際跑十次
python frozen_lake.py --render
```

我有另外寫個測試用腳本
```bash
# 預設執行 (10 episodes, Render ON)
python test.py

# 自訂播放回合數 (例如 5 回合)
python test.py --episodes 5

#快速跑完 1000 回合並計算最終平均勝率
python test.py --benchmark
```

### **Part 3: Crossy Road AI Agent (RL & OOP)**

This project implements a Reinforcement Learning agent (DQN) using `Stable-Baselines3` and `Gymnasium`. The environment and agent are structured using OOP principles.

### **1. Install Dependencies**

Ensure you have the required packages installed:

Bash

`pip install gymnasium numpy pygame stable-baselines3 shimmy tensorboard`

### **2. Run the Agent**

The main entry point is `agent.py`. It handles both training and evaluation.

Bash

`python agent.py`

After running the command, you will be prompted to select a mode:

- **`train`**: Starts training the AI model from scratch (or continues if a model exists). No game window will appear to speed up training.
- **`play`**: Loads the trained model and renders the game, letting you watch the AI play in real-time.
- **`benchmark`**: Runs a random agent for comparison.

### **3. Monitor Training (TensorBoard)**

To visualize training progress (Win rate, Average reward, Loss), run TensorBoard in a separate terminal:

Bash

`tensorboard --logdir ./logs/`

Then open your browser and go to: [http://localhost:6006/](https://www.google.com/search?q=http://localhost:6006/)
---
# Contribute
- aceyang108 : part2(frozen_lake.py, test.py)、modify crossy_road.py to be more OOP、reflection report、part2 demo slide
- Chiu0918: refined part3 (multiple vehicle classes, varied map layouts, agent.py), part3 demo slide
- MikanLord173: part3 base structure (my_env.py, crossy_road.py), uml graph
