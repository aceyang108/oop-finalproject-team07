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
從零開始訓練一個新的 Agent，訓練完成後會自動儲存最佳模型至frozen_lake8x8.pkl並繪製訓練曲線圖。
我以Tabular Q-Learning為主，結合Epsilon-Greedy，最後再用雙重線性衰減來優化。有使用了OOP的設計架構。
```bash
# 預設training(15,000 episodes)
python frozen_lake.py --train

# 可自訂回合數
python frozen_lake.py --train --episodes 10000

#實際跑十次
python frozen_lake.py --render
```

我有另外寫個測試用腳本
```bash
python test.py

#快速跑完 1000 回合並計算最終平均勝率
python test.py --benchmark
```

### **Part 3: OOP Project Environment**
Execute the custom OOP environment:

```bash
python oop_project_env.py
```

**Tip:**  
If you’re on Windows, replace  
```bash
source .venv/bin/activate
```  
with  
```bash
.venv\Scripts\activate
```
---
# Contribute
aceyang108 : part2(frozen_lake.py, test.py)