# AdMind-RL: Real-Time Marketing Budgeting with Reinforcement Learning

An agent that reallocates multi-channel marketing spend to maximize ROAS under budget & constraints.

## Quickstart
```bash
# 1) Python 3.11+ recommended
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train PPO agent (offline simulator)
python train.py --config configs/admind-rl.yaml

# 4) Launch dashboard
streamlit run app.py
```
Outputs (models, logs) land in `./artifacts/`.

## Project Structure
```
admind-rl/
  configs/
    admind-rl.yaml
  docs/
    architecture.md
  src/admind_rl/
    __init__.py
    simulator.py
    env.py
    baselines.py
    utils.py
  train.py
  app.py
  requirements.txt
  Makefile
  README.md
```
