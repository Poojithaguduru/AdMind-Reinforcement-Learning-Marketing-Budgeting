# AdMind – Reinforcement Learning Marketing Budgeting

**An end-to-end system that learns how to split a daily advertising budget across channels (Google Ads, Meta Ads, Email, Influencer Marketing, and more) to maximize profit and Return on Advertising Spend (ROAS), with an interactive dashboard and a fully reproducible training pipeline.**

> **Note on naming:** This repository’s folder is named `admind-rl` for file-system convenience, but the project’s full name is **AdMind – Reinforcement Learning Marketing Budgeting** (no short forms in this README).

---

## 1) Why I chose this project (and why it is different)

Most portfolio projects in marketing analytics focus on common templates like churn prediction or simple dashboards. Those are useful, but they rarely prove real-world decision-making ability.

This project is different because it:

- **Optimizes decisions, not just reports**. Instead of “showing charts,” it **decides how to allocate budget** every day across channels to maximize business outcomes.
- Uses **Reinforcement Learning (spelled out, not abbreviated)** to continuously learn from feedback (profit and return on advertising spend), adapting to changing marketing conditions.
- Simulates realistic marketing behavior (**diminishing returns, weekly seasonality, noise, spend carry-over**), which mirrors what real campaigns experience.
- Ships with a **usable dashboard** so stakeholders can see how budget is allocated, how profit and return on advertising spend evolve, and how experiments change outcomes.
- Is **modular and extendable**: you can add more channels, change the reward function (for example, prioritize conversions vs. profit), adjust constraints, or hook it to a real data pipeline.

This demonstrates skills in **data engineering**, **machine learning**, **applied decision science**, and **product thinking**—the exact combination many data roles seek.

---

## 2) What this system does

- Takes a **daily marketing budget** (for example, 10,000 units of currency).
- Simulates multiple advertising channels (Google Ads, Meta Ads, Email, Influencer Marketing, and you can add more), each with:
  - **Diminishing returns** (spending more on one channel eventually gives fewer extra conversions).
  - **Carry-over effects** (yesterday’s spend still affects today).
  - **Weekly seasonality** (weekdays vs weekends behave differently).
  - **Noise** (real markets are noisy; the simulator reflects that).
- Trains a **Proximal Policy Optimization** agent (a well-known reinforcement learning algorithm) to decide **what percentage of the daily budget** to assign to each channel.
- Applies a **reward function** that encourages high **profit** and good **return on advertising spend**, while **penalizing very jumpy day-to-day changes**.
- Provides a **Streamlit dashboard** to simulate 30+ days and compare:
  - A simple **Fixed Split baseline** (even split across channels)  
  - The **Trained Proximal Policy Optimization policy**  
  You see charts for **profit**, **return on advertising spend**, **conversions**, and the **final-day allocation** per channel.

---

## 3) Business problem (plain language)

Marketing teams must decide **how much budget to allocate** to each channel (search ads, social ads, email, influencer marketing).  
The challenge: **what worked yesterday may not work tomorrow** due to competition, seasonality, saturation, and user behavior changes.

- **Objective:** Maximize **profit** and **return on advertising spend**.
- **Constraints:** Limited daily budget and per-channel min/max limits (for example, you may be required to spend at least some amount on each channel, or not exceed a contract cap).
- **Why reinforcement learning?**  
  Traditional rules (“spend 40% on search, 30% on social…”) ignore how performance changes over time.  
  Reinforcement learning treats the problem like training a smart agent that **tries, learns, and adjusts** to changing outcomes, similar to how a trader optimizes a portfolio.

**Return on Advertising Spend (ROAS)** is the ratio:
ROAS = (Revenue from ads) / (Cost of ads)
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

