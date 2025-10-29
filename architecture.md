# Architecture
- **Simulator**: synthetic yet realistic multi-channel demand with diminishing returns, carryover, noise, and seasonality.
- **Gym Env**: exposes state (recent spend & KPIs), takes action (spend deltas), returns reward (profit or ROAS with penalties).
- **Agent**: PPO (stable-baselines3).
- **Baselines**: fixed split & greedy ROI.
- **Dashboard**: compare policy vs baselines, inspect constraints and KPI curves.
