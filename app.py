import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import yaml
from pathlib import Path
from stable_baselines3 import PPO
from admind_rl.env import AdSpendEnv, EnvConfig
from admind_rl.simulator import ChannelParams

st.set_page_config(page_title="AdMind-RL", layout="wide")
st.title("AdMind-RL — Real-Time Marketing Budgeting with RL")

@st.cache_data
def load_cfg(path:str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

cfg_dict = load_cfg("configs/admind-rl.yaml")
channels = [ChannelParams(**c) for c in cfg_dict["channels"]]
cfg = EnvConfig(
    budget_daily=cfg_dict["budget_daily"],
    window=cfg_dict["window"],
    price_per_conv=cfg_dict["price_per_conv"],
    volatility_penalty=cfg_dict["volatility_penalty"],
    noise_std=cfg_dict["noise_std"],
    seasonality_amp=cfg_dict["seasonality_amp"],
    channels=channels,
)

env = AdSpendEnv(cfg, seed=cfg_dict.get("seed", 42))

model_path = Path("artifacts/models/ppo_admind_rl.zip")
model = None
if model_path.exists():
    model = PPO.load(str(model_path))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Config")
    budget = st.number_input("Daily budget", value=float(cfg.budget_daily), step=500.0)
    days = st.number_input("Simulate days", min_value=7, max_value=365, value=30, step=1)
with col2:
    st.subheader("Mode")
    mode = st.radio("Policy", ["Fixed Split (baseline)", "PPO (trained)"], index=0 if model is None else 1)

def run(policy:str, days:int):
    obs, _ = env.reset()
    roas_hist, profit_hist, conv_hist = [], [], []
    alloc_hist = []
    for _ in range(int(days)):
        if policy == "PPO (trained)" and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.zeros(len(channels))  # baseline ~ uniform
        obs, reward, term, trunc, info = env.step(action)
        roas_hist.append(info["roas"]); profit_hist.append(info["profit"]); conv_hist.append(info["conversions"])
        alloc_hist.append(info["spend_props"])
    return np.array(roas_hist), np.array(profit_hist), np.array(conv_hist), np.array(alloc_hist)

if st.button("Run Simulation"):
    roas, profit, conv, alloc = run(mode, int(days))
    st.line_chart(roas, height=220)
    st.line_chart(profit, height=220)
    st.line_chart(conv, height=220)
    st.subheader("Allocation (last day)")
    last = alloc[-1]
    lab = [c.name for c in channels]
    st.table({"channel": lab, "proportion": last})
else:
    st.info("Click **Run Simulation** to compare policies.")
