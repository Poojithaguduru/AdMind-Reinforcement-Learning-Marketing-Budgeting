

from __future__ import annotations
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import argparse, yaml
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from admind_rl.env import AdSpendEnv, EnvConfig
from admind_rl.simulator import ChannelParams

def load_cfg(path:str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_env(cfg_dict:dict, seed:int):
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
    return AdSpendEnv(cfg, seed=seed)

def main(args):
    cfg = load_cfg(args.config)
    seed = int(cfg.get("seed", 42))
    env = DummyVecEnv([lambda: make_env(cfg, seed)])

    model = PPO("MlpPolicy", env, verbose=1, seed=seed, n_steps=2048, batch_size=256)
    timesteps = int(cfg.get("train_timesteps", 100_000))
    model.learn(total_timesteps=timesteps)

    out_dir = Path("artifacts/models"); out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ppo_admind_rl.zip"
    model.save(str(model_path))
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/admind-rl.yaml")
    main(ap.parse_args())
