"""Train a PPO reach policy on RoArmReachEnv, headless.

    python -m roarm_rl.train --timesteps 300000 --out runs/ppo_reach

Writes the final model to <out>/final.zip and periodic checkpoints to
<out>/checkpoints/. Use --eval <model.zip> to score a saved policy.
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from roarm_rl.env import RoArmReachEnv


def train(timesteps, out_dir, n_envs, seed):
    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
    env = make_vec_env(RoArmReachEnv, n_envs=n_envs, seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        seed=seed,
    )
    ckpt = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=os.path.join(out_dir, "checkpoints"),
        name_prefix="ppo_reach",
    )
    model.learn(total_timesteps=timesteps, callback=ckpt, progress_bar=False)
    model.save(os.path.join(out_dir, "final"))
    env.close()
    return os.path.join(out_dir, "final.zip")


def evaluate(model_path, episodes=20, seed=123):
    model = PPO.load(model_path)
    env = RoArmReachEnv(seed=seed)
    successes, final_dists = 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        successes += int(info.get("success", False))
        final_dists.append(info.get("dist", float("nan")))
    env.close()
    print(f"eval: success {successes}/{episodes}, mean final dist {sum(final_dists)/len(final_dists):.4f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--out", default="runs/ppo_reach")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", default=None, help="path to a saved model to evaluate instead of training")
    args = ap.parse_args()

    if args.eval:
        evaluate(args.eval)
        return
    path = train(args.timesteps, args.out, args.n_envs, args.seed)
    print(f"saved {path}")
    evaluate(path)


if __name__ == "__main__":
    main()
