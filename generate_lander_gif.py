from PIL import Image
import gymnasium as gym
import numpy as np
from gymnasium.utils.save_video import save_video
import torch
import torch.nn as nn
import torch.nn.functional as F


# initialize policy network
# takes in a state, determines the next action
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(obs_size, action_size, bias=False), nn.ReLU()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        logits = self.layer(x)
        return torch.log(F.softmax(logits, dim=-1) + 1e-9)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        log_probs = self.forward(state).squeeze(0)
        probs = torch.exp(log_probs)
        action = torch.multinomial(probs, 1)
        return action.item(), log_probs[action]


def generate_lander_episode_gif(model_path):
    model = PolicyNetwork(8, 4)
    model.load_state_dict(torch.load(model_path))

    env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
    observation, _ = env.reset()

    frames = []
    fps = env.metadata["render_fps"]

    while True:
        action, _log_probs = model.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated:
            frames = env.render()
            # save_video(frames, "videos", fps=env.metadata["render_fps"])
            break

    # convert saved frames to gif
    imgs = [Image.fromarray(img) for img in frames]
    imgs[0].save(
        "array.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=1000.0 / fps,
        loop=0,
    )

    env.close()


if __name__ == "__main__":
    generate_lander_episode_gif("test_model.pth")
