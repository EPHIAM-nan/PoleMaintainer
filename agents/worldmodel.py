"""World Model + CEM Controller for CartPole (inspired by Ha & Schmidhuber, 2018).

Pipeline:
 1) collect_random_dataset: random rollouts to build (s, a) -> s' pairs
 2) WorldModel: lightweight MLP predicting next state given (s, a)
 3) train_world_model: supervised fit with MSE loss, saves curve + weights
 4) Controller: MLP policy; parameters optimized via Cross Entropy Method (CEM)
 5) train_controller_cem: searches controller params inside learned model (no env)
 6) evaluate: run the learned controller in the real env

All hyperparameters are declared as constants below; no argparser used.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Global settings / hyperparameters
# -----------------------------
ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data collection
DATA_EPISODES = 10000
# CartPole-v1 truncates at 500 steps; keep collection horizon consistent.
DATA_MAX_STEPS = 500
DATASET_PATH = os.path.join("bc_data", "world_model_dataset.npz")

# World model (dynamics)
WORLD_HIDDEN = 128
WORLD_LR = 1e-3
WORLD_LR_FINAL = 1e-4
WORLD_EPOCHS = 30
WORLD_BATCH = 128
WORLD_PRED_HORIZON = 10
WORLD_MODEL_PATH = os.path.join(MODEL_DIR, "wm_dynamics.torch")
WORLD_LOSS_PNG = os.path.join(os.path.dirname(__file__), "wm_dynamics.png")

# Controller (policy) + CEM
CONTROLLER_HIDDEN = 16
CEM_ITERATIONS = 60
CEM_POP_SIZE = 128
CEM_ELITE_FRAC = 0.1
CEM_ROLLOUTS_PER_CANDIDATE = 10
# Match the real env horizon to avoid model/real mismatch.
MODEL_HORIZON = DATA_MAX_STEPS
CEM_SIGMA_INIT = 0.5
CONTROLLER_PATH = os.path.join(MODEL_DIR, "wm_controller.torch")
CONTROLLER_CURVE_PNG = os.path.join(os.path.dirname(__file__), "wm_controller.png")

# Evaluation
EVAL_EPISODES = 100


torch.manual_seed(SEED)
np.random.seed(SEED)


# -----------------------------
# Utilities
# -----------------------------
def cartpole_terminated(state: np.ndarray) -> bool:
	"""Approximate the env termination condition using state bounds."""
	x, _, theta, _ = state
	theta_thresh = 12 * np.pi / 180  # CartPole termination angle (radians)
	x_thresh = 2.4  # CartPole termination position (meters)
	return abs(x) > x_thresh or abs(theta) > theta_thresh


def cartpole_terminated_torch(states: torch.Tensor) -> torch.Tensor:
	"""Vectorized approximation of env termination condition using state bounds.
	
	Args:
		states: [batch_size, obs_dim] tensor
	Returns:
		terminated: [batch_size] boolean tensor
	"""
	x = states[:, 0]
	theta = states[:, 2]
	theta_thresh = 12 * torch.pi / 180
	x_thresh = 2.4
	return (x.abs() > x_thresh) | (theta.abs() > theta_thresh)



def to_tensor(x: np.ndarray, device: torch.device = DEVICE) -> torch.Tensor:
	return torch.as_tensor(x, dtype=torch.float32, device=device)


# -----------------------------
# Data collection
# -----------------------------
def collect_random_dataset(
	episodes: int = DATA_EPISODES,
	max_steps: int = DATA_MAX_STEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Collect random transitions (s, a, s') without training any policy.

	If a cached dataset exists at DATASET_PATH, it is loaded instead of recollected.

	Returns:
		states: [N, obs_dim]
		actions: [N]
		next_states: [N, obs_dim]
		init_states: [episodes, obs_dim] starting states for later CEM rollouts
		dones: [N] boolean array indicating end of episode
	"""

	if os.path.exists(DATASET_PATH):
		print(f"[Data] Using cached dataset at {DATASET_PATH}")
		return load_dataset(DATASET_PATH)

	env = gym.make(ENV_NAME)
	obs_dim = env.observation_space.shape[0]
	states, actions, next_states, init_states, dones = [], [], [], [], []

	for ep in range(episodes):
		state, _ = env.reset(seed=SEED + ep)
		init_states.append(state)

		for _ in range(max_steps):
			action = env.action_space.sample()
			nxt, _, terminated, truncated, _ = env.step(action)

			states.append(state)
			actions.append(action)
			next_states.append(nxt)
			dones.append(terminated or truncated)

			state = nxt
			if terminated or truncated:
				break

	env.close()
	os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
	print(f"[Data] Saving dataset -> {DATASET_PATH}")
	np.savez_compressed(
		DATASET_PATH,
		states=np.asarray(states, dtype=np.float32),
		actions=np.asarray(actions, dtype=np.int64),
		next_states=np.asarray(next_states, dtype=np.float32),
		init_states=np.asarray(init_states, dtype=np.float32),
		dones=np.asarray(dones, dtype=bool),
	)

	return (
		np.asarray(states, dtype=np.float32),
		np.asarray(actions, dtype=np.int64),
		np.asarray(next_states, dtype=np.float32),
		np.asarray(init_states, dtype=np.float32),
		np.asarray(dones, dtype=bool),
	)


def load_dataset(path: str = DATASET_PATH) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Load a saved dataset from disk."""
	data = np.load(path)
	return (
		data["states"],
		data["actions"],
		data["next_states"],
		data["init_states"],
		data["dones"],
	)


def collect_real_rollouts(
	controller: Controller,
	episodes: int,
	max_steps: int = DATA_MAX_STEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Collect rollouts using the trained controller in the real environment."""
	env = gym.make(ENV_NAME)
	states, actions, next_states, init_states = [], [], [], []

	for ep in range(episodes):
		state, _ = env.reset(seed=int(time.time()) + ep)
		init_states.append(state)

		for _ in range(max_steps):
			# Get action from controller
			s_tensor = to_tensor(state).unsqueeze(0)
			with torch.no_grad():
				action = controller.act(s_tensor)

			nxt, _, terminated, truncated, _ = env.step(action)

			states.append(state)
			actions.append(action)
			next_states.append(nxt)

			state = nxt
			if terminated or truncated:
				break

	env.close()
	return (
		np.asarray(states, dtype=np.float32),
		np.asarray(actions, dtype=np.int64),
		np.asarray(next_states, dtype=np.float32),
		np.asarray(init_states, dtype=np.float32),
	)



# -----------------------------
# World model (M): MLP predicting s_{t+1} from (s_t, a_t)
# -----------------------------
class WorldModel(nn.Module):
	def __init__(self, obs_dim: int, act_dim: int, hidden: int = WORLD_HIDDEN):
		super().__init__()
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		self.net = nn.Sequential(
			nn.Linear(obs_dim + act_dim, hidden),
			nn.ReLU(),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Linear(hidden, obs_dim),
		)
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
		
		# Normalization statistics
		self.register_buffer("state_mean", torch.zeros(1, obs_dim))
		self.register_buffer("state_std", torch.ones(1, obs_dim))

	def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		# action one-hot to match discrete CartPole actions
		if action.dim() == 1:
			action = F.one_hot(action, num_classes=self.act_dim).float()
		elif action.dim() == 2 and action.size(1) == 1:
			action = F.one_hot(action.squeeze(1), num_classes=self.act_dim).float()
		x = torch.cat([state, action], dim=1)
		return self.net(x)

	def get_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		# Handles normalization/denormalization
		# state: [batch, obs_dim] (raw)
		# action: [batch] or [batch, 1]
		
		norm_state = (state - self.state_mean) / (self.state_std + 1e-6)
		norm_next_state = self.forward(norm_state, action)
		next_state = norm_next_state * (self.state_std + 1e-6) + self.state_mean
		return next_state


def train_world_model(
	states: np.ndarray,
	actions: np.ndarray,
	next_states: np.ndarray,
	dones: np.ndarray,
	epochs: int = WORLD_EPOCHS,
	batch_size: int = WORLD_BATCH,
	lr: float = WORLD_LR,
	lr_final: float = WORLD_LR_FINAL,
	val_split: float = 0.1,
	horizon: int = WORLD_PRED_HORIZON,
) -> WorldModel:
	"""Supervised training of the dynamics model; saves weights + loss curve."""

	obs_dim = states.shape[1]
	act_dim = int(actions.max()) + 1

	# 0. Compute and apply normalization
	print("[WorldModel] Computing normalization statistics...")
	mean = states.mean(axis=0)
	std = states.std(axis=0) + 1e-6
	
	# Normalize dataset in-place (or copy if needed, but here we can just use local vars)
	# We need to normalize states and next_states
	states_norm = (states - mean) / std
	next_states_norm = (next_states - mean) / std
	
	# 1. Prepare sequences for N-step prediction
	print(f"[WorldModel] Preparing sequences with horizon={horizon}...")
	seq_s_list, seq_a_list, seq_t_list = [], [], []
	
	start_idx = 0
	N = len(states)
	# Find episode boundaries
	episode_ends = np.where(dones)[0]
	
	prev_end = 0
	for end_idx in episode_ends:
		# Episode range: [prev_end, end_idx + 1)
		# Note: end_idx is inclusive in the sense that dones[end_idx] is True
		# So slice is prev_end : end_idx + 1
		ep_len = (end_idx + 1) - prev_end
		
		if ep_len >= horizon:
			ep_s = states_norm[prev_end : end_idx + 1]
			ep_a = actions[prev_end : end_idx + 1]
			ep_ns = next_states_norm[prev_end : end_idx + 1]
			
			# Create sliding windows
			# We want s[t], a[t:t+H], ns[t:t+H]
			# Valid t range: 0 to ep_len - horizon
			# Example: len=10, H=10. t=0. s[0], a[0:10], ns[0:10]. Correct.
			
			# Vectorized window extraction for this episode
			# Using stride_tricks is cleaner but loop is safer for now
			for t in range(ep_len - horizon + 1):
				seq_s_list.append(ep_s[t])
				seq_a_list.append(ep_a[t : t + horizon])
				seq_t_list.append(ep_ns[t : t + horizon])
				
		prev_end = end_idx + 1

	seq_s = np.array(seq_s_list, dtype=np.float32)
	seq_a = np.array(seq_a_list, dtype=np.int64)
	seq_t = np.array(seq_t_list, dtype=np.float32)
	
	print(f"[WorldModel] Created {len(seq_s)} sequences.")

	# Split data into train/val
	total_samples = seq_s.shape[0]
	val_size = int(total_samples * val_split)
	train_size = total_samples - val_size

	# Shuffle indices to ensure random split
	indices = np.random.permutation(total_samples)
	train_idx = indices[:train_size]
	val_idx = indices[train_size:]

	train_dataset = TensorDataset(
		torch.from_numpy(seq_s[train_idx]),
		torch.from_numpy(seq_a[train_idx]),
		torch.from_numpy(seq_t[train_idx]),
	)
	val_dataset = TensorDataset(
		torch.from_numpy(seq_s[val_idx]),
		torch.from_numpy(seq_a[val_idx]),
		torch.from_numpy(seq_t[val_idx]),
	)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	model = WorldModel(obs_dim, act_dim).to(DEVICE)
	
	# Set normalization stats
	model.state_mean[:] = torch.from_numpy(mean).to(DEVICE)
	model.state_std[:] = torch.from_numpy(std).to(DEVICE)
	
	optim = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.LinearLR(
		optim,
		start_factor=1.0,
		end_factor=lr_final / lr,
		total_iters=max(1, epochs - 1),
	)
	criterion = nn.MSELoss()

	train_losses: list[float] = []
	val_losses: list[float] = []
	best_val_loss = float("inf")
	best_model_state = None

	for epoch in range(1, epochs + 1):
		current_lr = optim.param_groups[0]["lr"]
		
		# Training
		model.train()
		running_train = 0.0
		for s, a_seq, target_seq in train_loader:
			s = s.to(DEVICE, dtype=torch.float32)
			a_seq = a_seq.to(DEVICE)
			target_seq = target_seq.to(DEVICE, dtype=torch.float32)

			loss = 0.0
			curr_state = s
			
			# Unroll horizon
			for t in range(horizon):
				a = a_seq[:, t]
				target = target_seq[:, t]
				
				pred_next = model(curr_state, a)
				loss += criterion(pred_next, target)
				
				# Autoregressive step: use prediction as next input
				curr_state = pred_next
			
			loss = loss / horizon

			optim.zero_grad()
			loss.backward()
			optim.step()

			running_train += loss.item() * s.size(0)

		epoch_train_loss = running_train / len(train_dataset)
		train_losses.append(epoch_train_loss)

		# Validation
		model.eval()
		running_val = 0.0
		with torch.no_grad():
			for s, a_seq, target_seq in val_loader:
				s = s.to(DEVICE, dtype=torch.float32)
				a_seq = a_seq.to(DEVICE)
				target_seq = target_seq.to(DEVICE, dtype=torch.float32)

				loss = 0.0
				curr_state = s
				
				for t in range(horizon):
					a = a_seq[:, t]
					target = target_seq[:, t]
					
					pred_next = model(curr_state, a)
					loss += criterion(pred_next, target)
					curr_state = pred_next
				
				loss = loss / horizon
				running_val += loss.item() * s.size(0)
		
		epoch_val_loss = running_val / len(val_dataset) if len(val_dataset) > 0 else 0.0
		val_losses.append(epoch_val_loss)

		if epoch_val_loss < best_val_loss:
			best_val_loss = epoch_val_loss
			best_model_state = model.state_dict()
			print(f"[WorldModel] epoch={epoch:03d} lr={current_lr:.6f} train_loss={epoch_train_loss:.6f} val_loss={epoch_val_loss:.6f} (new best)")
		else:
			print(f"[WorldModel] epoch={epoch:03d} lr={current_lr:.6f} train_loss={epoch_train_loss:.6f} val_loss={epoch_val_loss:.6f}")

		if epoch < epochs:
			scheduler.step()

	os.makedirs(MODEL_DIR, exist_ok=True)
	if best_model_state is not None:
		model.load_state_dict(best_model_state)
		print(f"[WorldModel] Loaded best model with val_loss={best_val_loss:.6f}")
	
	torch.save(model.state_dict(), WORLD_MODEL_PATH)
	_plot_curve({"train": train_losses, "val": val_losses}, WORLD_LOSS_PNG, ylabel="MSE loss (sum over horizon)", title=f"World model training (H={horizon})")
	print(f"[WorldModel] saved weights -> {WORLD_MODEL_PATH}")
	print(f"[WorldModel] curve saved -> {WORLD_LOSS_PNG}")
	return model


def load_world_model(path: str | None = None) -> WorldModel:
	"""Load a trained world model from disk and return it on the configured device."""
	ckpt = path or WORLD_MODEL_PATH
	env = gym.make(ENV_NAME)
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.n
	env.close()

	model = WorldModel(obs_dim, act_dim).to(DEVICE)
	state_dict = torch.load(ckpt, map_location=DEVICE)
	model.load_state_dict(state_dict)
	model.eval()
	return model


# -----------------------------
# Controller (C): policy MLP + CEM search in model space
# -----------------------------
class Controller(nn.Module):
	def __init__(self, obs_dim: int, act_dim: int, hidden: int = CONTROLLER_HIDDEN):
		super().__init__()
		self.obs_dim = obs_dim
		self.act_dim = act_dim
		self.net = nn.Sequential(
			nn.Linear(obs_dim, hidden),
			nn.ReLU(),
			nn.Linear(hidden, hidden),
			nn.ReLU(),
			nn.Linear(hidden, act_dim),
		)
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)

	def forward(self, state: torch.Tensor) -> torch.Tensor:
		return self.net(state)

	def act(self, state: torch.Tensor) -> int:
		if state.dim() == 1:
			state = state.unsqueeze(0)
		logits = self.forward(state)
		action = torch.argmax(logits, dim=1)
		return int(action.item())

	def num_params(self) -> int:
		return sum(p.numel() for p in self.parameters())

	def set_flat_params(self, flat: np.ndarray):
		vec = torch.as_tensor(flat, dtype=torch.float32, device=DEVICE)
		torch.nn.utils.vector_to_parameters(vec, self.parameters())


def rollout_in_model(
	world_model: WorldModel,
	controller: Controller,
	initial_state: np.ndarray,
	horizon: int = MODEL_HORIZON,
) -> float:
	"""Simulate inside the learned model; reward = 1 per step until termination."""
	controller.eval()
	world_model.eval()

	with torch.no_grad():
		state = to_tensor(initial_state).unsqueeze(0)
		total = 0.0

		for _ in range(horizon):
			action = controller.act(state)
			action_tensor = torch.tensor([action], device=DEVICE)
			next_state = world_model.get_next_state(state, action_tensor)
			state_np = next_state.cpu().numpy().squeeze(0)
			total += 1.0
			if cartpole_terminated(state_np):
				break
			state = to_tensor(state_np).unsqueeze(0)

	return total


def rollout_batch(
	world_model: WorldModel,
	controller_template: Controller,
	flat_params_batch: torch.Tensor,
	init_states: torch.Tensor,
	horizon: int = MODEL_HORIZON,
) -> torch.Tensor:
	"""Vectorized rollout for a batch of controllers and initial states.

	Args:
		world_model: The shared dynamics model.
		controller_template: A Controller instance to use as structure.
		flat_params_batch: [pop_size, param_dim] tensor of parameters.
		init_states: [pop_size, rollouts_per_cand, obs_dim] tensor.
		horizon: Max steps.

	Returns:
		scores: [pop_size] tensor of mean returns.
	"""
	pop_size, rollouts_per_cand, obs_dim = init_states.shape
	total_rollouts = pop_size * rollouts_per_cand

	# Expand params: [pop_size, D] -> [pop_size, R, D] -> [N, D]
	flat_params_expanded = flat_params_batch.unsqueeze(1).expand(-1, rollouts_per_cand, -1).reshape(total_rollouts, -1)

	# Flatten states: [N, obs_dim]
	current_states = init_states.reshape(total_rollouts, obs_dim)

	# Reconstruct state_dict for vmap
	param_shapes = {name: p.shape for name, p in controller_template.named_parameters()}
	param_numels = {name: p.numel() for name, p in controller_template.named_parameters()}
	param_names = list(param_shapes.keys())

	params_dict_batch = {}
	start = 0
	for name in param_names:
		numel = param_numels[name]
		shape = param_shapes[name]
		chunk = flat_params_expanded[:, start : start + numel]
		params_dict_batch[name] = chunk.view(total_rollouts, *shape)
		start += numel

	active_mask = torch.ones(total_rollouts, dtype=torch.bool, device=DEVICE)
	total_rewards = torch.zeros(total_rollouts, dtype=torch.float32, device=DEVICE)

	# Functional wrapper for controller
	def compute_logits(params, state):
		return torch.func.functional_call(controller_template, params, (state,))

	for _ in range(horizon):
		if not active_mask.any():
			break

		# 1. Get actions (vmap over batch dim 0)
		logits = torch.vmap(compute_logits)(params_dict_batch, current_states)
		actions = torch.argmax(logits, dim=1)  # [N]

		# 2. World Model Step
		next_states = world_model.get_next_state(current_states, actions)

		# 3. Check termination
		dones = cartpole_terminated_torch(next_states)

		# 4. Update rewards
		# Add 1 if it was active at the start of this step
		total_rewards += active_mask.float()

		# Update mask
		active_mask = active_mask & (~dones)
		current_states = next_states

	# Aggregate results
	scores_matrix = total_rewards.view(pop_size, rollouts_per_cand)
	return scores_matrix.mean(dim=1)



def train_controller_cem(
	world_model: WorldModel,
	init_state_pool: np.ndarray,
	iterations: int = CEM_ITERATIONS,
	pop_size: int = CEM_POP_SIZE,
	elite_frac: float = CEM_ELITE_FRAC,
	sigma_init: float = CEM_SIGMA_INIT,
	horizon: int = MODEL_HORIZON,
	rollouts_per_candidate: int = CEM_ROLLOUTS_PER_CANDIDATE,
	stop_when_return: float | None = None,
) -> Controller:
	"""Cross Entropy Method on controller parameters using the learned model."""
	obs_dim = world_model.obs_dim
	act_dim = world_model.act_dim
	world_model.eval()
	world_model.requires_grad_(False)

	controller = Controller(obs_dim, act_dim).to(DEVICE)
	param_dim = controller.num_params()

	mu = np.zeros(param_dim, dtype=np.float32)
	sigma = np.ones(param_dim, dtype=np.float32) * sigma_init
	elite_k = max(1, int(pop_size * elite_frac))

	history: list[float] = []
	best_return = -np.inf
	best_params: np.ndarray | None = None

	for it in range(1, iterations + 1):
		params_batch = np.random.randn(pop_size, param_dim).astype(np.float32) * sigma + mu
		init_idx = np.random.randint(0, init_state_pool.shape[0], size=(pop_size, rollouts_per_candidate))

		# Vectorized rollout
		flat_params_batch_torch = torch.as_tensor(params_batch, dtype=torch.float32, device=DEVICE)
		init_states_torch = torch.as_tensor(init_state_pool[init_idx], dtype=torch.float32, device=DEVICE)

		scores_torch = rollout_batch(world_model, controller, flat_params_batch_torch, init_states_torch, horizon)
		scores_np = scores_torch.cpu().numpy()

		# Update best
		max_idx = np.argmax(scores_np)
		if scores_np[max_idx] > best_return:
			best_return = scores_np[max_idx]
			best_params = params_batch[max_idx].copy()

		elite_idx = np.argsort(scores_np)[-elite_k:]
		elite_params = params_batch[elite_idx]
		mu = elite_params.mean(axis=0)
		sigma = elite_params.std(axis=0) + 1e-3

		history.append(float(scores_np.mean()))
		print(
			f"[CEM] iter={it:02d} mean_return={scores_np.mean():.2f} "
			f"elite_mean={scores_np[elite_idx].mean():.2f} best={best_return:.2f}"
		)

		if stop_when_return is not None and best_return >= stop_when_return:
			print(f"[CEM] Early stop: best_return {best_return:.2f} >= {stop_when_return}")
			break

	final_controller = Controller(obs_dim, act_dim).to(DEVICE)
	final_params = best_params if best_params is not None else mu
	final_controller.set_flat_params(final_params)

	os.makedirs(MODEL_DIR, exist_ok=True)
	torch.save(final_controller.state_dict(), CONTROLLER_PATH)
	_plot_curve(history, CONTROLLER_CURVE_PNG, ylabel="avg return", title="Controller CEM")
	print(f"[CEM] best_return={best_return:.2f}")
	print(f"[CEM] controller saved -> {CONTROLLER_PATH}")
	print(f"[CEM] curve saved -> {CONTROLLER_CURVE_PNG}")
	return final_controller


def load_controller_model(path: str | None = None) -> Controller:
	"""Load a trained controller from disk and return it on the configured device."""
	env = gym.make(ENV_NAME)
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.n
	env.close()

	ckpt = path or CONTROLLER_PATH
	controller = Controller(obs_dim, act_dim).to(DEVICE)
	state_dict = torch.load(ckpt, map_location=DEVICE)
	controller.load_state_dict(state_dict)
	controller.eval()
	return controller


# -----------------------------
# Evaluation in the real environment
# -----------------------------
def evaluate(
	controller_path: str | None = None,
	episodes: int = EVAL_EPISODES,
	render: bool = True,
	fps: int = 60,
):
	"""Evaluate the learned controller in the true CartPole environment."""
	model_path = controller_path or CONTROLLER_PATH
	render_mode = "human" if render else None
	env = gym.make(ENV_NAME, render_mode=render_mode)
	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.n

	controller = Controller(obs_dim, act_dim).to(DEVICE)
	controller.load_state_dict(torch.load(model_path, map_location=DEVICE))
	controller.eval()

	dt = (1.0 / fps) if render and fps else 0.0
	scores = []

	for ep in range(1, episodes + 1):
		state, _ = env.reset(seed=10_000 + ep)
		done = False
		steps = 0

		while not done:
			s_tensor = to_tensor(state).unsqueeze(0)
			with torch.no_grad():
				action = controller.act(s_tensor)

			state, _, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			steps += 1

			if dt > 0:
				time.sleep(dt)

		scores.append(steps)
		print(f"[Eval] episode={ep} steps={steps}")

	env.close()
	avg = float(np.mean(scores)) if scores else 0.0
	print(f"[Eval] average over {episodes} episodes: {avg:.2f}")
	return scores


# -----------------------------
# Plot helper
# -----------------------------
def _plot_curve(values: Iterable[float] | dict[str, Iterable[float]], path: str, ylabel: str, title: str):
	plt.figure(figsize=(6, 4))
	if isinstance(values, dict):
		for label, val in values.items():
			plt.plot(val, label=label)
		plt.legend()
	else:
		plt.plot(values)
	plt.xlabel("epoch" if "World" in title else "iteration")
	plt.ylabel(ylabel)
	plt.title(title)
	plt.tight_layout()
	plt.savefig(path, dpi=200)
	plt.close()


if __name__ == "__main__":
	# End-to-end run: collect data -> train M -> train C -> evaluate in env

	s, a, s2, init_s, dones = collect_random_dataset()
	# s, a, s2, init_s, dones = load_dataset("bc_data/world_model_dataset.npz")
	world = train_world_model(s, a, s2, dones)
	world = load_world_model(WORLD_MODEL_PATH)
	ctrl = train_controller_cem(world, init_s)

	evaluate(controller_path=CONTROLLER_PATH, episodes=EVAL_EPISODES, render=False)