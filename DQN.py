import time
from DuelingNetwork import DuelingNetwork
from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from ReplayBuffer import ReplayBuffer
from tensorboardX import SummaryWriter
from Env import Env
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
import random
from tqdm import tqdm
import math
import numba as nb
import datetime
import os


class DQN(object):
    def __init__(self,
                 env: Env,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 update_period: int = 100,
                 gamma: float = 0.99,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.1,
                 x: float = 0.35,
                 # PER Parameters
                 alpha: float = 0.2,
                 beta: float = 0.6,
                 prior_eps: float = 1e-6,
                 # N-step Learning
                 n_step: int = 3,
                 ):
        self.env = env
        self.update_period = update_period
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon_decay = 0
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.x = x
        self.epsilon = max_epsilon
        self.gamma = gamma

        # PER
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(self.env.getNcls(), memory_size, batch_size, alpha)

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                self.env.getNcls(), memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # network
        self.current_model = DuelingNetwork(env.ncls, env.ncls).to(self.device)
        self.target_model = DuelingNetwork(env.ncls, env.ncls).to(self.device)
        self.target_model.load_state_dict(self.current_model.state_dict())
        self.target_model.eval()

        # optimizer
        self.optimizer = torch.optim.Adam(self.current_model.parameters())

        # transition
        self.transition = list()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection

        if self.epsilon > np.random.random():
            selected_action = np.array(np.random.choice(self.env.act_spc, 1)[0],  dtype=np.int64)
            # selected_action = np.array(np.random.choice(
            #     np.setdiff1d(self.env.act_spc, np.array(self.env.order, dtype=np.float32))
            #     , 1)[0], dtype=np.int64)
            # print("+", selected_action, type(selected_action))
        else:
            selected_action = self.current_model(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
            # print("*", selected_action, type(selected_action))

        return selected_action


    def step(self, act):
        state, state_, reward, done = self.env.step(act)
        self.transition = [state, act, reward, state_, done]
        if self.env.ifstd == True:
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return state_, reward, done

    def run(self,
            pg: str,
            index: int,
            date: str,
            rounds: int = 1000,
            plotting_period: int = 200):
        
        frames = self.env.getNcls() * rounds
        # self.epsilon_decay = (
        #                          np.log((1 / self.env.ncls - self.min_epsilon) /
        #                                 (self.max_epsilon - self.min_epsilon))
        #                      ) / (- Aframes * self.x)
        score = 0
        update_cnt = 0
        round_cnt = 0
        lost_frames = 0
        losses = []
        scores = []
        epsilons = []
        data = []
        steps = []

        dirname = pg + "_" + date

        wt = SummaryWriter(log_dir=f"tensorboard_data/{dirname}/{pg}_{index}")
        for frame_idx in tqdm(range(1, frames + 1)):
            act = self.select_action(self.env.state)
            state_, reward, done = self.step(act)
            score += reward
            frac = min((frame_idx - lost_frames) / (frames - lost_frames), 1.0) if self.env.ifstd == True else 0
            self.beta = self.beta + frac * (1.0 - self.beta)

            if len(self.memory) >= self.batch_size and self.env.ifstd == True:
                loss = self._update_model()
                losses.append(loss)
                wt.add_scalar("loss", loss, frame_idx)
                update_cnt += 1

                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-1. * (frame_idx - lost_frames) * self.epsilon_decay)
                epsilons.append(self.epsilon)

                # if hard update is needed
                if update_cnt % self.update_period == 0:
                    self._target_hard_update()


            if done:
                # start z-score standardization for profit if the number of frames is equal to or more than 100
                if self.env.ifstd == True:
                    if self.env.getBest:
                        data.append(copy.deepcopy(self.env.getBestResult()))
                    if self.env.selected.count(True) == self.env.steps:
                        steps.append(self.env.steps)
                        wt.add_scalar("steps",
                                      self.env.steps, frame_idx)  # change
                        wt.add_scalar("rcpl_ocplx",
                                      1 / self.env.cost, frame_idx)
                    else:
                        steps.append(self.env.steps - 1)
                        wt.add_scalar("steps",
                                      self.env.steps - 1, frame_idx)  # change
                        wt.add_scalar("rcpl_ocplx",
                                      0, frame_idx)
                    scores.append(score)
                    wt.add_scalar("score", score, frame_idx)
                    wt.add_scalar("epsilon", self.epsilon, frame_idx)

                self.env.reset()
                score = 0
                round_cnt += 1

                if frame_idx >= 1000 and self.env.ifstd == False:
                    self.env.ifstd = True
                    lost_frames = frame_idx + 1
                    self.epsilon_decay = (np.log((1 / self.env.ncls - self.min_epsilon) /
                        (self.max_epsilon - self.min_epsilon))) / (- (frames - lost_frames) * self.x)

        t = time.clock() - self.env.st
        wt.close()
        return data, t

    def _plot(
            self,
            frame_idx: int,
            scores,
            losses,
            epsilons,
            steps
    ):
        """Plot the training progresses."""
        # clear_output(True)
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.ylabel("score")
        plt.xlabel("round")
        plt.plot(scores)
        plt.subplot(2, 2, 2)
        plt.title('loss')
        plt.ylabel("smooth_l1_loss")
        plt.xlabel("frame - %d" % self.batch_size)
        plt.plot(losses)
        plt.subplot(2, 2, 3)
        plt.title('epsilons')
        plt.ylabel("probability")
        plt.xlabel("frame")
        plt.plot(epsilons)
        plt.subplot(2, 2, 4)
        plt.title('steps')
        plt.ylabel("length")
        plt.xlabel("round")
        plt.plot(steps)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=0.5, hspace=0.5)
        plt.show()

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.target_model.load_state_dict(self.current_model.state_dict())


    def _update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    @nb.jit()
    def _compute_dqn_loss(self, samples, gamma: float):
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.current_model(state).gather(1, action)
        next_q_value = self.target_model(next_state).gather(  # Double DQN
            1, self.current_model(next_state).argmax(dim=1, keepdim=True)
        ).detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return elementwise_loss
