import torch
import torch.nn as nn
import torch.nn.functional as F

from distributions import Categorical, DiagGaussian
from utils import init, init_normc_


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, meta=False, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        
        self.meta = meta
        
        if len(obs_shape) == 3 and not meta:
            self.base = CNNBase(obs_shape[0], **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        elif len(obs_shape) == 3 and self.meta:
            self.base = CNNMetaBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, actions=None, prev_rewards=None, prev_actions=None, infos=None, deterministic=False):
        if self.meta:
            value, actor_features, rnn_hxs = self.base(inputs, prev_actions, prev_rewards, rnn_hxs, masks, infos)
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, actions, prev_rewards, prev_actions, infos=None):
        if self.meta:
            value, _, _ = self.base(inputs, prev_actions, prev_rewards, rnn_hxs, masks, infos)
        else:
            value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, prev_rewards=None, prev_actions=None, infos=None):
        if self.meta:
            value, actor_features, rnn_hxs = self.base(inputs, prev_actions, prev_rewards, rnn_hxs, masks, infos)
        else:
            value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks, prev_action=None, prev_reward=None, infos=None):
        # convert action to one hot vector
        if prev_action is not None:
            prev_action = torch.nn.functional.one_hot(prev_action, 4).squeeze(1)

        if x.size(0) == hxs.size(0):
            if prev_action is not None and prev_reward is not None:
                # [16, 128] -> [16, 133]
                x = torch.cat([x, prev_action.float(), prev_reward.float()], dim=1)
            if infos is not None:
                # [16, 128] -> [16, 133] -> [16, 139]
                x = torch.cat([x, infos.float()], dim=1)
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))
        
            if prev_action is not None and prev_reward is not None:
                prev_action = prev_action.view(T, N, prev_action.size(1))
                prev_reward = prev_reward.view(T, N, prev_reward.size(1))

                # [80, 1, 128] -> [80, 1, 133]
                x = torch.cat([x, prev_action.float(), prev_reward.float()], dim=2)

            if infos is not None:
                infos = infos.view(T, N, infos.size(1))
                # [80, 1, 128] -> [80, 1, 133] -> [80, 1, 139]
                x = torch.cat([x, infos.float()], dim=2)

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class NNStackedBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNStackedBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            # TODO: consider producing hidden_size - 4 here
            self.gru1 = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru1.weight_ih.data)
            nn.init.orthogonal_(self.gru1.weight_hh.data)
            self.gru1.bias_ih.data.fill_(0)
            self.gru1.bias_hh.data.fill_(0)

            self.gru2 = nn.GRUCell(hidden_size, hidden_size)
            nn.init.orthogonal_(self.gru2.weight_ih.data)
            nn.init.orthogonal_(self.gru2.weight_hh.data)
            self.gru2.bias_ih.data.fill_(0)
            self.gru2.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks, prev_action=None, prev_reward=None):
        # convert action to one hot vector
        prev_action = torch.nn.functional.one_hot(prev_action, 4).squeeze(1)

        if x.size(0) == hxs.size(0):
            # [16, 128] -> [16, 129]
            x = torch.cat([x, prev_reward.float(), torch.zeros_like(prev_action).float()], dim=1)
            x = self.gru1(x, hxs * masks)
            # [16, 129] -> [16, 133]
            x[:,129:] = prev_action.float()[:,:]
            hxs = x
            x = hxs = self.gru2(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))
        
            prev_action = prev_action.view(T, N, prev_action.size(1))
            prev_reward = prev_reward.view(T, N, prev_reward.size(1))

            # [80, 1, 128] -> [80, 1, 129]
            x = torch.cat([x, prev_reward.float(), torch.zeros_like(prev_action).float()], dim=2)

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = self.gru1(x[i], hxs * masks[i])
                hx[:,129:] = prev_action[i].float()[:,:]
                hxsi = hx
                hx = hxsi = self.gru2(hx, hxsi * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
        return x, hxs


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('layer input:', x.shape)
        return x


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=128):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        # For 80x60 input
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            #Print(),
            Flatten(),

            #nn.Dropout(0.2),

            init_(nn.Linear(32 * 7 * 5, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        #print(inputs.size())

        x = inputs / 255.0
        #print(x.size())

        x = self.main(x)
        #print(x.size())

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class CNNMetaBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=133, use_info=True):
        hidden_size = hidden_size + 6 if use_info else hidden_size
        super(CNNMetaBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        # For 80x60 input
        # -4 from action, -1 from reward, -6 for info
        output_size = hidden_size - 1 - 4
        output_size = output_size - 6 if use_info else output_size
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            #Print(),
            Flatten(),

            #nn.Dropout(0.2),

            init_(nn.Linear(32 * 7 * 5, output_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, prev_actions, prev_rewards, rnn_hxs, masks, infos=None):
        #print(x.size())

        x = inputs / 255.0
        #print(x.size())

        x = self.main(x)
        #print(x.size())

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks, prev_actions, prev_rewards, infos)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
