from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
from torch import nn
from copy import deepcopy
from ding.torch_utils import Adam, to_device
from ding.rl_utils import get_train_sample
from ding.utils import POLICY_REGISTRY, deep_merge_dicts
from ding.utils.data import default_collate, default_decollate
from ding.policy import Policy
from ding.model import model_wrap
from ding.policy.common_utils import default_preprocess_learn

from .utils import imagine, compute_target, compute_actor_loss, RewardEMA, tensorstats
from ding.torch_utils.network.dreamer import Optimizer


@POLICY_REGISTRY.register('dreamer')
class DREAMERPolicy(Policy):
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='dreamer',
        # (bool) Whether to use cuda for network and loss computation.
        cuda=False,
        # (int) Number of training samples (randomly collected) in replay buffer when training starts.
        random_collect_size=5000,
        # (bool) Whether to need policy-specific data in preprocess transition.
        transition_with_policy_data=False,
        # (int)
        imag_horizon=15,
        learn=dict(
            # (float) Lambda for TD-lambda return.
            lambda_=0.95,
            # (float) Max norm of gradients.
            grad_clip=100,
            learning_rate=3e-5,
            ac_opt_eps=1e-5,
            precision=32,
            opt='adam',
            weight_decay=0.0,
            value_grad_clip=100,
            actor_grad_clip=100,
            batch_size=16,
            batch_length=64,
            imag_sample=True,
            slow_value_target=True,
            slow_target_update=1,
            slow_target_fraction=0.02,
            discount=0.997,
            reward_EMA=True,
            actor_entropy=3e-4,
            actor_state_entropy=0.0,
            value_decay=0.0,
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        return 'dreamervac', ['ding.model.template.vac']

    def _init_learn(self) -> None:
        r"""
        Overview:
            Learn mode init method. Called by ``self.__init__``.
            Init the optimizer, algorithm config, main and target models.
        """
        # Algorithm config
        self._use_amp = True if self._cfg.learn.precision == 16 else False
        self._lambda = self._cfg.learn.lambda_
        self._grad_clip = self._cfg.learn.grad_clip

        self._critic = self._model.critic
        self._actor = self._model.actor

        if self._cfg.learn.slow_value_target:
            self._slow_value = deepcopy(self._critic)
            self._updates = 0

        # # Optimizer
        # self._optimizer_value = Adam(
        #     self._critic.parameters(),
        #     lr=self._cfg.learn.learning_rate,
        # )
        # self._optimizer_actor = Adam(
        #     self._actor.parameters(),
        #     lr=self._cfg.learn.learning_rate,
        # )
        
        kw = dict(wd=self._cfg.learn.weight_decay, opt=self._cfg.learn.opt, use_amp=self._use_amp)
        self._optimizer_actor = Optimizer(
            "actor",
            self._actor.parameters(),
            self._cfg.learn.learning_rate,
            self._cfg.learn.ac_opt_eps,
            self._cfg.learn.actor_grad_clip,
            **kw,
        )
        self._optimizer_value = Optimizer(
            "critic",
            self._critic.parameters(),
            self._cfg.learn.learning_rate,
            self._cfg.learn.ac_opt_eps,
            self._cfg.learn.value_grad_clip,
            **kw,
        )

        self._learn_model = model_wrap(self._model, wrapper_name='base')
        self._learn_model.reset()

        self._forward_learn_cnt = 0

        if self._cfg.learn.reward_EMA:
            self.reward_ema = RewardEMA(device=self._device)

    def _forward_learn(self, start: dict, world_model, envstep) -> Dict[str, Any]:
        # log dict
        log_vars = {}
        self._learn_model.train()
        self._update_slow_target()

        start = {k: v[:-1] for k, v in start.items()}
        # start is dict of {stoch, deter, logit}

        # world_model.load_state_dict(torch.load('/home/PJLAB/zhangpai/duibi/world_model_origin.pth', map_location=self._device))
        # start = torch.load('/home/PJLAB/zhangpai/duibi/start_origin.pth')

        if self._cuda:
            start = to_device(start, self._device)
        
        self._actor.requires_grad_(requires_grad=True)
        # train self._actor
        with torch.cuda.amp.autocast(self._use_amp):
            imag_feat, imag_state, imag_action = imagine(
                self._cfg.learn, world_model, start, self._actor, self._cfg.imag_horizon
            )
            # torch.save(imag_feat, '/home/PJLAB/zhangpai/duibi/imag_feat.pth')
            # torch.save(imag_state, '/home/PJLAB/zhangpai/duibi/imag_state.pth')
            # torch.save(imag_action, '/home/PJLAB/zhangpai/duibi/imag_action.pth')
            # imag_feat = torch.load('imag_feat_origin.pth')
            # imag_state = torch.load('imag_state_origin.pth')
            # imag_action = torch.load('imag_action_origin.pth')
            imag_feat = to_device(imag_feat, self._device)
            imag_state = to_device(imag_state, self._device)
            imag_action = to_device(imag_action, self._device)
            reward = world_model.heads["reward"](world_model.dynamics.get_feat(imag_state)).mode()
            actor_ent = self._actor(imag_feat).entropy()
            state_ent = world_model.dynamics.get_dist(imag_state).entropy()
            # this target is not scaled
            # slow is flag to indicate whether slow_target is used for lambda-return
            target, weights, base = compute_target(
                self._cfg.learn, world_model, self._critic, imag_feat, imag_state, reward, actor_ent, state_ent
            )
            actor_loss, mets = compute_actor_loss(
                self._cfg.learn,
                self._actor,
                self.reward_ema,
                imag_feat,
                imag_state,
                imag_action,
                target,
                actor_ent,
                state_ent,
                weights,
                base,
            )
            log_vars.update(mets)
            value_input = imag_feat
        self._actor.requires_grad_(requires_grad=False)

        self._critic.requires_grad_(requires_grad=True)
        with torch.cuda.amp.autocast(self._use_amp):
            value = self._critic(value_input[:-1].detach())
            # to do
            target = torch.stack(target, dim=1)
            # (time, batch, 1), (time, batch, 1) -> (time, batch)
            value_loss = -value.log_prob(target.detach())
            slow_target = self._slow_value(value_input[:-1].detach())
            if self._cfg.learn.slow_value_target:
                value_loss = value_loss - value.log_prob(slow_target.mode().detach())
            if self._cfg.learn.value_decay:
                value_loss += self._cfg.learn.value_decay * value.mode()
            # (time, batch, 1), (time, batch, 1) -> (1,)
            value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])
        self._critic.requires_grad_(requires_grad=False)

        log_vars.update(tensorstats(value.mode(), "value"))
        log_vars.update(tensorstats(target, "target"))
        log_vars.update(tensorstats(reward, "imag_reward"))
        log_vars.update(tensorstats(imag_action, "imag_action"))
        log_vars["actor_ent"] = torch.mean(actor_ent).detach().cpu().numpy().item()
        # ====================
        # actor-critic update
        # ====================
        self._model.requires_grad_(requires_grad=True)
        world_model.requires_grad_(requires_grad=True)

        # loss_dict = {
        #     'critic_loss': value_loss,
        #     'actor_loss': actor_loss,
        # }

        # norm_dict = self._update(loss_dict)
        # torch.save(value_loss, '/home/PJLAB/zhangpai/duibi/value_loss.pth')
        # torch.save(actor_loss, '/home/PJLAB/zhangpai/duibi/actor_loss.pth')
        log_vars.update(self._optimizer_actor(actor_loss, self._model.actor.parameters()))
        log_vars.update(self._optimizer_value(value_loss, self._model.critic.parameters()))

        self._model.requires_grad_(requires_grad=False)
        world_model.requires_grad_(requires_grad=False)
        # =============
        # after update
        # =============
        self._forward_learn_cnt += 1

        # return {
        #     **log_vars,
        #     **norm_dict,
        #     **loss_dict,
        # }
        return log_vars

    def _update(self, loss_dict):
        # update actor
        self._optimizer_actor.zero_grad()
        loss_dict['actor_loss'].backward()
        actor_norm = nn.utils.clip_grad_norm_(self._model.actor.parameters(), self._grad_clip)
        self._optimizer_actor.step()
        # update critic
        self._optimizer_value.zero_grad()
        loss_dict['critic_loss'].backward()
        critic_norm = nn.utils.clip_grad_norm_(self._model.critic.parameters(), self._grad_clip)
        self._optimizer_value.step()
        return {'actor_grad_norm': actor_norm, 'critic_grad_norm': critic_norm}

    def _update_slow_target(self):
        if self._cfg.learn.slow_value_target:
            if self._updates % self._cfg.learn.slow_target_update == 0:
                mix = self._cfg.learn.slow_target_fraction
                for s, d in zip(self._critic.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
    
    def _state_dict_learn(self) -> Dict[str, Any]:
        ret = {
            'model': self._learn_model.state_dict(),
            #'optimizer_value': self._optimizer_value.state_dict(),
            #'optimizer_actor': self._optimizer_actor.state_dict(),
        }
        return ret

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        self._learn_model.load_state_dict(state_dict['model'])
        self._optimizer_value.load_state_dict(state_dict['optimizer_value'])
        self._optimizer_actor.load_state_dict(state_dict['optimizer_actor'])

    def _init_collect(self) -> None:
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='base')
        self._collect_model.reset()

    def _forward_collect(self, data: dict, world_model, envstep, reset=None, state=None) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        
        if state is None:
            batch_size = len(data_id)
            latent = world_model.dynamics.initial(batch_size)  # {logit, stoch, deter}
            action = torch.zeros((batch_size, self._cfg.collect.action_size)).to(
                self._device
            )
        else:
            #state = default_collate(list(state.values()))
            latent = to_device(default_collate(list(zip(*state))[0]), self._device)
            action = to_device(default_collate(list(zip(*state))[1]), self._device)
            if len(action.shape)==1:
                action = action.unsqueeze(-1)
            if reset.any():
                mask = 1 - reset
                for key in latent.keys():
                    for i in range(latent[key].shape[0]):
                        latent[key][i] *= mask[i]
                for i in range(len(action)):
                    action[i] *= mask[i]
        
        data = data - 0.5
        embed = world_model.encoder(data)
        latent, _ = world_model.dynamics.obs_step(
            latent, action, embed, self._cfg.collect.collect_dyn_sample
        )
        feat = world_model.dynamics.get_feat(latent)
        
        actor = self._actor(feat)
        action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        
        state = (latent, action)
        output = {"action": action, "logprob": logprob, "state": state}
        
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}
    
    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> dict:
        r"""
        Overview:
            Generate dict type transition data from inputs.
        Arguments:
            - obs (:obj:`Any`): Env observation
            - model_output (:obj:`dict`): Output of collect model, including at least ['action']
            - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
                (here 'obs' indicates obs after env step).
        Returns:
            - transition (:obj:`dict`): Dict type transition data.
        """
        transition = {
            'obs': obs,
            #'next_obs': timestep.obs,
            'action': model_output['action'],
            # TODO(zp) random_collect just have action
            #'logprob': model_output['logprob'],
            'reward': timestep.reward,
            'discount': timestep.info['discount'],
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, data: list) -> Union[None, List[Any]]:
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        self._eval_model = model_wrap(self._model, wrapper_name='base')
        self._eval_model.reset()

    def _forward_eval(self, data: dict, world_model, reset=None, state=None) -> dict:
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        
        if state is None:
            batch_size = len(data_id)
            latent = world_model.dynamics.initial(batch_size)  # {logit, stoch, deter}
            action = torch.zeros((batch_size, self._cfg.collect.action_size)).to(
                self._device
            )
        else:
            #state = default_collate(list(state.values()))
            latent = to_device(default_collate(list(zip(*state))[0]), self._device)
            action = to_device(default_collate(list(zip(*state))[1]), self._device)
            if len(action.shape)==1:
                action = action.unsqueeze(-1)
            if reset.any():
                mask = 1 - reset
                for key in latent.keys():
                    for i in range(latent[key].shape[0]):
                        latent[key][i] *= mask[i]
                for i in range(len(action)):
                    action[i] *= mask[i]
        
        data = data - 0.5
        embed = world_model.encoder(data)
        latent, _ = world_model.dynamics.obs_step(
            latent, action, embed, self._cfg.collect.collect_dyn_sample
        )
        feat = world_model.dynamics.get_feat(latent)
        
        actor = self._actor(feat)
        action = actor.mode()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()

        state = (latent, action)
        output = {"action": action, "logprob": logprob, "state": state}
        
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> List[str]:
        r"""
        Overview:
            Return variables' name if variables are to used in monitor.
        Returns:
            - vars (:obj:`List[str]`): Variables' name list.
        """
        return [
            'normed_target_mean', 'normed_target_std', 'normed_target_min', 'normed_target_max', 'EMA_005', 'EMA_095',
            'actor_entropy', 'actor_state_entropy', 'value_mean', 'value_std', 'value_min', 'value_max', 'target_mean',
            'target_std', 'target_min', 'target_max', 'imag_reward_mean', 'imag_reward_std', 'imag_reward_min',
            'imag_reward_max', 'imag_action_mean', 'imag_action_std', 'imag_action_min', 'imag_action_max', 'actor_ent',
            'actor_loss', 'critic_loss', 'actor_grad_norm', 'critic_grad_norm'
        ]
