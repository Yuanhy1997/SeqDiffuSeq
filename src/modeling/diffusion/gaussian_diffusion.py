"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.

----
Aman's notes:

- `q` always refers to the forward diffusion process, and `p` always refers to the reverse diffusion process. `p` is learned, `q` is deterministic.

- DDIM has been removed.
"""
import torch.distributed as dist

import enum
import math

import numpy as np
import torch as th

from src.modeling.diffusion.nn import mean_flat
from src.modeling.diffusion.losses import normal_kl

from src.utils.show_sampling_progress import pprint_sentences
import os

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_trunc_cosine(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == "trunc_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "pw_lin":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  # scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64
        )
        return np.concatenate([first_part, second_part])

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar_trunc_cosine(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = enum.auto()  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    E2E_KL = enum.auto()
    E2E_MSE = enum.auto()
    E2E_Simple_MSE = enum.auto()
    E2E_Simple_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        model_arch=None,
        training_mode="emb",
        token_max_length=None,
        save_dir=None,
        pad_tok_id=None,
        loss_update_granu=None,
        schedule_update_stride=0,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.model_arch = model_arch
        self.pad_tok_id = pad_tok_id
        assert self.pad_tok_id is not None

        self.token_max_length = token_max_length
        self.save_dir = save_dir

        print("$"*10, self.save_dir)

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self._loss_update_time = 0
        self._loss_interp_granu = int(loss_update_granu)
        assert self._loss_interp_granu is not None
        self._loss_history_update_stride = schedule_update_stride
        print('schedule update stride', self._loss_history_update_stride)
        self._loss_history = np.ones((self.num_timesteps//self._loss_interp_granu, self.token_max_length)) * np.linspace(0, 0.5, self.num_timesteps//self._loss_interp_granu)[:,None]
        self._loss_history_count = np.ones((self.num_timesteps//self._loss_interp_granu, self.token_max_length))

        alphas = 1.0 - betas

        if len(betas.shape) < 2:
            alphas = np.expand_dims(alphas, 1)
            alphas = np.tile(alphas, (1, self.token_max_length))
            betas = 1.0 - alphas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.vstack((np.ones((1, self.token_max_length)), self.alphas_cumprod[:-1]))
        self.alphas_cumprod_next = np.vstack((self.alphas_cumprod[1:], np.zeros((1, self.token_max_length))))
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps, self.token_max_length)
        self.alpha_cumprod_range = np.max(self.alphas_cumprod) - np.min(self.alphas_cumprod)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

        self.training_mode = training_mode
        print("training mode is ", training_mode)
    
    def update_time_discretized_parameters(self, alphas_cumprod):

        self.alphas_cumprod[:, 1:] = alphas_cumprod[:, 1:] # only change schedule of tokens other than bos token
        alphas = np.zeros_like(alphas_cumprod)
        for i in range(len(alphas_cumprod)):
            if i == 0:
                alphas[i] = self.alphas_cumprod[i]
            else:
                alphas[i] = self.alphas_cumprod[i] / self.alphas_cumprod[i-1]
        betas = 1.0 - alphas

        if self.token_max_length is not None:
            self.alphas_cumprod_prev = np.vstack((np.ones((1, self.token_max_length)), self.alphas_cumprod[:-1]))
            self.alphas_cumprod_next = np.vstack((self.alphas_cumprod[1:], np.zeros((1, self.token_max_length))))
            assert self.alphas_cumprod_prev.shape == (self.num_timesteps, self.token_max_length)
        else:
            self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
            self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
            assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _load_time_schedule(self, path):

        alphas_cumprod = np.load(path)
        self.update_time_discretized_parameters(alphas_cumprod)
    
    def _loss_history_update(self, ts, losses, loss_masks, training_step): #v5
        """
        ts is a vector of shape B
        losses is a tensor of shape BxS

        self._loss_history is TxS
        """
        all_losses = []
        losses_gather_buffer = [th.zeros_like(losses) for _ in range(dist.get_world_size())]
        dist.all_gather(losses_gather_buffer, losses.detach())
        all_losses.extend([sample.cpu().numpy() for sample in losses_gather_buffer])
        all_losses = np.concatenate(all_losses, axis=0) #BXS

        all_loss_masks = []
        loss_masks_gather_buffer = [th.zeros_like(loss_masks) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_masks_gather_buffer, loss_masks.detach())
        all_loss_masks.extend([sample.cpu().numpy() for sample in loss_masks_gather_buffer])
        all_loss_masks = np.concatenate(all_loss_masks, axis=0) #BxS

        all_ts = []
        ts_gather_buffer = [th.zeros_like(ts) for _ in range(dist.get_world_size())]
        dist.all_gather(ts_gather_buffer, ts.detach())
        all_ts.extend([sample.cpu().numpy() for sample in ts_gather_buffer])
        all_ts = np.concatenate(all_ts, axis=0) #B.  0-1999

        all_ts = all_ts // self._loss_interp_granu # 0-99 self._loss_interp_granu=20
        
        for t, loss, loss_m in zip(all_ts, all_losses, all_loss_masks):
            self._loss_history[t] += loss
            self._loss_history_count[t] += loss_m.astype(float)  #loss_m 64 shape.  [1,1,1,0,0,0,0,0,0,0]

        if training_step >= (self._loss_history_update_stride*3) and training_step % self._loss_history_update_stride == 0:
            interp_alpha_cumprod = []
            loss_dist = self._loss_history / self._loss_history_count # TxS
            for i in range(loss_dist.shape[0]):
                if i > 0:
                    loss_dist[i, :] = np.max([loss_dist[i, :], loss_dist[i-1, :]+1e-5], axis=0)
            loss_dist = np.vstack([loss_dist[:1, :]-(loss_dist[1:2, :]-loss_dist[:1, :])/2, loss_dist, loss_dist[-1:, :]+(loss_dist[-1:, :]-loss_dist[-2:-1, :])/2])

            for s in range(loss_dist.shape[1]):
                loss_val = np.linspace(np.min(loss_dist[:, s])-1e-5, np.max(loss_dist[:, s])+1e-5, self.num_timesteps)
                alpha_cumprod_dist = np.mean(self.alphas_cumprod[:, s].reshape(-1, self._loss_interp_granu), axis=1)
                alpha_cumprod_dist = np.append(np.max(self.alphas_cumprod), alpha_cumprod_dist)
                alpha_cumprod_dist = np.append(alpha_cumprod_dist, np.min(self.alphas_cumprod))
                interp_alpha_cumprod.append(np.interp(loss_val, loss_dist[:, s], alpha_cumprod_dist))

            
            interp_alpha_cumprod = np.stack(interp_alpha_cumprod).transpose(1,0)
            self.update_time_discretized_parameters(interp_alpha_cumprod)
            
            if dist.get_rank() == 0:
                print('*'*10, f'updated alpha_cumprod to /alpha_cumprod_step_{training_step}.npy', '*'*10)
                np.save(os.path.join(self.save_dir, f'alpha_cumprod_step_{training_step}.npy'), self.alphas_cumprod)
                np.save(os.path.join(self.save_dir, f'loss_step_{training_step}.npy'), self._loss_history)
                np.save(os.path.join(self.save_dir, f'loss_count_{training_step}.npy'), self._loss_history_count)

            self._loss_history = np.ones((self.num_timesteps//self._loss_interp_granu, self.token_max_length)) * np.linspace(0, 0.5, self.num_timesteps//self._loss_interp_granu)[:,None]
            self._loss_history_count = np.ones((self.num_timesteps//self._loss_interp_granu, self.token_max_length))
                
    def training_losses(self, model, training_step, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. It is NEVER used -- the embeddings are recreated every time from the input IDs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        assert "input_ids" in model_kwargs
        assert "decoder_input_ids" in model_kwargs
        input_ids = model_kwargs.pop("decoder_input_ids").to(t.device)
        if 'loss_mask' in model_kwargs:
            loss_mask = model_kwargs.pop('loss_mask').to(t.device)
        else:
            loss_mask = None
        x_start_mean = model.model.module.get_embeds(input_ids)

        std = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod,
            th.tensor([0]).to(x_start_mean.device),
            x_start_mean.shape,
        )
        x_start = self.get_x_start(x_start_mean, std)

        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)  # reparametrization trick.

        get_logits = model.model.module.get_logits

        ### self-conditioning part
        model_kwargs['self_conditions'] = th.zeros_like(x_t)
        if np.random.uniform() > 0.5:
            with th.no_grad():
                model_output = model(x = x_t, ts = self._scale_timesteps(t), **model_kwargs)
            model_kwargs['self_conditions'] = model_output.detach()
                        
        model_output = model(x = x_t, ts = self._scale_timesteps(t), **model_kwargs)

        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[
                0
            ],
            ModelMeanType.START_X: x_start,  # THIS is actually used
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]

        assert (
            model_output.shape == target.shape == x_start.shape
        ), f"model_output.shape: {model_output.shape}, target.shape: {target.shape}, x_start.shape: {x_start.shape}"
        # the usual diffusion loss
        terms = {}
        terms["mse"] = mean_flat((target - model_output) ** 2, loss_mask)
        model_out_x_start = self.x0_helper(model_output, x_t, t)["pred_xstart"]
        t0_mask = t == 0
        t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2, loss_mask)
        terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])

        ### adaptive noise schedule logging part
        with th.no_grad():
            mse_loss_log_ = th.mean((target - model_output) ** 2, dim = -1).detach()
            t0_loss_log_ = th.mean((x_start_mean - model_out_x_start) ** 2, dim = -1).detach()
            _loss_log = mse_loss_log_
            _loss_log[t0_mask] = t0_loss_log_[t0_mask]
            _loss_log[input_ids==self.pad_tok_id] = 0
            self._loss_history_update(t, _loss_log, input_ids!=self.pad_tok_id, training_step)

        out_mean, _, _ = self.q_mean_variance(
            x_start, th.LongTensor([self.num_timesteps - 1]).to(x_start.device)
        )
        tT_loss = mean_flat(out_mean**2)
        
        decoder_nll = self.token_discrete_loss(x_start, get_logits, input_ids, mask=loss_mask)

        terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)

        return terms

    def get_x_start(self, x_start_mean, std):
        """
        Using the interpolating policy OR using the convolution policy...
        :param x_start_mean:
        :return:
        """
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return x_start_mean + std * noise

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        - Equation (7) of DDPM paper.

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def token_discrete_loss(self, x_t, get_logits, input_ids, mask=None):
        logits = get_logits(x_t)  # bsz, seqlen, vocab
        loss_fct = th.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(
            input_ids.shape
        )
        if mask is not None:
            decoder_nll[mask == 0] = 0
            decoder_nll = decoder_nll.sum(dim=-1) / mask.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)

        return decoder_nll

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.size(0), x.size(-1)
        # B -> batch size, C -> channel size (embedding size)
        assert t.shape == (B,)
        if 'loss_mask' in model_kwargs:
            model_kwargs.pop('loss_mask')
        if 'self_conditions' not in model_kwargs:
            model_kwargs["self_conditions"] = th.zeros_like(x)
            
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        model_kwargs["self_conditions"] = model_output
          

        model_variance, model_log_variance = {
            # for fixedlarge, we set the initial (log-)variance like so
            # to get a better decoder log likelihood.
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        top_p=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"],
            "out": out,
        }

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        langevin_func=None,
        decoder_inputs = None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        
        print(model_kwargs.keys())
        if "encoder_outputs" not in model_kwargs:
            t = th.tensor([10] * shape[0], device=device)
            if 'self_conditions' not in model_kwargs:
                model_kwargs['self_conditions'] = th.zeros_like(img)
            with th.no_grad():
                model_kwargs["encoder_outputs"] = (model.forward_encoder(decoder_inputs_embeds = img, 
                                                                        timesteps = self._scale_timesteps(t), 
                                                                        **model_kwargs), )
            model_kwargs.pop('input_ids')
            if 'self_conditions' in model_kwargs:
                model_kwargs.pop('self_conditions')

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                img = out["sample"]

            
    def p_sample_loop_progressive_mix_sample(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        langevin_func=None,
        decoder_inputs = None,
        generate_by_mix_prob=0,
        generate_by_mix_part=1,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
    
        print(model_kwargs.keys())

        if "encoder_outputs" not in model_kwargs:
            t = th.tensor([10] * shape[0], device=device)
            if 'self_conditions' not in model_kwargs:
                model_kwargs['self_conditions'] = th.zeros_like(img)
            with th.no_grad():
                model_kwargs["encoder_outputs"] = (model.forward_encoder(decoder_inputs_embeds = img, 
                                                                        timesteps = self._scale_timesteps(t), 
                                                                        **model_kwargs), )
            model_kwargs.pop('input_ids')
            if 'self_conditions' in model_kwargs:
                model_kwargs.pop('self_conditions')
        
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                if np.random.uniform() > 1 - generate_by_mix_prob and t[0] > (1-generate_by_mix_part) * self.num_timesteps:
                    img = self.q_sample(out['pred_xstart'], t-1)
                else:
                    img = out["sample"]
            
    def p_sample_loop_progressive_by_q_sample(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        langevin_func=None,
        decoder_inputs = None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        if "encoder_outputs" not in model_kwargs:
            t = th.tensor([10] * shape[0], device=device)
            if 'self_conditions' not in model_kwargs:
                model_kwargs['self_conditions'] = th.zeros_like(img)
            with th.no_grad():
                model_kwargs["encoder_outputs"] = (model.forward_encoder(decoder_inputs_embeds = img, 
                                                                        timesteps = self._scale_timesteps(t), 
                                                                        **model_kwargs), )
            model_kwargs.pop('input_ids')
            if 'self_conditions' in model_kwargs:
                model_kwargs.pop('self_conditions')
        
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                )
                yield out
                if i > 0:
                    img = self.q_sample(out['pred_xstart'], t-1)
                else:
                    img = out["sample"]
            
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        tokenizer=None,
        log_verbose=False,
        logging_freq: int = 100,
        num_samples_to_show: int = 1,
        langevin_fn=None,
        decoder_inputs = None,
        generate_by_q=False,
        generate_by_mix=False,
        generate_by_mix_prob=0,
        generate_by_mix_part=0,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None

        if generate_by_q:
            loop_fn = self.p_sample_loop_progressive_by_q_sample
        
        elif generate_by_mix:

            loop_fn = self.p_sample_loop_progressive_mix_sample

        else:
            loop_fn = self.p_sample_loop_progressive
            
        if generate_by_mix:
            for i, sample in enumerate(
                loop_fn(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    top_p=top_p,
                    langevin_func=langevin_fn,
                    decoder_inputs=decoder_inputs,
                    generate_by_mix_part=generate_by_mix_part,
                    generate_by_mix_prob=generate_by_mix_prob,
                )
            ):
                final = sample

            return final["sample"]
        
        else:
            for i, sample in enumerate(
                loop_fn(
                    model,
                    shape,
                    noise=noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    device=device,
                    progress=progress,
                    top_p=top_p,
                    langevin_func=langevin_fn,
                    decoder_inputs=decoder_inputs,
                )
            ):
                final = sample

            return final["sample"]

    def _vb_terms_bpd_e2e(
        self,
        model,
        x_start,
        x_t,
        t,
        input_ids,
        get_logits,
        x_start_mean,
        x_start_log_var,
        clip_denoised=True,
        model_kwargs=None,
        noise=None,
        denoised_fn=None,
        self_condition=False,
        self_condition_as_train=False,
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # lambda *args, r=frozen_out: r,
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        assert input_ids is not None

        out = self.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
            self_condition=self_condition,
            self_condition_as_train=self_condition_as_train,
        )

        with th.no_grad():
            model_output = out['pred_xstart']
            mse_loss = th.mean((x_start - model_output) ** 2, dim = -1)

            logits = get_logits(model_output)
            nll_loss = th.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), 
                                                      input_ids.view(-1), 
                                                      reduction="none").view(input_ids.shape)
            
            kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
            kl = th.mean(kl, dim = -1) / np.log(2.0)

        return {
            "pred_xstart": out["pred_xstart"],
            "kl": kl,
            "mse": mse_loss,
            "nll": nll_loss,
        }


    def x0_helper(self, model_output, x, t):
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            pred_prev = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = model_output
            else:
                pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            pred_prev, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        else:
            raise NotImplementedError(self.model_mean_type)
        return {"pred_xprev": pred_prev, "pred_xstart": pred_xstart}

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop_e2e(
        self, model, input_ids, clip_denoised=True, model_kwargs=None, denoised_fn=None, self_condition=False,
        self_condition_as_train=False,
    ):

        device = input_ids.device
        batch_size = input_ids.shape[0]

        x_start_mean = model.get_embeds(input_ids)

        std = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod,
            th.tensor([0]).to(x_start_mean.device),
            x_start_mean.shape,
        )
        x_start_log_var = 2 * th.log(std)
        x_start = self.get_x_start(x_start_mean, std)
        get_logits = model.get_logits


        mse_logger = th.zeros((self.num_timesteps, input_ids.shape[-1])).to(x_start_mean.device)
        nll_logger = th.zeros((self.num_timesteps, input_ids.shape[-1])).to(x_start_mean.device)
        kl_logger = th.zeros((self.num_timesteps, input_ids.shape[-1])).to(x_start_mean.device)


        from tqdm.auto import tqdm

        for t in tqdm(list(range(self.num_timesteps))[::-1]):
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd_e2e(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    input_ids=input_ids,
                    get_logits=get_logits,
                    x_start_mean=x_start_mean,
                    x_start_log_var=x_start_log_var,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    denoised_fn=denoised_fn,
                    self_condition=self_condition,
                    self_condition_as_train=self_condition_as_train,
                )
            
            mse_logger[t] = th.mean(out['mse'], dim=0)
            nll_logger[t] = th.mean(out['nll'], dim=0)
            kl_logger[t] = th.mean(out['kl'], dim=0)
        
        return mse_logger, nll_logger, kl_logger


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
