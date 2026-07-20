import os
import time
from torchvision.utils import make_grid
import ipdb        
import numpy as np
import tqdm
import torch
import torch.nn as nn
from utils.sampling import generalized_steps
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionMA
from tensorboardX import SummaryWriter



def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, noise, betas, patch_idx, start, step, wind):
    a = (1-betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
    x = x0[:, start:, :, :] * a.sqrt() + noise * (1.0 - a).sqrt()
    
    output, ap_ren = model(torch.cat([x0[:, :start], x], dim=1), t.float(), patch_idx)
    noise_loss = (noise[:,0] - output).square().sum(dim=(0, 1, 2, 3)).mean(dim=0)
    aprec_loss = (x0[:,0]-ap_ren).square().sum(dim=(0, 1, 2, 3)).mean(dim=0)
    loss = noise_loss  + 0.1 * aprec_loss
    wind.add_scalar("NoiseEst",noise_loss.item(),step)
    wind.add_scalar("aprec_loss",aprec_loss.item(),step)

    return loss

class DenoisingDiffusion(object):
    def __init__(self, args, config):                                                        
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DiffusionMA(config)
        self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)
        if os.path.exists(config.logger.logdir):
            pass
        else:
            os.makedirs(config.logger.logdir)
        
        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

    def load_ddm_ckpt(self, load_path, ema=True):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, train_loader):
        cudnn.benchmark = True
        # train_loader, val_loader = DATASET.get_loaders()
        wind =SummaryWriter(os.path.join(self.config.logger.logdir))
        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            
            data_start = time.time()  
            
            data_time = 0
            for i, (x, y, ploc) in enumerate(train_loader):
                
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 6 else x
                n = x.size(0)
                ploc = (ploc/self.config.data.patch_size).reshape(-1,2)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                ploc = ploc.to(self.device)
                x = data_transform(x)
                noise = torch.randn_like(x[:, self.config.data.time_step:, :, :])
                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = noise_estimation_loss(self.model, x, t, noise, self.betas, ploc,self.config.data.time_step, self.step, wind)

                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}, data time: {data_time / (i+1)}")
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
               

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1 :
                    ckpt = {
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }
                    ckpt_root = os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset)
                    # 최신(덮어씀) — resume/기본 eval 호환
                    utils.logging.save_checkpoint(ckpt, filename=ckpt_root)
                    # 스텝별(보존) — 스텝×AUC 비교용. 예: marine_10000.pth
                    utils.logging.save_checkpoint(ckpt, filename=f"{ckpt_root}_{self.step}")
                if self.step % self.config.training.image_sample_freq == 0 and self.step > 1 :
                    self.model.eval()
                    with torch.no_grad():
                        # ipdb.set_trace()
                        skip = self.config.sampling.num_diffusion_timesteps // self.args.sampling_timesteps
                        seq = range(0, self.config.sampling.num_diffusion_timesteps, skip)
                        x_cond = x[:, :self.config.data.time_step][:16]
                        denoise_img, ap_ren = generalized_steps(noise[:16],x_cond,seq,self.model,self.betas, ploc[:16])
                        denoise_img = torch.clamp((denoise_img + 1.0) / 2.0, 0.0, 1.0)[:,0]
                        denoise_img = make_grid(denoise_img)
                        ap_ren = make_grid(torch.clamp((ap_ren + 1.0) / 2.0, 0.0, 1.0))
                        wind.add_image("denoise_image",denoise_img, self.step)
                        wind.add_image("ap_ren",ap_ren, self.step)
                    self.model.train()
        
        utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_final'))
                    
    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None, merge=None):
        
       
        skip = self.config.sampling.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.sampling.num_diffusion_timesteps, skip)
        if merge=="False":
            xs = utils.sampling.generalized_steps_womerge(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size)
        
        if last:
            closs = xs[1]
            xs = xs[0]
        return xs,closs
    
