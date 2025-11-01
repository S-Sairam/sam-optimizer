import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05,**kwargs):
        self.rho = rho
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device 
        norm_sum_sq = torch.tensor(0.0, device=shared_device)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad
                p_norm = grad.norm(p=2)
                p_norm_sq = p_norm.pow(2)
                norm_sum_sq.add_(p_norm_sq)
        final_norm = norm_sum_sq.sqrt()
        return final_norm

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue 
                e_w = p.grad * scale
                self.state[p]["e_w"] = e_w
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" not in self.state[p]: continue  
                e_w = self.state[p]["e_w"]
                p.sub_(e_w)
        self.base_optimizer.step()

        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        pass




    """
    Dummy step function to be compatible with PyTorch LRSchedulers.
    The real work is done in first_step and second_step.
    """
    # The scheduler will call this to increment its internal counter.
    # We don't need to do anything here because the base_optimizer.step()
    # is called inside second_step().
