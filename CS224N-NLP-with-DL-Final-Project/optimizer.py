from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                ### TODO
                
                #t = len(state)
                #if t == 0:
                   # m_t_minus_1, v_t_minus_1 = 0,0
                #else:
                    #m_t_minus_1, v_t_minus_1 = state[t][0], state[t][1]
                #t += 1

                if len(state) == 0:
                    m_t_minus_1, v_t_minus_1 = 0,0
                    t = 0
                else:
                    t = list(state.keys())[0]
                    m_t_minus_1, v_t_minus_1 = state.pop(t)

                t += 1
                beta1, beta2 = group["betas"]
                m_t = beta1*m_t_minus_1 + (1-beta1)*grad
                v_t = beta2*v_t_minus_1 + (1-beta2)*torch.mul(grad,grad)
                at= alpha * (math.sqrt(1-math.pow(beta2,t)))/(1-math.pow(beta1,t))
                p.data = p.data - at * m_t / (torch.sqrt(v_t)+group['eps'])
                
                state[t] = [m_t, v_t]

                p.data = p.data - alpha*group["weight_decay"]*p.data
                loss = p.data
                
                
                #raise NotImplementedError


        return loss
