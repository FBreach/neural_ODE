###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import torch
import torch.nn as nn

# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint

#####################################################################################################

class DiffeqSolver(nn.Module):
    def __init__(self, input_dim, ode_func, method, latents, 
            odeint_rtol = 1e-4, odeint_atol = 1e-5):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.latents = latents		
        self.ode_func = ode_func

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps_to_predict, backwards = False):
        """
        # Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
        #n_dims = first_point.size()[-1]

        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        pred_y = pred_y.permute(1,2,0,3)

        assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)

        return pred_y

    def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, 
        n_traj_samples = 1):
        """
        # Decode the trajectory through ODE Solver using samples from the prior

        time_steps_to_predict: time steps at which we want to sample the new trajectory
        """
        func = self.ode_func.sample_next_point_from_prior

        pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
            rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
        # shape: [n_traj_samples, n_traj, n_tp, n_dim]
        pred_y = pred_y.permute(1,2,0,3)
        return pred_y


class DiffeqSolverParallel(DiffeqSolver):
    def __init__(self, input_dim, ode_func, method, latents, odeint_rtol=0.0001, odeint_atol=0.00001):
        super().__init__(input_dim, ode_func, method, latents, odeint_rtol, odeint_atol)
        
        if method == 'euler':
            self.ode_method = self.euler
        elif method == 'rk2':
            self.ode_method = self.rk2
        elif method == 'rk4':
            self.ode_method = self.rk4
        else:
            raise NotImplementedError(f'Method {method} is not implemented!')
        
    def euler(self, first_point, time_steps_to_predict):
        # first_point: N, B, F
        # time_steps_to_predict: B, T, F
        out = [first_point]
        t = time_steps_to_predict.shape[1]
        for i in range(1, t):
            delta_t = time_steps_to_predict[:, i] - time_steps_to_predict[:, i-1]
            next_state = out[i-1] + self.ode_func(time_steps_to_predict[:, i-1], out[i-1]) * delta_t            
            out.append(next_state)
        
        # [n_traj_samples, n_samples, n_timepoints, n_latents]
        return torch.stack(out, -2)
    
    def rk2(self, first_point, time_steps_to_predict):
        out = [first_point]
        t = time_steps_to_predict.shape[1]

        for i in range(1, t):
            t_prev = time_steps_to_predict[:, i - 1]
            t_curr = time_steps_to_predict[:, i]
            # could be [B] or [B, F]
            delta_t = t_curr - t_prev   

            y_prev = out[i - 1]

            k1 = self.ode_func(t_prev, y_prev)
            k2 = self.ode_func(t_prev + delta_t / 2, y_prev + k1 * delta_t / 2)

            next_state = y_prev + delta_t * k2
            out.append(next_state)

        return torch.stack(out, -2)

    def rk4(self, first_point, time_steps_to_predict):
        out = [first_point]
        t = time_steps_to_predict.shape[1]

        for i in range(1, t):
            t_prev = time_steps_to_predict[:, i - 1]
            t_curr = time_steps_to_predict[:, i]
            # could be [B] or [B, F]
            delta_t = t_curr - t_prev   

            y_prev = out[i - 1]

            k1 = self.ode_func(t_prev, y_prev)
            k2 = self.ode_func(t_prev + delta_t / 2, y_prev + k1 * delta_t / 2)
            k3 = self.ode_func(t_prev + delta_t / 2, y_prev + k2 * delta_t / 2)
            k4 = self.ode_func(t_prev + delta_t, y_prev + k3 * delta_t)

            next_state = y_prev + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(next_state)

        return torch.stack(out, -2)
        
    def forward(self, first_point, time_steps_to_predict, backwards = False):
        """
        # Decode the trajectory through ODE Solver
        """
        n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]

        pred_y = self.ode_method(first_point, time_steps_to_predict)

        assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
        assert(pred_y.size()[0] == n_traj_samples)
        assert(pred_y.size()[1] == n_traj)
        return pred_y
            
  



