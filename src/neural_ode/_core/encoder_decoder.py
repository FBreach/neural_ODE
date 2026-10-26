###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import torch
import torch.nn as nn
from . import utils
from torch.nn.modules.rnn import GRU
from .utils import create_mlp, get_device, hidden_dims_list


# GRU description:
# http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim,
        update_gate = None,
        reset_gate = None,
        new_state_net = None,
        hidden_dims = None):
        super(GRU_unit, self).__init__()
        n_inputs = latent_dim * 2 + input_dim

        if update_gate is None:
            self.update_gate = create_mlp(
                n_inputs,
                hidden_dims,
                latent_dim,
                final_activation=nn.Sigmoid,
                name="z0_encoder_odernn.gru_update_hidden_dims",
            )
            utils.init_network_weights(self.update_gate)
        else:
            self.update_gate  = update_gate

        if reset_gate is None:
            self.reset_gate = create_mlp(
                n_inputs,
                hidden_dims,
                latent_dim,
                final_activation=nn.Sigmoid,
                name="z0_encoder_odernn.gru_update_hidden_dims",
            )
            utils.init_network_weights(self.reset_gate)
        else:
            self.reset_gate  = reset_gate

        if new_state_net is None:
            self.new_state_net = create_mlp(
                n_inputs,
                hidden_dims,
                latent_dim * 2,
                name="z0_encoder_odernn.gru_update_hidden_dims",
            )
            utils.init_network_weights(self.new_state_net)
        else:
            self.new_state_net  = new_state_net


    def forward(self, y_mean, y_std, x, masked_update = True):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1-update_gate) * new_state + update_gate * y_mean
        new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

        assert(not torch.isnan(new_y).any())

        if masked_update:
            # IMPORTANT: assumes that x contains both data and mask
            # update only the hidden states for hidden state only if at least one feature is present for the current time point
            n_data_dims = x.size(-1)//2
            mask = x[:, :, n_data_dims:]
            utils.check_mask(x[:, :, :n_data_dims], mask)

            mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

            assert(not torch.isnan(mask).any())

            new_y = mask * new_y + (1-mask) * y_mean
            new_y_std = mask * new_y_std + (1-mask) * y_std

            if torch.isnan(new_y).any():
                print("new_y is nan!")
                print(mask)
                print(y_mean)
                exit()

        new_y_std = new_y_std.abs()
        return new_y, new_y_std



class Encoder_z0_RNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_dim,
        hidden_dims,
        posterior_hidden_dims,
        use_delta_t = True,
    ):

        super(Encoder_z0_RNN, self).__init__()

        rnn_hidden_dims = hidden_dims_list(
            hidden_dims,
            "z0_encoder_rnn.hidden_dims",
            allow_empty=False,
        )
        if len(set(rnn_hidden_dims)) != 1:
            raise ValueError(
                "z0_encoder_rnn.hidden_dims must use one repeated dimension "
                f"for stacked GRU layers, got {rnn_hidden_dims}."
            )

        self.gru_rnn_output_size = rnn_hidden_dims[-1]
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.use_delta_t = use_delta_t

        self.hiddens_to_z0 = create_mlp(
            self.gru_rnn_output_size,
            posterior_hidden_dims,
            latent_dim * 2,
            nonlinear=nn.LeakyReLU,
            name="z0_encoder_rnn.posterior_hidden_dims",
        )

        utils.init_network_weights(self.hiddens_to_z0)

        if use_delta_t:
            self.input_dim += 1
        self.gru_rnn = GRU(
            self.input_dim,
            self.gru_rnn_output_size,
            num_layers=len(rnn_hidden_dims),
        )

    def forward(self, data, time_steps, run_backwards = True):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        # data shape: [n_traj, n_tp, n_dims]
        # shape required for rnn: (seq_len, batch, input_size)
        # t0: not used here
        n_traj = data.size(0)

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        data = data.permute(1,0,2)

        if run_backwards:
            # Look at data in the reverse order: from later points to the first
            data = utils.reverse(data)

        if self.use_delta_t:
            if time_steps.ndim == 1:
                delta_t = time_steps[1:] - time_steps[:-1]
                if run_backwards:
                    # we are going backwards in time with
                    delta_t = utils.reverse(delta_t)
                # append zero delta t in the end
                delta_t = torch.cat((delta_t, torch.zeros(1).to(data.device)))
                delta_t = delta_t.unsqueeze(1).repeat((1,n_traj)).unsqueeze(-1)
            else:
                delta_t = time_steps[:, 1:] - time_steps[:, :-1]
                if run_backwards:
                    # we are going backwards in time with
                    idx = [i for i in range(delta_t.size(1)-1, -1, -1)]
                    delta_t = delta_t[:, idx]
                delta_t = torch.cat((delta_t, torch.zeros((delta_t.shape[0], 1, delta_t.shape[-1])).to(data.device)), dim=1)
                delta_t = delta_t.transpose(0, 1)
            data = torch.cat((delta_t, data),-1)

        outputs, _ = self.gru_rnn(data)

        # LSTM output shape: (seq_len, batch, num_directions * hidden_size)
        last_output = outputs[-1]

        self.extra_info ={"rnn_outputs": outputs, "time_points": time_steps}

        mean, std = utils.split_last_dim(self.hiddens_to_z0(last_output))
        std = std.abs()

        assert(not torch.isnan(mean).any())
        assert(not torch.isnan(std).any())

        return mean.unsqueeze(0), std.unsqueeze(0)


class Encoder_z0_ODE_RNN(nn.Module):
    # Derive z0 by running ode backwards.
    # For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
    # Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
    # Continue until we get to z0
    def __init__(self, latent_dim, input_dim, z0_diffeq_solver = None,
        z0_dim = None, GRU_update = None,
        gru_update_hidden_dims = None,
        posterior_hidden_dims = None):

        super(Encoder_z0_ODE_RNN, self).__init__()

        if z0_dim is None:
            self.z0_dim = latent_dim
        else:
            self.z0_dim = z0_dim

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim, input_dim,
                hidden_dims=gru_update_hidden_dims)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.extra_info = None

        self.transform_z0 = create_mlp(
            latent_dim * 2,
            posterior_hidden_dims,
            self.z0_dim * 2,
            name="z0_encoder_odernn.posterior_hidden_dims",
        )
        utils.init_network_weights(self.transform_z0)


    def forward(self, data: torch.Tensor, time_steps, run_backwards = True, save_info = False):
        # data, time_steps -- observations and their time stamps
        # IMPORTANT: assumes that 'data' already has mask concatenated to it
        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        n_traj, n_tp, n_dims = data.size()
        if n_tp == 1:
            prev_y = torch.zeros((1, n_traj, self.latent_dim), device=data.device)
            prev_std = torch.zeros((1, n_traj, self.latent_dim), device=data.device)

            xi = data[:,0,:].unsqueeze(0)

            last_yi, last_yi_std = self.GRU_update(prev_y, prev_std, xi)
            extra_info = None
        else:

            last_yi, last_yi_std, _, extra_info = self.run_odernn(
                data, time_steps, run_backwards = run_backwards,
                save_info = save_info)

        means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
        std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

        mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
        std_z0 = std_z0.abs()
        if save_info:
            self.extra_info = extra_info

        return mean_z0, std_z0


    def run_odernn(self, data, time_steps,
        run_backwards = True, save_info = False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it.
        n_traj, n_tp, n_dims = data.size()
        device = get_device(data)
        extra_info = []

        assert(not torch.isnan(data).any())
        assert(not torch.isnan(time_steps).any())

        prev_y = torch.zeros((1, n_traj, self.latent_dim), device=device)
        prev_std = torch.zeros((1, n_traj, self.latent_dim), device=device)
        prev_t = torch.zeros((n_traj, 1), device=device)
        has_prev = torch.zeros((n_traj, 1), dtype=torch.bool, device=device)

        n_data_dims = n_dims // 2
        if n_dims == 2 * n_data_dims and n_data_dims > 0:
            valid_obs = data[:, :, n_data_dims:].sum(-1, keepdim=True) > 0
        else:
            valid_obs = torch.ones((n_traj, n_tp, 1), dtype=torch.bool, device=device)

        def time_at(index):
            if time_steps.ndim == 1:
                return time_steps[index].reshape(1, 1).repeat(n_traj, 1)
            if time_steps.ndim == 2:
                return time_steps[:, index].reshape(n_traj, 1)
            return time_steps[:, index].reshape(n_traj, -1)

        def solver_time_pair(start_t, end_t):
            if self.z0_diffeq_solver.__class__.__name__ == "DiffeqSolverParallel":
                return torch.stack((start_t, end_t), dim=1)
            if not torch.allclose(start_t, start_t[:1]) or not torch.allclose(end_t, end_t[:1]):
                raise ValueError("Non-parallel ODE-RNN encoder cannot handle per-sample time grids.")
            return torch.stack((start_t[0, 0], end_t[0, 0]))

        time_points_iter = range(n_tp - 1, -1, -1) if run_backwards else range(n_tp)
        latent_ys = []
        yi = prev_y
        yi_std = prev_std

        for i in time_points_iter:
            curr_t = time_at(i)
            valid_i = valid_obs[:, i]
            start_t = torch.where(has_prev, prev_t, curr_t)
            should_integrate = has_prev & valid_i & ((curr_t - start_t).abs() > 0)

            if should_integrate.any():
                time_points = solver_time_pair(start_t, curr_t)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)
                assert(not torch.isnan(ode_sol).any())
                yi_integrated = ode_sol[:, :, -1, :]
                yi_ode = torch.where(should_integrate.unsqueeze(0), yi_integrated, prev_y)
            else:
                time_points = solver_time_pair(curr_t, curr_t)
                ode_sol = torch.stack((prev_y, prev_y), 2).to(device)
                yi_ode = prev_y

            if torch.mean(torch.abs(ode_sol[:, :, 0, :] - prev_y)) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(torch.abs(ode_sol[:, :, 0, :] - prev_y)))
                exit()

            xi = data[:, i, :].unsqueeze(0)
            yi, yi_std = self.GRU_update(yi_ode, prev_std, xi)

            valid_i_b = valid_i.unsqueeze(0)
            prev_y = torch.where(valid_i_b, yi, prev_y)
            prev_std = torch.where(valid_i_b, yi_std, prev_std)
            prev_t = torch.where(valid_i, curr_t, prev_t)
            has_prev = has_prev | valid_i
            latent_ys.append(prev_y)

            if save_info:
                d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
                     "yi": prev_y.detach(), "yi_std": prev_std.detach(),
                     "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
                extra_info.append(d)

        latent_ys = torch.stack(latent_ys, 1)

        assert(not torch.isnan(prev_y).any())
        assert(not torch.isnan(prev_std).any())

        return prev_y, prev_std, latent_ys, extra_info



class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space

        decoder = nn.Sequential(
           nn.Linear(latent_dim, input_dim),)

        utils.init_network_weights(decoder)
        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)


