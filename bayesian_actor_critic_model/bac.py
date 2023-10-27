import os

import gpytorch
import gpytorch.constraints as constraints
import numpy as np
import torch

from utils import fast_svd, set_flat_grad_to


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10, device=torch.device("cpu"), avp_args={}):
    # This method computes A^{-1}*v using repeated A*v computations, where A is a 2D matrix and v is a 1D vector.
    x = torch.zeros(b.size()).to(device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p, **avp_args)
        beta = rdotr / torch.dot(p, _Avp)
        x += beta * p
        r -= beta * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


class BAC:

    def __init__(self, state_dim, action_dim, actor, critic, discount, tau, advantage_flag, actor_args={},
                 critic_args={},
                 actor_lr=3e-3, critic_lr=2e-2, likelihood_noise_level=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = actor(state_dim, action_dim, **actor_args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=constraints.GreaterThan(likelihood_noise_level)).to(self.device)
        self.critic = critic(state_dim, action_dim, self.likelihood, **critic_args).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        # Used for computing GP critic's loss
        self.gp_mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.critic)
        GP_params = [
            param for name, param in self.critic.named_parameters()
            if 'value_head' not in name
        ]
        # Used for optimizing GP critic
        self.critic_gp_optimizer = torch.optim.Adam(GP_params, lr=critic_lr, weight_decay=1e-3)
        self.critic.train()
        self.likelihood.train()

        self.discount = discount
        self.tau = tau
        self.advantage_flag = advantage_flag

    def select_action(self, state, action_mask=None, train=True):
        state = torch.tensor(state.reshape(1, *state.shape)).to(self.device)
        if action_mask is None:
            action_mask = np.array([True])
        action_mask = torch.tensor(action_mask.reshape(1, -1)).to(self.device)
        action = self.actor.select_action(state, action_mask, train)
        return action.cpu().data.numpy().flatten()[0]

    def update(self, replay_memory, svd_low_rank, state_coefficient, fisher_coefficient):
        states, actions, action_masks, next_states, rewards, masks = replay_memory.sample()
        returns = torch.Tensor(actions.size(0), 1).to(self.device)
        prev_return = 0

        if self.advantage_flag:
            state_values_estimates = self.critic.nn_forward(states)
            deltas = torch.Tensor(actions.size(0), 1).to(self.device)
            advantages = torch.Tensor(actions.size(0), 1).to(self.device)
            prev_value = 0
            prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.discount * prev_return * masks[i]
            prev_return = returns[i, 0]
            if self.advantage_flag:
                deltas[i] = rewards[i] + self.discount * prev_value * masks[i] - state_values_estimates.data[i]
                # GAE uses 'gamma' to trade estimation bias for lower variance.
                advantages[i] = deltas[i] + self.discount * self.tau * prev_advantage * masks[i]
                # For unbiased advantage estimates, replace GAE with this
                # advantages[i] = returns[i] - state_values_estimates[i]
                prev_value = state_values_estimates.data[i, 0]
                prev_advantage = advantages[i, 0]
        if self.advantage_flag:
            self.critic_optimizer.zero_grad()
            state_value_loss = (state_values_estimates - returns).pow(2).mean()
            state_value_loss.backward()
            self.critic_optimizer.step()
            advantages = (advantages - advantages.mean()) / advantages.std()
        targets = advantages if self.advantage_flag else returns
        # Optimizing the parameters of action value function Q(s,a), i.e., neural net feature_extractor + GP parameters
        # U matrix from the paper is simply the gradient of U_prob w.r.t policy parameters.
        U_prob = self.actor.get_log_prob(states, actions, action_masks)
        # A_{m by n} = U_{m by m} S_{m by n} V_{n by n}
        u_fb_t, s_fb_t, v_fb_t = fast_svd.pca_U(svd_low_rank, U_prob.squeeze(-1), self.actor, self.device)
        v_tens = v_fb_t.transpose(1, 0)
        GP_inputs = torch.cat([states, v_tens], 1)
        # Optimize the critic
        self.critic.set_train_data(GP_inputs, targets.squeeze(-1), strict=False)
        with gpytorch.settings.max_cg_iterations(1000), gpytorch.settings.max_preconditioner_size(50):
            self.critic_gp_optimizer.zero_grad()
            fisher_multiplier = fisher_coefficient * v_tens.shape[0]
            action_values = self.critic(GP_inputs,
                                        state_multiplier=state_coefficient,
                                        fisher_multiplier=fisher_multiplier)
            action_value_loss = -self.gp_mll(action_values, targets.squeeze(-1)).mean()
            action_value_loss.backward()
            self.critic_gp_optimizer.step()
        with gpytorch.settings.max_cg_iterations(1000), gpytorch.settings.max_preconditioner_size(
                50), gpytorch.settings.fast_pred_var():
            action_value_multivariate_normal = self.likelihood(self.critic(GP_inputs,
                                                                           state_multiplier=state_coefficient,
                                                                           fisher_multiplier=fisher_multiplier))
            # action_value_means = action_value_multivariate_normal.mean.unsqueeze(-1) # Q(s,a) predictions
            alpha = action_value_multivariate_normal.lazy_covariance_matrix.inv_matmul(targets.squeeze(-1)).unsqueeze(-1)

        # Optimize the actor
        log_prob = self.actor.get_log_prob(states, actions, action_masks)
        action_value_proxy = alpha.detach()
        actor_loss = (-action_value_proxy * log_prob).mean()

        grads = torch.autograd.grad(actor_loss, self.actor.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        neg_stepdir = conjugate_gradients(self.actor.Fvp_fim, loss_grad, 50, device=self.device,
                                          avp_args={'states': states, 'action_masks': action_masks})
        #print(neg_stepdir)
        self.actor_optimizer.zero_grad()
        set_flat_grad_to(self.actor, neg_stepdir)
        self.actor_optimizer.step()

    def save(self, filename='weights/'):
        if not os.path.exists(filename):
            os.makedirs(filename)
            os.path.join(filename, "bac_critic")
        torch.save(self.critic.state_dict(), os.path.join(filename, "bac_critic"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(filename, "bac_critic_optimizer"))

        torch.save(self.actor.state_dict(), os.path.join(filename, "bac_actor"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(filename, "bac_actor_optimizer"))

    def load(self, filename='weights/'):
        self.critic.load_state_dict(torch.load(os.path.join(filename, "bac_critic")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(filename, "bac_critic_optimizer")))

        self.actor.load_state_dict(torch.load(os.path.join(filename, "bac_actor")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(filename, "bac_actor_optimizer")))

if __name__ == '__main__':
    # Create a sample tensor
    my_tensor = torch.tensor([[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]])

    # Create a boolean mask
    mask = torch.tensor([[True], [False], [True]])

    # Select rows based on the mask
    selected_rows = torch.masked_select(my_tensor, mask).reshape(-1, *my_tensor.shape[1:])

    print(selected_rows)


