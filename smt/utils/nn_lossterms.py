import numpy as np

class LossTerm:
    def __init__(self, x_train, loss_term_weight=1.):
        self.x_train = x_train
        self.loss_term_weight = loss_term_weight

    def setup(self, rbf_surrogate):
        """
        Setup the loss term (e.g., build probes, evaluate RBF bases).
        """
        pass

    def __call__(self, W):
        """
        Calculate the loss given the network weights W.
        """
        raise NotImplementedError("LossTerm subclasses must implement __call__")

    def sample_within_convex_hull(self, n_pts, rng=None):
        """
        Sample n_pts points randomly within the convex hull of self.x_train.
        Uses a Dirichlet distribution to generate weights for a convex combination used to form the points.
        """
        if rng is None:
            rng = np.random.default_rng(1)

        if self.x_train is None:
             raise ValueError("x_train must be set to sample within convex hull.")

        n_samples, n_dim = self.x_train.shape
        
        # For each probe point, select k = n_dim + 1 points from x_train
        # and form a convex combination.
        # This guarantees membership in the convex hull.
        k = n_dim + 1
        
        # Select random indices for each probe point
        # Shape: (n_pts, k)
        indices = rng.integers(0, n_samples, size=(n_pts, k))
        
        # Gather the points
        # Shape: (n_pts, k, n_dim)
        selected_points = self.x_train[indices]
        
        # Generate random weights (Dirichlet distribution essentially)
        # Shape: (n_pts, k)
        weights = rng.exponential(size=(n_pts, k))
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Compute convex combination: sum(w_i * p_i)
        # Shape: (n_pts, n_dim)
        # einsum: 'nk, nkd -> nd'
        points = np.einsum('nk,nkd->nd', weights, selected_points)
        return points


class SliceBasedPriorLossTerm(LossTerm):
    def __init__(self, x_train, y_train, num_parameters, parameter_space_bounds, base_model, constraint_upper_limit, num_eval_pts=5, loss_term_weight=1.):
        super().__init__(x_train, loss_term_weight)
        self.y_train = y_train
        self.num_parameters = num_parameters
        self.parameter_space_bounds = parameter_space_bounds
        self.base_model = base_model
        self.constraint_upper_limit = constraint_upper_limit
        self.num_eval_pts = num_eval_pts
        
        self.prior_list = None
        self.prior_rbf_evals = None
        self.rbf_d0 = None

    def setup(self, rbf_surrogate):
        self.prior_rbf_evals = rbf_surrogate.rbf_centers
        self.rbf_d0 = rbf_surrogate.d0
        self.make_multi_dim_point_priors(self.base_model, self.constraint_upper_limit, num_x_vals=self.num_eval_pts)

    def make_multi_dim_point_priors(self, base_model, constraint_upper_limit, num_x_vals=5, rng=None):
        """
        For each dimension j, build 20 slice points (others fixed to feasible nominal mean),
        evaluate TRUE compliance, perturb ±30% → log targets, return list of (Phi_cond, log_target).
        """

        self.x_vals_per_dim = np.linspace(self.parameter_space_bounds[0], self.parameter_space_bounds[1], num_x_vals)

        if rng is None:
            rng = np.random.default_rng(777)

        # feasible nominal anchor (by scaling mean if needed)
        # Note: using self.x_train requires it was passed in init
        anchor = self.x_train.mean(axis=0).copy()
        base_model_inputs = {base_model.input_key: anchor}

        vol_a = base_model.compute_constraint_value(base_model_inputs, None)
        if vol_a > constraint_upper_limit:
            anchor *= (constraint_upper_limit / vol_a)

        prior_list = []
        for j in range(self.num_parameters):
            for xj in self.x_vals_per_dim:
                h = anchor.copy()
                h[j] = float(xj)
                base_model_inputs = {base_model.input_key: h}
                model_output = base_model.compute_model_output(base_model_inputs)
                base_model_outputs = {base_model.output_key: model_output}
                true_val = base_model.compute_objective_func(base_model_inputs, base_model_outputs)
                sign = rng.choice([-1.0, 1.0])     # ±30%
                y_prior = max(true_val * (1.0 + 0.30 * sign), 1e-12)
                log_target = y_prior  # np.log(y_prior)
                
                # rbf_features wants (X, C, eps)
                Phi_cond = rbf_features(h.reshape(1, -1), self.prior_rbf_evals, self.d0)  # (1, m)
                
                log_target_torch = torch.tensor(log_target, dtype=torch.float32)
                Phi_cond_torch = torch.tensor(Phi_cond, dtype=torch.float32)
                prior_list.append((Phi_cond_torch, log_target_torch))
        
        self.prior_list = prior_list

    def __call__(self, W):
        loss_prior = torch.tensor(0.0)
        for (Phi_cond, t_log) in self.prior_list:
            f = (W @ Phi_cond.T).squeeze(1)
            f = torch.clamp(f, min=1e-12)
            log_f = f  # torch.log(f)
            loss_prior = loss_prior + (log_f.mean() - t_log).pow(2).sum()
        return loss_prior


class MonotonicityLossTerm(LossTerm):
    def __init__(self, x_train, sign=1, stepsize_frac=0.01, mono_pts_per_input_dim=5, random_base_points=False, inside_convex_hull=False, input_indices=None, loss_term_weight=1.):
        super().__init__(x_train, loss_term_weight)
        self.stepsize_frac = stepsize_frac
        self.mono_pts_per_input_dim = mono_pts_per_input_dim
        self.random_base_points = random_base_points
        self.inside_convex_hull = inside_convex_hull
        self.sign = sign
        self.input_indices = input_indices

        self.x_train_minus = None
        self.x_train_plus = None

        self.rbf_evals_minus = None
        self.rbf_evals_plus = None

    def setup(self, rbf_surrogate):
        self.build_mono_pairs()
        self.eval_rbf_basis_in_mono_pts(rbf_surrogate.rbf_centers, rbf_surrogate.d0)

    def build_mono_pairs(self, rng=None):
        if rng is None:
            rng = np.random.default_rng(1)
        lo = self.x_train.min(axis=0); hi = self.x_train.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        step = self.stepsize_frac * span
        Xm_list, Xp_list = [], []
        
        indices = self.input_indices if self.input_indices is not None else range(self.x_train.shape[1])
        
        for i in indices:
            if self.random_base_points:
                if self.inside_convex_hull:
                     base = self.sample_within_convex_hull(self.mono_pts_per_input_dim, rng=rng)
                else:
                     base = lo + (hi - lo) * rng.random((self.mono_pts_per_input_dim, self.x_train.shape[1]))
            else:
                 idx = rng.integers(0, len(self.x_train), size=self.mono_pts_per_input_dim)
                 base = self.x_train[idx].copy()

            xm = base.copy(); xp = base.copy()
            xm[:, i] = np.clip(xm[:, i] - step[i], lo[i], hi[i])
            xp[:, i] = np.clip(xp[:, i] + step[i], lo[i], hi[i])
            Xm_list.append(xm); Xp_list.append(xp)
        
        self.x_train_minus = np.vstack(Xm_list)
        self.x_train_plus = np.vstack(Xp_list)

    def eval_rbf_basis_in_mono_pts(self, rbf_centers, rbf_d0):
        self.rbf_evals_minus = torch.tensor(rbf_features(self.x_train_minus, rbf_centers, rbf_d0), dtype=torch.float32)
        self.rbf_evals_plus = torch.tensor(rbf_features(self.x_train_plus, rbf_centers, rbf_d0), dtype=torch.float32)

    def __call__(self, W, beta=100):
        fm = W @ self.rbf_evals_minus.T
        fp = W @ self.rbf_evals_plus.T
        loss_mono = torch.nn.functional.softplus( -self.sign * (fp - fm), beta=beta).mean()
        return loss_mono


class PositivityLossTerm(LossTerm):
    def __init__(self, x_train, n_pos_pts=128, loss_term_weight=1., inside_convex_hull=False):
        super().__init__(x_train, loss_term_weight)
        self.n_pos_pts = n_pos_pts
        self.inside_convex_hull = inside_convex_hull
        self.pos_probe_pts = None
        self.rbf_evals = None

    def setup(self, rbf_surrogate):
        self.build_pos_probes_with_rng()
        self.eval_rbf_basis_in_pos_probes(rbf_surrogate.rbf_centers, rbf_surrogate.d0)

    def build_pos_probes_with_rng(self, rng_seed=None):
        if rng_seed is None:
            rng_seed = np.random.default_rng(2)
        
        if self.inside_convex_hull:
            self.pos_probe_pts = self.sample_within_convex_hull(self.n_pos_pts, rng=rng_seed)
        
        else:
            lo = self.x_train.min(axis=0); hi = self.x_train.max(axis=0)
            self.pos_probe_pts = lo + (hi - lo) * rng_seed.random((self.n_pos_pts, self.x_train.shape[1]))

    def eval_rbf_basis_in_pos_probes(self, rbf_centers, rbf_d0):
        if self.pos_probe_pts is None:
            raise ValueError("Positivity probes need to be built before they can be evaluated")
        
        self.rbf_evals = torch.tensor(rbf_features(self.pos_probe_pts, rbf_centers, rbf_d0), dtype=torch.float32)

    def __call__(self, W, beta=100):
        f_pos = W @ self.rbf_evals.T
        loss_pos = torch.nn.functional.softplus(-f_pos, beta=beta).mean()
        return loss_pos
