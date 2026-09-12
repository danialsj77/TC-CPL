"""Constrained Policy Optimization (Achiam et al., ICML 2017) for the PST problem.

WHY THIS EXISTS
    CPO is the canonical trust-region algorithm for CMDPs and the natural
    question a reviewer asks of any Lagrangian method: would a principled
    constrained-optimization approach have done as well without M1/M2? This is a
    BASELINE, not an ablation -- it replaces TC-CPL's whole constraint stack.

WHAT IT SHARES WITH TC-CPL, AND WHY
    Same env, same grid-aware observation, same scale-free reward. It does NOT
    get the M3 encoder: the structured encoder is one of the paper's
    contributions and is withheld from every baseline (the notebook registers a
    flat-MLP extractor for CPO in Sec. 6.1). What CPO also cannot use is M1 (it
    replaces the dual variable with a trust region) and M2 (there is no replay
    buffer to warm-start -- CPO is strictly on-policy).

ONE CONSTRAINT, DELIBERATELY
    Achiam's closed-form dual, the feasibility analysis and the recovery step are
    all derived for a SINGLE constraint. Problem 1 has two. Rather than invent a
    multi-constraint variant and attribute its behaviour to CPO, this constrains
    the transformer overload -- the safety constraint the paper is about -- and
    leaves user satisfaction in the reward exactly as the PST baselines do. That
    is CPO as published, and it makes TC-CPL's native handling of an arbitrary
    number of constraints a real and statable advantage rather than a claim.

THE UPDATE
    maximise   g'x        subject to   c + b'x <= 0,   Â½ x'Hx <= delta
    where g is the reward policy gradient, b the cost policy gradient, c the
    current constraint violation and H the Fisher information matrix. With
    q = g'H^-1 g,  r = g'H^-1 b,  s = b'H^-1 b, the dual reduces to a
    one-dimensional problem in lambda, solved here by the case analysis of
    Achiam Appendix 10.2, followed by a backtracking line search on the true
    (un-approximated) surrogate and constraint.
"""
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def flat_params(module: nn.Module) -> th.Tensor:
    return th.cat([p.data.view(-1) for p in module.parameters()])


def set_flat_params(module: nn.Module, flat: th.Tensor) -> None:
    i = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def flat_grad(loss: th.Tensor, module: nn.Module,
              retain: bool = False, create: bool = False) -> th.Tensor:
    grads = th.autograd.grad(loss, list(module.parameters()),
                             retain_graph=retain, create_graph=create,
                             allow_unused=True)
    out = []
    for g, p in zip(grads, module.parameters()):
        out.append(th.zeros_like(p).view(-1) if g is None else g.reshape(-1))
    return th.cat(out)


def conjugate_gradient(mvp: Callable[[th.Tensor], th.Tensor], b: th.Tensor,
                       iters: int = 10, tol: float = 1e-10) -> th.Tensor:
    """Solve H x = b without ever forming H."""
    x = th.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = th.dot(r, r)
    for _ in range(iters):
        Hp = mvp(p)
        denom = th.dot(p, Hp)
        if denom <= 1e-12:
            break
        alpha = rr / denom
        x += alpha * p
        r -= alpha * Hp
        rr_new = th.dot(r, r)
        if rr_new < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(x)
    run = 0.0
    for t in reversed(range(len(x))):
        run = x[t] + gamma * run
        out[t] = run
    return out


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------
class GaussianActor(nn.Module):
    """Diagonal-Gaussian policy on top of a features extractor (M3 or flat)."""

    def __init__(self, extractor: nn.Module, features_dim: int, act_dim: int,
                 hidden: Tuple[int, ...] = (256, 256), log_std_init: float = -0.5):
        super().__init__()
        self.extractor = extractor
        layers: List[nn.Module] = []
        last = features_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.SiLU()]
            last = h
        layers += [nn.Linear(last, act_dim)]
        self.mu = nn.Sequential(*layers)
        self.log_std = nn.Parameter(th.ones(act_dim) * log_std_init)

    def distribution(self, obs: th.Tensor) -> th.distributions.Normal:
        mean = self.mu(self.extractor(obs))
        return th.distributions.Normal(mean, self.log_std.exp())

    def log_prob(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        return self.distribution(obs).log_prob(act).sum(-1)

    @th.no_grad()
    def act(self, obs: th.Tensor, deterministic: bool = False):
        dist = self.distribution(obs)
        a = dist.mean if deterministic else dist.sample()
        return a, dist.log_prob(a).sum(-1)


class Critic(nn.Module):
    def __init__(self, extractor: nn.Module, features_dim: int,
                 hidden: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.extractor = extractor
        layers: List[nn.Module] = []
        last = features_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.SiLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.v = nn.Sequential(*layers)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.v(self.extractor(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# CPO
# ---------------------------------------------------------------------------
class CPO:
    """CPO with one cost constraint, on a SB3-style VecEnv.

    cost_fn(info) -> float pulls the per-step constraint cost out of the env's
    info dict. For TC-CPL's env that is info["g1"], the margin-shaped overload.
    """

    def __init__(
        self,
        venv,
        make_extractor: Callable[[], Tuple[nn.Module, int]],
        cost_limit: float,
        cost_fn: Callable[[dict], float] = lambda i: float(i.get("g1", 0.0)),
        n_steps: int = 1024,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        cost_gamma: float = 0.99,
        delta: float = 0.01,            # KL trust region
        vf_lr: float = 1e-3,
        vf_iters: int = 40,
        cg_iters: int = 10,
        damping: float = 0.1,
        backtrack_coeff: float = 0.8,
        backtrack_iters: int = 10,
        max_batch: int = 1024,
        fisher_batch: int = 2048,
        device: str = "auto",
        verbose: int = 1,
    ):
        self.venv = venv
        self.n_envs = venv.num_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.cost_gamma = cost_gamma
        self.delta = delta
        self.cost_limit = cost_limit
        self.cost_fn = cost_fn
        self.cg_iters = cg_iters
        self.damping = damping
        self.backtrack_coeff = backtrack_coeff
        self.backtrack_iters = backtrack_iters
        self.vf_iters = vf_iters
        self.verbose = verbose
        # A rollout is n_steps x n_envs observations, and the M3 encoder expands
        # each one into n_evses rows. At 32 envs x 256 steps x 600 ports that is
        # 4.9M rows in a single forward -- 1.17 GiB, which OOMs an 8 GB card. So
        # gradients are accumulated over chunks of max_batch (exact), and the
        # Fisher-vector product uses a fixed subsample (standard TRPO practice,
        # since H only needs to be accurate enough to shape the step).
        self.max_batch = max_batch
        self.fisher_batch = fisher_batch

        if device == "auto":
            # like tccpl_learner.resolve_device: cuda, then Apple mps, then cpu
            if th.cuda.is_available():
                device = "cuda"
            elif getattr(th.backends, "mps", None) is not None \
                    and th.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = th.device(device)

        obs_dim = int(np.prod(venv.observation_space.shape))
        act_dim = int(np.prod(venv.action_space.shape))
        self.obs_dim, self.act_dim = obs_dim, act_dim

        ext_pi, fdim = make_extractor()
        ext_v, _ = make_extractor()
        ext_vc, _ = make_extractor()
        self.actor = GaussianActor(ext_pi, fdim, act_dim).to(self.device)
        self.critic = Critic(ext_v, fdim).to(self.device)
        self.cost_critic = Critic(ext_vc, fdim).to(self.device)

        self.vf_opt = th.optim.AdamW(self.critic.parameters(), lr=vf_lr,
                                     weight_decay=1e-4)
        self.cvf_opt = th.optim.AdamW(self.cost_critic.parameters(), lr=vf_lr,
                                      weight_decay=1e-4)

        self.num_timesteps = 0
        self.history: List[dict] = []
        self._obs = venv.reset()
        # running per-episode cost, used to estimate J_C(pi_k)
        self._ep_cost = np.zeros(self.n_envs)
        self._recent_ep_costs: List[float] = []

    # -- rollout -------------------------------------------------------------
    def collect(self) -> Dict[str, th.Tensor]:
        T, E = self.n_steps, self.n_envs
        obs_b = np.zeros((T, E, self.obs_dim), np.float32)
        act_b = np.zeros((T, E, self.act_dim), np.float32)
        logp_b = np.zeros((T, E), np.float32)
        rew_b = np.zeros((T, E), np.float32)
        cost_b = np.zeros((T, E), np.float32)
        done_b = np.zeros((T, E), np.float32)

        for t in range(T):
            obs_t = th.as_tensor(self._obs, dtype=th.float32, device=self.device)
            a, logp = self.actor.act(obs_t)
            a_np = a.cpu().numpy()
            lo, hi = self.venv.action_space.low, self.venv.action_space.high
            nxt, rew, dones, infos = self.venv.step(np.clip(a_np, lo, hi))

            obs_b[t], act_b[t], logp_b[t] = self._obs, a_np, logp.cpu().numpy()
            rew_b[t], done_b[t] = rew, dones
            c = np.array([self.cost_fn(i) for i in infos], np.float32)
            cost_b[t] = c
            self._ep_cost += c
            for e, d in enumerate(dones):
                if d:
                    self._recent_ep_costs.append(float(self._ep_cost[e]))
                    self._ep_cost[e] = 0.0
            self._obs = nxt
            self.num_timesteps += E

        with th.no_grad():
            flat_obs = th.as_tensor(obs_b.reshape(T * E, -1), device=self.device)
            vals = self.critic(flat_obs).cpu().numpy().reshape(T, E)
            cvals = self.cost_critic(flat_obs).cpu().numpy().reshape(T, E)
            last = th.as_tensor(self._obs, dtype=th.float32, device=self.device)
            last_v = self.critic(last).cpu().numpy()
            last_cv = self.cost_critic(last).cpu().numpy()

        adv = np.zeros((T, E), np.float32)
        cadv = np.zeros((T, E), np.float32)
        for e in range(E):
            v = np.append(vals[:, e], last_v[e])
            cv = np.append(cvals[:, e], last_cv[e])
            nonterm = 1.0 - done_b[:, e]
            gae = cgae = 0.0
            for t in reversed(range(T)):
                d = rew_b[t, e] + self.gamma * v[t + 1] * nonterm[t] - v[t]
                gae = d + self.gamma * self.gae_lambda * nonterm[t] * gae
                adv[t, e] = gae
                dc = cost_b[t, e] + self.cost_gamma * cv[t + 1] * nonterm[t] - cv[t]
                cgae = dc + self.cost_gamma * self.gae_lambda * nonterm[t] * cgae
                cadv[t, e] = cgae
        ret = adv + vals
        cret = cadv + cvals

        # kept on the CPU: chunks are moved to the device as they are consumed,
        # so peak memory is set by max_batch rather than by the rollout size
        f = lambda x, d=None: th.as_tensor(
            x.reshape(T * E, -1) if d else x.reshape(T * E))
        return dict(
            obs=f(obs_b, 1), act=f(act_b, 1), logp=f(logp_b),
            adv=f(adv), cadv=f(cadv), ret=f(ret), cret=f(cret),
        )

    # -- CPO step ------------------------------------------------------------
    def update(self, data: Dict[str, th.Tensor]) -> dict:
        obs, act, logp_old = data["obs"], data["act"], data["logp"]
        adv = (data["adv"] - data["adv"].mean()) / (data["adv"].std() + 1e-8)
        cadv = (data["cadv"] - data["cadv"].mean()) / (data["cadv"].std() + 1e-8)

        # J_C(pi_k) - d, measured on the TRUE per-episode cost
        jc = float(np.mean(self._recent_ep_costs)) if self._recent_ep_costs \
            else float(data["cret"].mean().item())
        c = jc - self.cost_limit
        self._recent_ep_costs.clear()

        N = obs.shape[0]
        dev = self.device
        chunks = [slice(i, min(i + self.max_batch, N))
                  for i in range(0, N, self.max_batch)]

        def surrogates():
            """Mean surrogate reward and cost, accumulated over chunks (no grad)."""
            r_tot = c_tot = 0.0
            with th.no_grad():
                for sl in chunks:
                    o, a = obs[sl].to(dev), act[sl].to(dev)
                    ratio = th.exp(self.actor.log_prob(o, a) - logp_old[sl].to(dev))
                    r_tot += (ratio * adv[sl].to(dev)).sum().item()
                    c_tot += (ratio * cadv[sl].to(dev)).sum().item()
            return r_tot / N, c_tot / N

        def surrogate_grads():
            """Exact g and b: the mean is a sum over chunks, so are its gradients."""
            g_acc = b_acc = None
            for sl in chunks:
                o, a = obs[sl].to(dev), act[sl].to(dev)
                ratio = th.exp(self.actor.log_prob(o, a) - logp_old[sl].to(dev))
                lr = (ratio * adv[sl].to(dev)).sum() / N
                lc = (ratio * cadv[sl].to(dev)).sum() / N
                gi = flat_grad(lr, self.actor, retain=True)
                bi = flat_grad(lc, self.actor)
                g_acc = gi if g_acc is None else g_acc + gi
                b_acc = bi if b_acc is None else b_acc + bi
            return g_acc, b_acc

        pi_loss_val, _ = surrogates()
        g, b = surrogate_grads()

        # Fisher-vector product on a fixed subsample: H only has to shape the
        # step, and a full-rollout Hessian-vector product does not fit in memory.
        m = min(self.fisher_batch, N)
        sub = th.randperm(N)[:m]
        obs_f = obs[sub].to(dev)
        with th.no_grad():
            d_old = self.actor.distribution(obs_f)
            mu_old, std_old = d_old.mean.detach(), d_old.stddev.detach()

        def Hx(x: th.Tensor) -> th.Tensor:
            dist = self.actor.distribution(obs_f)
            kl = th.distributions.kl_divergence(
                th.distributions.Normal(mu_old, std_old), dist).sum(-1).mean()
            gkl = flat_grad(kl, self.actor, retain=True, create=True)
            return flat_grad((gkl * x).sum(), self.actor, retain=True) \
                + self.damping * x

        Hinv_g = conjugate_gradient(Hx, g, self.cg_iters)
        q = th.dot(g, Hinv_g).item()
        if q <= 0:
            return {"status": "bad_curvature", "c": c}

        b_norm = th.dot(b, b).item()
        theta_old = flat_params(self.actor).clone()

        if b_norm < 1e-12:                       # no cost gradient -> plain TRPO
            step = th.sqrt(th.tensor(2 * self.delta / q)) * Hinv_g
            status = "trpo"
        else:
            Hinv_b = conjugate_gradient(Hx, b, self.cg_iters)
            r = th.dot(g, Hinv_b).item()
            s = th.dot(b, Hinv_b).item()
            s = max(s, 1e-12)

            if c > 0 and c ** 2 / s - 2 * self.delta > 0:
                # infeasible and the trust region cannot reach feasibility:
                # pure recovery, move only to reduce the constraint
                step = -th.sqrt(th.tensor(2 * self.delta / s)) * Hinv_b
                status = "recovery"
            else:
                # dual: minimise (q - r^2/s)/(2 lam) + lam(delta - c^2/(2s)) - cr/s
                A = max(q - r ** 2 / s, 1e-12)
                B = 2 * self.delta - c ** 2 / s
                if B <= 0:
                    lam = np.sqrt(q / (2 * self.delta))
                    nu = 0.0
                else:
                    lam = np.sqrt(A / B)
                    nu = max(0.0, (lam * c + r) / s)
                lam = max(lam, 1e-8)
                step = (Hinv_g - nu * Hinv_b) / lam
                status = "cpo"

        # backtracking line search on the TRUE surrogate + constraint + KL
        accepted, kl_val = False, 0.0
        for j in range(self.backtrack_iters):
            frac = self.backtrack_coeff ** j
            set_flat_params(self.actor, theta_old + frac * step)
            new_pi, new_cost = surrogates()
            with th.no_grad():
                dist = self.actor.distribution(obs_f)
                kl_val = th.distributions.kl_divergence(
                    th.distributions.Normal(mu_old, std_old),
                    dist).sum(-1).mean().item()
            improved = new_pi > pi_loss_val or status == "recovery"
            cost_ok = (c + new_cost <= 0) or (new_cost <= 0) or status == "recovery"
            if kl_val <= self.delta and improved and cost_ok:
                accepted = True
                break
        if not accepted:
            set_flat_params(self.actor, theta_old)
            kl_val = 0.0

        # value functions, also chunked
        ret, cret = data["ret"], data["cret"]
        for _ in range(self.vf_iters):
            for sl in chunks:
                o = obs[sl].to(dev)
                self.vf_opt.zero_grad()
                ((self.critic(o) - ret[sl].to(dev)) ** 2).mean().backward()
                self.vf_opt.step()
                self.cvf_opt.zero_grad()
                ((self.cost_critic(o) - cret[sl].to(dev)) ** 2).mean().backward()
                self.cvf_opt.step()

        return {"status": status, "accepted": accepted, "c": c,
                "kl": float(kl_val), "backtracks": j, "Jc": jc}

    # -- driver --------------------------------------------------------------
    def learn(self, total_timesteps: int) -> "CPO":
        while self.num_timesteps < total_timesteps:
            data = self.collect()
            info = self.update(data)
            info["timesteps"] = self.num_timesteps
            self.history.append(info)
            if self.verbose:
                print(f"[CPO {self.num_timesteps:>7}] status={info['status']:<9} "
                      f"accepted={info.get('accepted')} "
                      f"J_C={info.get('Jc', float('nan')):.4f} "
                      f"c={info['c']:+.4f} kl={info.get('kl', 0):.5f}",
                      flush=True)
        return self

    def save(self, path: str) -> None:
        if not path.endswith(".zip"):
            path += ".zip"          # the notebook's evaluator appends .zip too
        th.save({"actor": self.actor.state_dict(),
                 "critic": self.critic.state_dict(),
                 "cost_critic": self.cost_critic.state_dict(),
                 "history": self.history,
                 "config": {"obs_dim": self.obs_dim, "act_dim": self.act_dim,
                            "cost_limit": self.cost_limit,
                            "delta": self.delta}}, path)


class CPOPolicy:
    """SB3-shaped shim (`load` / `predict`) so a CPO checkpoint drops straight
    into the notebook's evaluate_model() alongside SAC, DDPG and PPO.

    The feature extractor lives in the notebook, so register a factory once
    before loading:

        CPOPolicy.set_extractor_factory(make_extractor)
    """

    _extractor_factory = None

    def __init__(self, actor: GaussianActor, device: th.device,
                 low: float = 0.0, high: float = 1.0):
        self.actor, self.device, self.low, self.high = actor, device, low, high

    @classmethod
    def set_extractor_factory(cls, fn: Callable[[], Tuple[nn.Module, int]]) -> None:
        cls._extractor_factory = fn

    @classmethod
    def load(cls, path: str, device: str = "auto", **_) -> "CPOPolicy":
        if cls._extractor_factory is None:
            raise RuntimeError(
                "CPOPolicy.set_extractor_factory(...) must be called before load()")
        if not path.endswith(".zip"):
            path += ".zip"
        ck = th.load(path, map_location="cpu", weights_only=False)
        ext, fdim = cls._extractor_factory()
        actor = GaussianActor(ext, fdim, ck["config"]["act_dim"])
        actor.load_state_dict(ck["actor"])
        dev = th.device("cuda" if (device in ("auto", "cuda")
                                   and th.cuda.is_available()) else "cpu")
        actor.to(dev).eval()
        return cls(actor, dev)

    def predict(self, obs, deterministic: bool = True, **_):
        x = th.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        with th.no_grad():
            a, _ = self.actor.act(x, deterministic=deterministic)
        a = np.clip(a.cpu().numpy(), self.low, self.high)
        return (a[0] if single else a), None
