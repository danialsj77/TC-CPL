"""Named variants of the revision experiments. Each entry overrides
exp_lib.DEFAULT_SPEC (= the submitted TC-CPL protocol) in the listed keys only."""

VARIANTS = {
    # the method and the paper's ablation
    "tccpl":         {},
    "fixed_100_50":  dict(adapt=False, lam_ov0=100.0, lam_sat0=50.0),   # Fixed-price TC-CPL (the paper's M1 ablation)
    # fixed-price sensitivity grid
    "fixed_10_5":    dict(adapt=False, lam_ov0=10.0, lam_sat0=5.0),     # = TC-CPL's initial prices
    "fixed_10_50":   dict(adapt=False, lam_ov0=10.0, lam_sat0=50.0),
    "fixed_100_5":   dict(adapt=False, lam_ov0=100.0, lam_sat0=5.0),
    "fixed_500_5":   dict(adapt=False, lam_ov0=500.0, lam_sat0=5.0),    # = TC-CPL's cap
    "fixed_500_50":  dict(adapt=False, lam_ov0=500.0, lam_sat0=50.0),
    "fixed_2000_50": dict(adapt=False, lam_ov0=2000.0, lam_sat0=50.0),
    # dual-controller settings
    "lmax_100":      dict(lam_max=100.0),
    "lmax_2000":     dict(lam_max=2000.0),
    "tau_0p5":       dict(tau_ov=0.5),          # positive tolerance (Remark 1), kWh/day above tightened limit
    "tau_2":         dict(tau_ov=2.0),
    "pi_kp0":        dict(kp=0.0),              # pure integral = classic dual ascent
    "pi_ki0p5":      dict(ki=0.5),
    "pi_ki8":        dict(ki=8.0),
    # component ablations
    "no_demos":      dict(demos=False),
    "no_residual":   dict(residual=False),
    "no_m2":         dict(demos=False, residual=False),
    "no_gridblock":  dict(grid_block=False),
    "flat_encoder":  dict(encoder="flat"),
    "inc_cafap":     dict(incumbent="CAFAP"),
    "inc_cafap_rho1": dict(incumbent="CAFAP", rho=1.0),
    "inc_rr_rho1":   dict(rho=1.0),
    # added after the comment-by-comment review
    "margin0":       dict(ov_margin=0.0),             # no 5% back-off
    "pi_al":         dict(kp=2.0, ki=2.0),            # Kp = Ki, augmented-Lagrangian-like
    "tau_2_lmax2000": dict(tau_ov=2.0, lam_max=2000.0),  # cap irrelevant once tau > 0
    # PID Lagrangian (derivative on the growing violation, Stooke et al. 2020), 19 Oct 2026
    "pid_kd5":        dict(kd=5.0),
    "pid_kd20":       dict(kd=20.0),
    "pid_kd5_tau2":   dict(kd=5.0, tau_ov=2.0),
    "pid_kd20_tau2":  dict(kd=20.0, tau_ov=2.0),
}
