from const import EstimatorConstant
import numpy as np


class EKF:
    """
    Extended Kalman Filter for the whale tracking problem under Gaussian noise.

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator (initial state bounds, process 
            noise paramaters, etc...)
    """

    def __init__(self, estimator_constant: EstimatorConstant):
        self.constant = estimator_constant
        self.xm = None
        self.Pm = None
        self.k  = 0

    def theta_star(self, k: int) -> float:
        c = self.constant
        return c.theta_dive if k < c.N * c.k_switch_frac else -c.theta_dive

    def dynamics(self, x: np.ndarray, k: int = None) -> np.ndarray:
        if k is None:
            k = self.k
        c  = self.constant
        Ts = c.Ts
        pos_x, pos_y, z, psi, theta, p, r_psi, r_theta, a = x
        theta_star = self.theta_star(k)
        f = np.empty(9)
        f[0] = pos_x + p * np.cos(theta) * np.cos(psi) * Ts
        f[1] = pos_y + p * np.cos(theta) * np.sin(psi) * Ts
        f[2] = z     + p * np.sin(theta) * Ts
        f[3] = psi   + r_psi   * Ts
        f[4] = theta + r_theta * Ts
        f[5] = p     + (a - c.cd * p) * Ts
        f[6] = r_psi   + c.K_yaw    * (c.omega_yaw - r_psi)
        f[7] = r_theta + c.K_theta_p * (theta_star  - theta) - c.K_theta_d * r_theta
        f[8] = a       + c.K_p_p    * (c.p_cruise   - p)    - c.K_p_d    * a
        return f

    def compute_jacobian_A(self, x: np.ndarray, k: int = None) -> np.ndarray:
        if k is None:
            k = self.k
        c  = self.constant
        Ts = c.Ts
        _, _, _, psi, theta, p, _, _, _ = x
        A = np.eye(9)
        A[0, 3] = -p * np.cos(theta) * np.sin(psi) * Ts
        A[0, 4] = -p * np.sin(theta) * np.cos(psi) * Ts
        A[0, 5] =      np.cos(theta) * np.cos(psi) * Ts
        A[1, 3] =  p * np.cos(theta) * np.cos(psi) * Ts
        A[1, 4] = -p * np.sin(theta) * np.sin(psi) * Ts
        A[1, 5] =      np.cos(theta) * np.sin(psi) * Ts
        A[2, 4] = p * np.cos(theta) * Ts
        A[2, 5] =     np.sin(theta) * Ts
        A[3, 6] = Ts
        A[4, 7] = Ts
        A[5, 5] = 1.0 - c.cd * Ts
        A[5, 8] = Ts
        A[6, 6] = 1.0 - c.K_yaw
        A[7, 4] = -c.K_theta_p
        A[7, 7] =  1.0 - c.K_theta_d
        A[8, 5] = -c.K_p_p
        A[8, 8] =  1.0 - c.K_p_d
        return A

    def compute_jacobian_L(self, x: np.ndarray) -> np.ndarray:
        L = np.zeros((9, 3))
        L[6, 0] = 1.0; L[7, 1] = 1.0; L[8, 2] = 1.0
        return L

    def process_noise(self, x: np.ndarray) -> np.ndarray:
        c = self.constant
        return np.diag([c.sigma_r_psi**2, c.sigma_r_theta**2, c.sigma_a**2])

    def build_H(self, usbl_active: bool) -> np.ndarray:
        if usbl_active:
            H = np.zeros((5, 9))
            H[0, 0] = 1.0; H[1, 1] = 1.0
            H[2, 0] = 1.0; H[3, 1] = 1.0; H[4, 2] = 1.0
        else:
            H = np.zeros((3, 9))
            H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 2] = 1.0
        return H

    def build_R(self, usbl_active: bool, z_depth: float) -> np.ndarray:
        c        = self.constant
        sigma_S2 = c.sigma_S ** 2
        if usbl_active:
            sigma_U2 = (c.sigma_base + c.alpha * abs(z_depth)) ** 2
            return np.diag([sigma_U2, sigma_U2, sigma_S2, sigma_S2, sigma_S2])
        else:
            return np.diag([sigma_S2, sigma_S2, sigma_S2])

    def initialize(self) -> tuple[np.ndarray, np.ndarray]:
        c  = self.constant
        xm = np.zeros(9)
        Pm = np.zeros((9, 9))
        xm[2] = c.z0
        xm[4] = 0.5 * (c.theta0_min + c.theta0_max)
        xm[5] = 0.5 * (c.p0_min     + c.p0_max)
        Pm[0, 0] = c.R**2 / 4.0
        Pm[1, 1] = c.R**2 / 4.0
        Pm[3, 3] = c.psi_bound**2 / 3.0
        Pm[4, 4] = (c.theta0_max - c.theta0_min)**2 / 12.0
        Pm[5, 5] = (c.p0_max     - c.p0_min)**2     / 12.0
        self.xm = xm.copy()
        self.Pm = Pm.copy()
        self.k  = 0
        return xm, Pm

    def estimate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Standard EKF step"""
        xp = self.dynamics(self.xm, self.k)
        A  = self.compute_jacobian_A(self.xm, self.k)
        self.k += 1
        L  = self.compute_jacobian_L(self.xm)
        Q  = self.process_noise(self.xm)
        Pp = A @ self.Pm @ A.T + L @ Q @ L.T

        usbl_active = not (np.isnan(measurement[0]) or np.isnan(measurement[1]))
        z_meas      = measurement if usbl_active else measurement[2:]
        z_depth     = xp[2]

        H = self.build_H(usbl_active)
        R = self.build_R(usbl_active, z_depth)
        S = H @ Pp @ H.T + R
        K = np.linalg.solve(S.T, (Pp @ H.T).T).T

        xm = xp + K @ (z_meas - H @ xp)
        Pm = (np.eye(9) - K @ H) @ Pp

        self.xm = xm.copy()
        self.Pm = Pm.copy()
        return xm, Pm


class NonGaussianEstimator:
    """
    Bootstrap Particle Filter for the whale tracking problem under non-Gaussian noise.

    Algorithm:
    ----------
    Standard bootstrap PF: particles are propagated through the deterministic
    dynamics, perturbed by uniform process noise, weighted by the exact
    measurement likelihood, and resampled when the effective sample size drops
    below a threshold.

    Likelihood models:
    ------------------
    - Process noise : Uniform, matching the variance of the Gaussian case.
    - USBL          : Gaussian with state-dependent standard deviation,
                      consistent with the uniform USBL noise model (matching variance).
    - Sonar         : Exact asymmetric bimodal Gaussian mixture,
                      with constants precomputed at construction time.

    Key design choices:
    -------------------
    - Likelihood tempering (BETA > 1 scales negative log-weights, effectively
      flattening the weight distribution to prevent collapse from the sharp
      bimodal sonar likelihood). Separate BETA values for USBL-active and
      dropout phases allow tuning each regime independently.
    - Adaptive systematic resampling triggered when N_eff/N < resample_frac,
      followed by roughening on psi (col 3), theta (col 4), and p (col 5)
      using the Silverman kernel rule std = K * E * N^(-1/9). Columns 6-8
      (r_psi, r_theta, a) are excluded as they already receive process noise
      every step.
    - Circular mean for psi via arctan2(mean_sin, mean_cos) to correctly
      handle angle wraparound in the posterior mean computation.
    - Depth clipped to >= 0 in dynamics and after roughening.
    - psi wrapped to [-pi, pi] after roughening.

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator (initial state bounds, process
            noise parameters, sensor noise characteristics, behavioral model
            coefficients, etc.)
    """

    def __init__(self, estimator_constant: EstimatorConstant):
        self.constant = estimator_constant

        self.N              = 70000   # particle number
        self.BETA_usbl      = 1.3     # tempering for USBL-active steps
        self.BETA_sonar     = 1.3     # tempering for dropout steps
        self.roughening_K_psi = 0.005 # smaller scale for psi
        self.roughening_K_ang = 0.015 # standard scale for theta and p
        self.resample_frac  = 0.5     # resample frequency

        self.particles = None
        self.weights   = None
        self.k         = 0

        # Pre-compute sonar likelihood constants once
        c    = estimator_constant
        beta = c.sonar_bimodality
        w1   = c.sonar_mix_weight
        sS   = c.sigma_S
        sig2 = (1.0 - 9.0 * beta**2) * sS**2
        sig  = float(np.sqrt(sig2))
        self._sl_mu1  = float( beta * sS)
        self._sl_mu2  = float(-9.0 * beta * sS)
        self._sl_isig = 1.0 / sig
        self._sl_C    = 1.0 / (sig * np.sqrt(2.0 * np.pi))
        self._sl_w1   = float(w1)
        self._sl_w2   = float(1.0 - w1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _theta_star(self, k: int) -> float:
        c = self.constant
        return c.theta_dive if k < c.N * c.k_switch_frac else -c.theta_dive

    def _particle_dynamics(self, x: np.ndarray, k: int) -> np.ndarray:
        """Vectorized deterministic dynamics"""
        c          = self.constant
        Ts         = c.Ts
        theta_star = self._theta_star(k)

        pos_x = x[:, 0]; pos_y = x[:, 1]; z       = x[:, 2]
        psi   = x[:, 3]; theta = x[:, 4]; p       = x[:, 5]
        r_psi = x[:, 6]; r_theta = x[:, 7]; a     = x[:, 8]

        f = np.empty_like(x)
        f[:, 0] = pos_x + p * np.cos(theta) * np.cos(psi) * Ts
        f[:, 1] = pos_y + p * np.cos(theta) * np.sin(psi) * Ts
        f[:, 2] = np.maximum(0.0, z + p * np.sin(theta) * Ts)
        f[:, 3] = psi     + r_psi   * Ts
        f[:, 4] = theta   + r_theta * Ts
        f[:, 5] = p       + (a - c.cd * p) * Ts
        f[:, 6] = r_psi   + c.K_yaw    * (c.omega_yaw - r_psi)
        f[:, 7] = r_theta + c.K_theta_p * (theta_star  - theta) - c.K_theta_d * r_theta
        f[:, 8] = a       + c.K_p_p    * (c.p_cruise   - p)     - c.K_p_d    * a
        return f

    def _process_noise(self, x: np.ndarray) -> np.ndarray:
        """Uniform process noise on r_psi, r_theta, a."""
        c = self.constant
        N = self.N
        x[:, 6] += np.random.uniform(-np.sqrt(3.0) * c.sigma_r_psi,
                                      np.sqrt(3.0) * c.sigma_r_psi,   N)
        x[:, 7] += np.random.uniform(-np.sqrt(3.0) * c.sigma_r_theta,
                                      np.sqrt(3.0) * c.sigma_r_theta, N)
        x[:, 8] += np.random.uniform(-np.sqrt(3.0) * c.sigma_a,
                                      np.sqrt(3.0) * c.sigma_a,       N)
        return x

    def _log_lik_sonar(self, z_s: float, x_state: np.ndarray) -> np.ndarray:
        """Exact bimodal Gaussian mixture log-likelihood"""
        innov = z_s - x_state
        e1 = np.clip(-0.5 * ((innov - self._sl_mu1) * self._sl_isig)**2, -500.0, 0.0)
        e2 = np.clip(-0.5 * ((innov - self._sl_mu2) * self._sl_isig)**2, -500.0, 0.0)
        pdf = self._sl_w1 * self._sl_C * np.exp(e1) \
            + self._sl_w2 * self._sl_C * np.exp(e2)
        return np.log(np.maximum(pdf, 1e-300))

    def _log_lik_usbl(self, z_u: float, x_state: np.ndarray,
                      z_depth: np.ndarray) -> np.ndarray:
        """Gaussian likelihood for USBL with state-dependent variance"""
        c = self.constant
        sigma_U = np.maximum(c.sigma_base + c.alpha * np.abs(z_depth),
                             c.sigma_base * 0.1)
        innov = z_u - x_state
        return (np.clip(-0.5 * (innov / sigma_U)**2, -500.0, 0.0)
                - np.log(sigma_U * np.sqrt(2.0 * np.pi)))

    def _resample(self, weights: np.ndarray) -> np.ndarray:
        """Systematic resampling"""
        N         = self.N
        cdf       = np.cumsum(weights); cdf[-1] = 1.0
        positions = (np.arange(N) + np.random.uniform()) / N
        return np.searchsorted(cdf, positions)

    # ------------------------------------------------------------------
    # Initialize
    # ------------------------------------------------------------------

    def initialize(self) -> np.ndarray:
        c = self.constant
        N = self.N

        # Vectorized rejection sampling for uniform disk
        xy = np.empty((N, 2)); filled = 0
        while filled < N:
            batch = np.random.uniform(-c.R, c.R, size=((N - filled) * 4, 2))
            valid = batch[np.sum(batch**2, axis=1) <= c.R**2]
            take  = min(len(valid), N - filled)
            xy[filled:filled + take] = valid[:take]; filled += take

        self.particles        = np.zeros((N, 9))
        self.particles[:, 0] = xy[:, 0]
        self.particles[:, 1] = xy[:, 1]
        self.particles[:, 2] = c.z0
        self.particles[:, 3] = np.random.uniform(-c.psi_bound, c.psi_bound, N)
        self.particles[:, 4] = np.random.uniform(c.theta0_min, c.theta0_max, N)
        self.particles[:, 5] = np.random.uniform(c.p0_min, c.p0_max, N)

        self.weights = np.ones(N) / N
        self.k       = 0

        return np.average(self.particles, weights=self.weights, axis=0)

    # ------------------------------------------------------------------
    # Estimate step
    # ------------------------------------------------------------------

    def estimate(self, measurement: np.ndarray) -> np.ndarray:
        """
        Bootstrap PF step: propagate -> weight -> resample -> estimate
        """
        N = self.N

        # ---- 1. Propagate ----
        X = self._particle_dynamics(self.particles, self.k)
        self.k += 1
        X = self._process_noise(X)

        # ---- 2. Log-likelihood weights ----
        log_w = np.zeros(N)
        usbl_active = not (np.isnan(measurement[0]) or np.isnan(measurement[1]))
        if usbl_active:
            log_w += self._log_lik_usbl(measurement[0], X[:, 0], X[:, 2])
            log_w += self._log_lik_usbl(measurement[1], X[:, 1], X[:, 2])
        log_w += self._log_lik_sonar(measurement[2], X[:, 0])
        log_w += self._log_lik_sonar(measurement[3], X[:, 1])
        log_w += self._log_lik_sonar(measurement[4], X[:, 2])

        # Tempering — scales negative log-weights, flattening weight distribution
        beta = self.BETA_usbl if usbl_active else self.BETA_sonar
        if beta != 1.0:
            log_w *= beta

        # ---- 3. Stable normalization ----
        log_w -= np.max(log_w)
        log_w  = np.clip(log_w, -500.0, 0.0)
        w      = np.exp(log_w)
        w_sum  = w.sum()
        w      = w / w_sum if w_sum > 0.0 else np.ones(N) / N

        # ---- 4. Adaptive systematic resampling ----
        N_eff = 1.0 / np.dot(w, w)

        if N_eff < N * self.resample_frac:
            idx = self._resample(w)
            X   = X[idx]
            w   = np.ones(N) / N

            # Adaptive roughening after resampling
            N19 = N ** (-1.0 / 9.0)
            for col, K in [(3, self.roughening_K_psi),
                           (4, self.roughening_K_ang),
                           (5, self.roughening_K_ang)]:
                E   = np.max(X[:, col]) - np.min(X[:, col])
                std = K * E * N19
                X[:, col] += np.random.normal(0.0, max(std, 1e-8), N)

            X[:, 2] = np.maximum(0.0, X[:, 2])
            X[:, 3] = (X[:, 3] + np.pi) % (2.0 * np.pi) - np.pi

        # ---- 5. Store ----
        self.particles = X
        self.weights   = w

        # ---- 6. Posterior mean — circular mean for psi ----
        xm    = np.average(X, weights=w, axis=0)
        s_psi = np.average(np.sin(X[:, 3]), weights=w)
        c_psi = np.average(np.cos(X[:, 3]), weights=w)
        xm[3] = np.arctan2(s_psi, c_psi)

        return xm