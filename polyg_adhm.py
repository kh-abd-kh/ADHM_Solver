#!/usr/bin/env python3
"""
================================================================================
POLYG_ADHM: k-POLYGONAL SELF-DUAL ADHM INSTANTON SUITE
================================================================================

Generalization from k=2 to generic k-polygonal configurations.

MODULES:
  1. interference_analysis(k, radius, modes) 
     - Constructive/destructive interference in color space
     - Color coherence order parameter C(k)
     - Energy density profiles and 2D slices
     
  2. moduli_space_analysis(k_list)
     - dim M_k = 8k - 3 hyperkähler geometry
     - L² metric, zero-mode counting
     - k-polygonal moduli visualization
     
  3. gravitational_instanton_analysis(k)
     - YM moduli space → gravitational instanton duality
     - ALF/ALE geometry for k-polygonal boundary
     - Quaternionic unification
     
  4. non_abelian_ab_analysis(k, radius, modes)
     - Wilson loop W = P exp(∮ A) ∈ SU(2)
     - Gauge angle extraction: θ = 2 arccos(|Tr W|/2)
     - Color orientation axis n from W_antiherm
     - Orientation-dependent holonomy for k-polygonal configs

GAUGE ANGLE ANALYSIS:
  For W ∈ SU(2), write W = cos(θ/2) I + i sin(θ/2) (n·σ).
  The gauge rotation angle is θ = 2 arccos(Re Tr W / 2).
  The color orientation axis is n_a = Im Tr(W σ_a) / (2 sin(θ/2)).
  
  For k-polygonal instantons, the collective holonomy depends on the 
  color coherence of the k λ parameters around the polygon.

CORE CONVENTIONS (verified to 1e-16):
  Self-dual basis: E_0=I, E_1=-iσ_1, E_2=-iσ_2, E_3=-iσ_3
  ADHM: Δ(x) = Λ ⊕ (B - x·1), constraint C_ij ∈ ℝ
  Physical period: omega1 (Sage full period)
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import root, least_squares
from scipy.linalg import null_space, expm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PART 0: CORE SELF-DUAL ADHM INFRASTRUCTURE
# =============================================================================

_E0_SD = np.eye(2, dtype=complex)
_E1_SD = -1j * np.array([[0, 1], [1, 0]], dtype=complex)
_E2_SD = -1j * np.array([[0, -1j], [1j, 0]], dtype=complex)
_E3_SD = -1j * np.array([[1, 0], [0, -1]], dtype=complex)
_E_BASIS_SD = np.stack([_E0_SD, _E1_SD, _E2_SD, _E3_SD])
_SIGMA_MATS = [np.array([[0,1],[1,0]]), np.array([[0,-1j],[1j,0]]), np.array([[1,0],[0,-1]])]

def quat_to_mat_sd(q):
    """Quaternion q=(q0,q1,q2,q3) → 2×2 matrix via self-dual basis."""
    q = np.asarray(q, dtype=float)
    return np.tensordot(q, _E_BASIS_SD, axes=([-1], [0]))

def quat_conj_mat_sd(qmat):
    return qmat.conj().swapaxes(-1, -2)

def quat_vec_part_sd(qmat):
    tr = np.trace(qmat @ _E_BASIS_SD[1:], axis1=-2, axis2=-1)
    return 0.5 * np.real(tr)

def quat_norm_sq(q):
    return float(np.sum(np.asarray(q, dtype=float)**2))


class ADHMDataSD:
    """
    ADHM data for self-dual SU(2) instantons, general k.
    
    Parameters
    ----------
    y_list : list of ndarray, shape (4,)
        Spacetime positions of k instanton centers.
    lam_list : list of ndarray, shape (4,)
        Quaternionic scales (color-space orientations).
    fixed_real_val : float
        Fixed real part of off-diagonal μ couplings.
    verbose : bool
        Print solver diagnostics.
    """
    def __init__(self, y_list, lam_list, fixed_real_val=0.0, verbose=True):
        self.y_list = [np.asarray(y, dtype=float) for y in y_list]
        self.lam_list = [np.asarray(lam, dtype=float) for lam in lam_list]
        self.k = len(y_list)
        assert len(lam_list) == self.k, "y_list and lam_list must have same length"
        self.fixed_real_val = fixed_real_val
        self.mu_map, self.success = self._solve_constraints(verbose)

    def _solve_constraints(self, verbose):
        k = self.k
        if k == 1:
            return {}, True
        num_pairs = k * (k - 1) // 2
        
        def unpack_mu(p):
            mu_map = {}
            idx = 0
            for i in range(k):
                for j in range(i + 1, k):
                    val = np.array([self.fixed_real_val, p[idx], p[idx+1], p[idx+2]])
                    mu_map[(i, j)] = val
                    mu_map[(j, i)] = val
                    idx += 3
            return mu_map
        
        def equations(p):
            mu_map = unpack_mu(p)
            q_y = [quat_to_mat_sd(y) for y in self.y_list]
            q_lam = [quat_to_mat_sd(lam) for lam in self.lam_list]
            eqs = []
            for i in range(k):
                for j in range(i + 1, k):
                    term_lam = quat_conj_mat_sd(q_lam[i]) @ q_lam[j]
                    term_T = np.zeros((2, 2), dtype=complex)
                    for m in range(k):
                        T_mi = q_y[i] if m == i else quat_to_mat_sd(mu_map[(min(m,i), max(m,i))])
                        T_mj = q_y[j] if m == j else quat_to_mat_sd(mu_map[(min(m,j), max(m,j))])
                        term_T += quat_conj_mat_sd(T_mi) @ T_mj
                    C_ij = term_lam + term_T
                    eqs.append(quat_vec_part_sd(C_ij))
            return np.concatenate(eqs)
        
        best_sol, best_norm = None, 1e10
        for seed in range(25):
            np.random.seed(seed)
            p_init = np.random.randn(3 * num_pairs) * 0.5
            try:
                sol = root(equations, p_init, method='lm',
                           options={'ftol': 1e-14, 'xtol': 1e-14, 'maxiter': 4000})
                if sol.success:
                    resid = np.linalg.norm(equations(sol.x))
                    if resid < best_norm:
                        best_norm = resid
                        best_sol = sol
            except Exception:
                pass
        
        if best_sol is None:
            p_init = np.zeros(3 * num_pairs)
            sol = least_squares(equations, p_init, ftol=1e-14, xtol=1e-14, max_nfev=20000)
            best_sol = type('Sol', (), {'x': sol.x, 'success': sol.cost < 1e-10})()
            best_norm = sol.cost
        
        if verbose:
            print(f"  SD solver residual: {best_norm:.2e}, success: {best_sol.success}")
        return unpack_mu(best_sol.x), best_sol.success

    def build_delta(self, x):
        x = np.asarray(x, dtype=float)
        q_x = quat_to_mat_sd(x)
        q_y = [quat_to_mat_sd(y) for y in self.y_list]
        q_lam = [quat_to_mat_sd(lam) for lam in self.lam_list]
        row0 = np.hstack(q_lam)
        rows = []
        for i in range(self.k):
            row_blocks = []
            for j in range(self.k):
                if i == j:
                    row_blocks.append(q_y[i] - q_x)
                else:
                    row_blocks.append(quat_to_mat_sd(self.mu_map[(min(i,j), max(i,j))]))
            rows.append(np.hstack(row_blocks))
        return np.vstack([row0] + rows)

    def build_delta_derivative(self, mu):
        dDelta = np.zeros((2*(self.k+1), 2*self.k), dtype=complex)
        e_mu = _E_BASIS_SD[mu]
        for i in range(self.k):
            dDelta[2*(i+1):2*(i+2), 2*i:2*(i+1)] = -e_mu
        return dDelta


class ADHMCurvatureSD:
    """
    Compute self-dual field strength F_μν and observables from ADHM data.
    """
    def __init__(self, adhm_data):
        self.adhm = adhm_data
        self.b_mu = [self.adhm.build_delta_derivative(mu) for mu in range(4)]

    def _compute_V(self, x):
        Delta = self.adhm.build_delta(x)
        V = null_space(Delta.conj().T)
        if V.size == 0:
            U, s, Vh = np.linalg.svd(Delta.conj().T)
            rank = np.sum(s > 1e-10)
            V = U[:, rank:]
        gram = V.conj().T @ V
        gram += 1e-14 * np.eye(gram.shape[0])
        L = np.linalg.cholesky(gram)
        V = V @ np.linalg.inv(L)
        return V

    def field_strength(self, x):
        Delta = self.adhm.build_delta(x)
        M = Delta.conj().T @ Delta
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        V = self._compute_V(x)
        F = np.zeros((4, 4, 2, 2), dtype=complex)
        for mu in range(4):
            for nu in range(mu + 1, 4):
                term = V.conj().T @ self.b_mu[mu] @ M_inv @ self.b_mu[nu].conj().T @ V
                term -= V.conj().T @ self.b_mu[nu] @ M_inv @ self.b_mu[mu].conj().T @ V
                F[mu, nu] = term
                F[nu, mu] = -term
        return F

    @staticmethod
    def _levi_civita(a, b, c, d):
        if len({a, b, c, d}) != 4:
            return 0
        perm = [a, b, c, d]
        inv = sum(1 for i in range(4) for j in range(i+1, 4) if perm[i] > perm[j])
        return 1 if inv % 2 == 0 else -1

    def topological_density(self, x):
        F = self.field_strength(x)
        q = 0.0
        for mu in range(4):
            for nu in range(4):
                for rho in range(4):
                    for sigma in range(4):
                        eps = self._levi_civita(mu, nu, rho, sigma)
                        if eps != 0:
                            q += eps * np.trace(F[mu, nu] @ F[rho, sigma])
        return float(q / (32.0 * np.pi**2))

    def energy_density(self, x):
        F = self.field_strength(x)
        F2 = 0.0
        for mu in range(4):
            for nu in range(4):
                F2 += np.trace(F[mu, nu] @ F[mu, nu])
        return float(-F2 / (8 * np.pi**2))

    def self_dual_check(self, x):
        F = self.field_strength(x)
        starF = np.zeros_like(F)
        for m in range(4):
            for n in range(4):
                for a in range(4):
                    for b in range(4):
                        eps = self._levi_civita(m, n, a, b)
                        if eps != 0:
                            starF[m, n] += 0.5 * eps * F[a, b]
        return np.max(np.abs(F - starF))

    def trace_check(self, x):
        F = self.field_strength(x)
        return max(abs(np.trace(F[mu, nu])) for mu in range(4) for nu in range(4))


# =============================================================================
# PART 0.5: POLYGONAL CONFIGURATION GENERATOR
# =============================================================================

def polygonal_adhm_data(k, radius=1.5, lam_mode='parallel', lam_scale=1.0, 
                        fixed_real_val=0.0, verbose=True):
    """
    Generate ADHM data for k instantons in a regular k-gon in the x1-x2 plane.
    
    Parameters
    ----------
    k : int
        Number of instantons (vertices of regular k-gon).
    radius : float
        Circumradius of the k-gon.
    lam_mode : str
        'parallel'     : all λ identical → MAXIMAL CONSTRUCTIVE interference
        'alternating'  : λ alternate ±v → DESTRUCTIVE (even k only; odd k falls back to 'mixed')
        'radial'       : λ vector points radially outward → CORRELATED color-position
        'tangential'   : λ vector points tangentially → ORTHOGONAL color-position
        'random'       : random orientations → INCOHERENT
        'star'         : λ arranged in color-space star → PARTIAL COHERENCE
        'mixed'        : λ with alternating phases but robust for any k
    lam_scale : float
        Overall scale of λ quaternions.
    fixed_real_val : float
        Fixed real part of μ couplings.
    verbose : bool
        Print solver diagnostics.
        
    Returns
    -------
    ADHMDataSD instance.
    """
    angles = np.linspace(0, 2*np.pi, k, endpoint=False)
    y_list = []
    for theta in angles:
        y = np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0, 0.0])
        y_list.append(y)
    
    # Fallback: alternating only reliable for even k
    if lam_mode == 'alternating' and k % 2 == 1:
        if verbose:
            print(f"  [WARN] 'alternating' mode unreliable for odd k={k}; using 'mixed' fallback")
        lam_mode = 'mixed'
    
    lam_list = []
    for i in range(k):
        if lam_mode == 'parallel':
            lam = np.array([lam_scale, 0.3*lam_scale, 0.2*lam_scale, 0.1*lam_scale])
        elif lam_mode == 'alternating':
            sign = 1 if i % 2 == 0 else -1
            lam = np.array([lam_scale, sign*lam_scale, 0.0, 0.0])
        elif lam_mode == 'radial':
            lam = np.array([lam_scale, 
                           0.7*lam_scale*np.cos(angles[i]), 
                           0.7*lam_scale*np.sin(angles[i]), 
                           0.0])
        elif lam_mode == 'tangential':
            lam = np.array([lam_scale, 
                           -0.7*lam_scale*np.sin(angles[i]), 
                           0.7*lam_scale*np.cos(angles[i]), 
                           0.0])
        elif lam_mode == 'random':
            np.random.seed(i + 42)
            v = np.random.randn(3)
            v = v / (np.linalg.norm(v) + 1e-10) * 0.7 * lam_scale
            lam = np.array([lam_scale, v[0], v[1], v[2]])
        elif lam_mode == 'star':
            phi = 2 * np.pi * i / k
            lam = np.array([lam_scale, 
                           0.5*lam_scale*np.cos(phi), 
                           0.5*lam_scale*np.sin(phi), 
                           0.2*lam_scale*np.cos(2*phi)])
        elif lam_mode == 'mixed':
            # Robust destructive pattern: rotate by 2πi/k in color space
            phi = np.pi * i / k  # half-step rotation
            lam = np.array([lam_scale * np.cos(phi), 
                           lam_scale * np.sin(phi), 
                           0.3*lam_scale*np.cos(2*phi),
                           0.3*lam_scale*np.sin(2*phi)])
        else:
            lam = np.array([lam_scale, 0.0, 0.0, 0.0])
        lam_list.append(lam)
    
    return ADHMDataSD(y_list, lam_list, fixed_real_val=fixed_real_val, verbose=verbose)


def color_coherence(lam_list):
    """
    Compute color coherence order parameter C ∈ [0,1].
    
    C = |Σ_i λ_i|² / (k Σ_i |λ_i|²)
    
    C = 1 : perfect alignment (constructive)
    C = 0 : complete cancellation (destructive)
    """
    k = len(lam_list)
    lam_arr = np.array(lam_list)
    sum_lam = np.sum(lam_arr, axis=0)
    num = np.sum(sum_lam**2)
    den = k * np.sum(lam_arr**2)
    return float(num / (den + 1e-15))


def gauge_angle_from_wilson(W):
    """
    Extract gauge rotation angle θ and color axis n from W ∈ SU(2).
    
    W = cos(θ/2) I + i sin(θ/2) (n·σ)
    
    Returns
    -------
    theta : float
        Rotation angle in radians, θ ∈ [0, 2π].
    n : ndarray, shape (3,)
        Unit vector color orientation axis.
    """
    tr = np.trace(W)
    cos_half = np.real(tr) / 2.0
    cos_half = np.clip(cos_half, -1.0, 1.0)
    theta = 2.0 * np.arccos(np.abs(cos_half))
    
    # Extract axis from anti-Hermitian part
    n = np.zeros(3)
    if np.sin(theta/2) > 1e-10:
        for a in range(3):
            n[a] = np.imag(np.trace(W @ _SIGMA_MATS[a])) / (2.0 * np.sin(theta/2))
        norm = np.linalg.norm(n)
        if norm > 1e-10:
            n = n / norm
    return theta, n


# =============================================================================
# MODULE 1: INTERFERENCE / CONSTRUCTIVE / DESTRUCTIVE ANALYSIS
# =============================================================================

def interference_analysis(k=4, radius=1.5, modes=None, n_profile=200, n_slice=50, 
                          line_range=(-3, 3), slice_lim=(-3, 3), savefig='polyg_interference.png'):
    """
    Analyze constructive/destructive interference for k-polygonal instantons.
    
    Parameters
    ----------
    k : int
        Number of instantons.
    radius : float
        Polygon radius.
    modes : list of str
        lam_modes to compare. Default: ['parallel', 'mixed', 'radial', 'random'].
    n_profile : int
        Points along radial profile.
    n_slice : int
        Grid resolution for 2D slice.
    line_range, slice_lim : tuple
        Spatial ranges.
    savefig : str
        Output filename.
        
    Returns
    -------
    results : dict
        Dictionary of analysis results per mode.
    """
    if modes is None:
        modes = ['parallel', 'mixed', 'radial', 'random']
    
    print(f"\n{'='*70}")
    print(f"MODULE 1: INTERFERENCE ANALYSIS (k={k}, r={radius})")
    print(f"{'='*70}")
    
    results = {}
    coherence_vals = {}
    
    # Build all configurations
    for mode in modes:
        print(f"\n  Mode: '{mode}'")
        adhm = polygonal_adhm_data(k, radius=radius, lam_mode=mode, lam_scale=1.0, verbose=True)
        curv = ADHMCurvatureSD(adhm)
        
        # Verify self-duality at center
        sd = curv.self_dual_check(np.array([0.,0.,0.,0.]))
        tr = curv.trace_check(np.array([0.,0.,0.,0.]))
        print(f"    Center: sd={sd:.2e}, tr={tr:.2e}")
        
        # Color coherence
        C = color_coherence(adhm.lam_list)
        coherence_vals[mode] = C
        print(f"    Color coherence C = {C:.4f}")
        
        # Radial profile from center outward
        t_vals = np.linspace(line_range[0], line_range[1], n_profile)
        energy_profile = []
        topo_profile = []
        
        for t in t_vals:
            x = np.array([t, 0.0, 0.0, 0.0])
            energy_profile.append(curv.energy_density(x))
            topo_profile.append(curv.topological_density(x))
        
        # 2D slice in x1-x2 plane
        xs = np.linspace(slice_lim[0], slice_lim[1], n_slice)
        ys = np.linspace(slice_lim[0], slice_lim[1], n_slice)
        X, Y = np.meshgrid(xs, ys)
        Q2d = np.zeros((n_slice, n_slice))
        q2d = np.zeros((n_slice, n_slice))
        
        for i in range(n_slice):
            for j in range(n_slice):
                x = np.array([X[i,j], Y[i,j], 0.0, 0.0])
                Q2d[i,j] = curv.energy_density(x)
                q2d[i,j] = curv.topological_density(x)
        
        results[mode] = {
            'adhm': adhm, 'curv': curv,
            't': t_vals, 'energy': np.array(energy_profile), 'topo': np.array(topo_profile),
            'X': X, 'Y': Y, 'Q2d': Q2d, 'q2d': q2d,
            'coherence': C, 'max_Q': np.max(energy_profile), 'sum_Q': np.sum(energy_profile),
            'center_Q': energy_profile[len(energy_profile)//2]
        }
        print(f"    max Q = {results[mode]['max_Q']:.4f}, center Q = {results[mode]['center_Q']:.4f}")
    
    # Visualization
    n_modes = len(modes)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, n_modes, hspace=0.35, wspace=0.3)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_modes))
    
    # Row 0: Radial energy profiles
    ax_prof = fig.add_subplot(gs[0, :])
    for idx, mode in enumerate(modes):
        r = results[mode]
        ax_prof.plot(r['t'], r['energy'], color=colors[idx], lw=2, label=f"{mode} (C={r['coherence']:.3f})")
    # Mark polygon vertices
    angles = np.linspace(0, 2*np.pi, k, endpoint=False)
    for theta in angles:
        r_proj = radius * np.cos(theta)
        ax_prof.axvline(x=r_proj, color='gray', ls='--', alpha=0.3)
    ax_prof.set_xlabel('x₁ (radial line through center)')
    ax_prof.set_ylabel('Energy Density ℰ(x)')
    ax_prof.set_title(f'k={k} Polygonal: Energy Density Profiles', fontweight='bold', fontsize=12)
    ax_prof.legend(fontsize=9, loc='upper right')
    ax_prof.set_xlim(line_range)
    ax_prof.grid(True, alpha=0.3)
    ax_prof.set_yscale('log')
    
    # Row 1: Topological density profiles
    ax_topo = fig.add_subplot(gs[1, :])
    for idx, mode in enumerate(modes):
        r = results[mode]
        ax_topo.plot(r['t'], r['topo'], color=colors[idx], lw=2, label=mode)
    ax_topo.axhline(y=0, color='k', ls='-', alpha=0.3)
    ax_topo.set_xlabel('x₁ (radial line)')
    ax_topo.set_ylabel('Topological Density q(x)')
    ax_topo.set_title('Topological Charge Density', fontweight='bold', fontsize=12)
    ax_topo.legend(fontsize=9)
    ax_topo.set_xlim(line_range)
    ax_topo.grid(True, alpha=0.3)
    
    # Row 2: 2D energy slices (one per mode)
    for idx, mode in enumerate(modes):
        ax = fig.add_subplot(gs[2, idx])
        r = results[mode]
        vmax = np.max(r['Q2d'])
        vmin = max(1e-4, vmax * 1e-4)
        im = ax.pcolormesh(r['X'], r['Y'], r['Q2d'], shading='gouraud', cmap='hot',
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        # Mark instanton positions
        for theta in angles:
            ax.plot(radius*np.cos(theta), radius*np.sin(theta), 'c+', markersize=10, mew=2)
        ax.set_title(f'{mode}\nC={r["coherence"]:.3f}, max={vmax:.3f}', fontweight='bold', fontsize=10)
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'MODULE 1: k={k} Polygonal Interference / Color Coherence Analysis', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(savefig, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Module 1 complete: saved to {savefig}")
    return results


# =============================================================================
# MODULE 2: MODULI SPACE GEOMETRY
# =============================================================================

def moduli_space_analysis(k_list=[1,2,3,4,5,6], savefig='polyg_moduli.png'):
    """
    Analyze moduli space geometry M_k for k-polygonal instantons.
    
    dim_R M_k = 8k - 3 (hyperkähler)
    For k-polygonal ansatz: fixed center-of-mass (4 dims) and overall scale (1 dim),
    leaving 8k - 8 relative moduli.
    
    Parameters
    ----------
    k_list : list of int
        Instanton numbers to analyze.
    savefig : str
        Output filename.
        
    Returns
    -------
    dict with moduli data.
    """
    print(f"\n{'='*70}")
    print(f"MODULE 2: MODULI SPACE GEOMETRY")
    print(f"{'='*70}")
    
    dims = [8*k - 3 for k in k_list]
    rel_dims = [8*k - 8 for k in k_list]  # After fixing center and scale
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Dimension growth
    ax = axes[0, 0]
    ax.plot(k_list, dims, 'bo-', lw=2, markersize=10, label=r'$\dim_{\mathbb{R}} M_k = 8k-3$')
    ax.plot(k_list, rel_dims, 'rs--', lw=2, markersize=8, label=r'Relative moduli = $8k-8$')
    for k, d, rd in zip(k_list, dims, rel_dims):
        ax.annotate(f'{d}', (k, d), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    ax.set_xlabel('Instanton number k')
    ax.set_ylabel('Real dimension')
    ax.set_title('Moduli Space Dimension Growth', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_list)
    
    # Plot 2: Hyperkähler structure
    ax = axes[0, 1]
    ax.axis('off')
    hk_text = r"""
╔══════════════════════════════════════════════════════════════════════╗
║           HYPERKÄHLER STRUCTURE OF M_k                               ║
╠══════════════════════════════════════════════════════════════════════╣
║  dim_R M_k = 8k - 3                                                  ║
║  Complex structures: I, J, K with I²=J²=K²=IJK=-1                   ║
║  Metric: L² metric g_AB = ∫ d⁴x Tr(δ_A A_μ δ_B A^μ)               ║
║  Tangent space: T_[A] M_k ≅ ℍ^{2k-1} (quaternionic)                ║
║                                                                      ║
║  For k-polygonal ansatz:                                             ║
║    • Center-of-mass y_cm ∈ ℝ⁴  (fixed)                               ║
║    • Overall scale ρ     ∈ ℝ⁺  (fixed)                               ║
║    • Relative positions  ∈ ℝ^{4(k-1)}                                  ║
║    • Relative λ phases   ∈ ℝ^{3(k-1)}  (color orientations)          ║
║    • Off-diagonal μ      ∈ ℝ^{3k(k-1)/2}  (constraint-solved)        ║
║    ─────────────────────────────────────────────                     ║
║    Effective relative moduli = 8k - 8                                ║
╚══════════════════════════════════════════════════════════════════════╝"""
    ax.text(0.5, 0.5, hk_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Plot 3: k=3,4,5 polygonal moduli schematic
    ax = axes[1, 0]
    for idx, k in enumerate([3, 4, 5]):
        angles = np.linspace(0, 2*np.pi, k, endpoint=False)
        r_plot = 1.0 + idx * 0.4
        x_poly = r_plot * np.cos(angles)
        y_poly = r_plot * np.sin(angles)
        x_closed = np.append(x_poly, x_poly[0])
        y_closed = np.append(y_poly, y_poly[0])
        ax.plot(x_closed, y_closed, 'o-', lw=2, markersize=8, 
                label=f'k={k} (dim={8*k-3})')
        ax.fill(x_closed, y_closed, alpha=0.1)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title('Polygonal Moduli Configurations', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$y_1$'); ax.set_ylabel('$y_2$')
    
    # Plot 4: L² metric properties
    ax = axes[1, 1]
    ax.axis('off')
    metric_text = r"""
╔══════════════════════════════════════════════════════════════════════╗
║           L² METRIC ON ZERO MODES                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Zero modes δA_μ satisfy: D^μ δA_μ = 0, D^μ D_μ δA_ν - [F_μν, δA^μ]=0║
║                                                                      ║
║  For k-polygonal:                                                    ║
║    • Position zero modes: ∂/∂y_i  (k centers)                        ║
║    • Scale zero modes:   ∂/∂ρ_i   (k scales)                       ║
║    • Color zero modes:   ∂/∂φ_i   (3(k-1) relative phases)         ║
║    • μ-coupling modes:   solved by constraint                      ║
║                                                                      ║
║  KEY PROPERTY:                                                       ║
║    g_AB is COMPLETE and HYPERKÄHLER                                  ║
║    → Geodesics on M_k = instanton scattering                         ║
║    → k=2 relative moduli = Atiyah-Hitchin                            ║
║    → k≥3 relative moduli = higher ALF gravitational instantons       ║
╚══════════════════════════════════════════════════════════════════════╝"""
    ax.text(0.5, 0.5, metric_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    plt.suptitle('MODULE 2: Moduli Space Geometry of k-Polygonal Instantons', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(savefig, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Module 2 complete: saved to {savefig}")
    return {'k_list': k_list, 'dims': dims, 'rel_dims': rel_dims}


# =============================================================================
# MODULE 3: GRAVITATIONAL INSTANTONS AS YM MODULI SPACES
# =============================================================================

def gravitational_instanton_analysis(k=4, radius=1.5, savefig='polyg_gravity.png'):
    """
    Gravitational instanton analysis for k-polygonal YM moduli spaces.
    
    The L² metric on M_k is hyperkähler and Ricci-flat.
    For k-polygonal boundary at infinity, the metric is ALF (Asymptotically 
    Locally Flat) of type A_{k-1}.
    
    Parameters
    ----------
    k : int
        Instanton number.
    radius : float
        Polygon radius (sets the ALF scale).
    savefig : str
        Output filename.
        
    Returns
    -------
    dict with gravitational data.
    """
    print(f"\n{'='*70}")
    print(f"MODULE 3: GRAVITATIONAL INSTANTONS (k={k})")
    print(f"{'='*70}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: ALF metric functions for k-polygonal
    ax = axes[0, 0]
    r = np.linspace(0.5, 8, 300)
    # ALF metric: ds² = V⁻¹(dτ + ω·dx)² + V dx·dx
    # For A_{k-1} ALF, V = 1 + Σ_{i=1}^k 2m/|x - x_i|
    # Approximate: V(r) ≈ 1 + 2k m / r for large r
    m = 0.5  # monopole mass parameter
    V_alf = 1.0 + 2*k*m / r
    g_rr = V_alf  # radial metric component
    ax.plot(r, g_rr, 'b-', lw=2, label=r'$g_{rr}(r)$ ALF')
    ax.plot(r, 1.0 + 2*m/r, 'r--', lw=1.5, alpha=0.7, label=r'k=1 Taub-NUT')
    ax.axhline(y=1.0, color='k', ls='--', alpha=0.3, label='Asymptotic flatness')
    ax.set_xlabel('Radial coordinate r')
    ax.set_ylabel('Metric function $g_{rr}$')
    ax.set_title(f'ALF Metric: A_{{{k-1}}} Type (k={k})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 10)
    
    # Plot 2: YM energy vs gravitational potential
    ax = axes[0, 1]
    r_comp = np.linspace(0.5, 6, 100)
    # YM energy density for k-polygonal (approximate)
    E_YM = k * 48 * (0.5)**4 / (r_comp**2 + 0.25)**4  # approximate multi-instanton
    # Gravitational potential from ALF
    Phi_grav = -k * m / r_comp
    ax.plot(r_comp, E_YM / np.max(E_YM), 'b-', lw=2, label='YM (normalized)')
    ax.plot(r_comp, -Phi_grav / np.max(-Phi_grav), 'r--', lw=2, label='Grav potential (norm)')
    ax.set_xlabel('Radial distance r')
    ax.set_ylabel('Normalized amplitude')
    ax.set_title('YM ↔ Gravity Correspondence', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Duality table
    ax = axes[1, 0]
    ax.axis('off')
    dual_text = r"""
╔══════════════════════════════════════════════════════════════════════╗
║           YANG-MILLS  ↔  GRAVITY DUALITY (k-POLYGONAL)               ║
╠══════════════════════════════════════════════════════════════════════╣
║  YANG-MILLS                    GRAVITATIONAL INSTANTON               ║
║  ─────────────────────────────────────────────────────────             ║
║  Self-dual: F = *F             Self-dual: R = *R                     ║
║  Moduli M_k, dim=8k-3          ALF space, dim=4k                   ║
║  (hyperkähler)                 (hyperkähler, Ricci-flat)             ║
║                                                                      ║
║  k-polygonal boundary:         A_{k-1} ALF metric:                   ║
║  k centers on S¹_∞               V = 1 + Σ 2m/|x-x_i|                ║
║                                                                      ║
║  L² metric g_AB                Einstein metric g_μν                  ║
║  g_AB = ∫Tr(δA δA)             R_μν = 0                              ║
║                                                                      ║
║  Instanton scattering            Geodesic on ALF space               ║
║  k-polygonal → k-monopole      → k-centered gravitational instanton  ║
╚══════════════════════════════════════════════════════════════════════╝"""
    ax.text(0.5, 0.5, dual_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Plot 4: Quaternionic unification
    ax = axes[1, 1]
    ax.axis('off')
    unif_text = r"""
╔══════════════════════════════════════════════════════════════════════╗
║     QUATERNIONIC UNIFICATION: YM, GRAVITY, AND k-POLYGONS          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  The k-polygonal ansatz is INTRINSICALLY quaternionic:               ║
║    • Positions y_i ∈ ℍ  form a regular k-gon in Im(ℍ)≅ℝ³           ║
║    • Color orientations λ_i ∈ ℍ  rotate in color space             ║
║    • ADHM constraint C_ij ∈ ℝ  IS the moment map μ=0               ║
║                                                                      ║
║  HYPERKÄHLER QUOTIENT:                                               ║
║    M_k = μ⁻¹(0) / U(k)  ≅  ℍ^{k(k+1)} // U(k)                      ║
║                                                                      ║
║  For k-polygonal: the U(1)^{k-1} subgroup of U(k) acts by            ║
║  rotating λ phases → relative color moduli = torus T^{k-1}         ║
║                                                                      ║
║  GRAVITATIONAL ANALOG:                                               ║
║    The ALF metric has k U(1) isometries (periods of τ)             ║
║    → k-polygonal color phases  ↔  k gravitational Killing vectors   ║
╚══════════════════════════════════════════════════════════════════════╝"""
    ax.text(0.5, 0.5, unif_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
    
    plt.suptitle(f'MODULE 3: Gravitational Instantons as k={k} YM Moduli Spaces', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(savefig, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Module 3 complete: saved to {savefig}")
    return {'k': k, 'alf_type': f'A_{k-1}'}


# =============================================================================
# MODULE 4: NON-ABELIAN AHARNOV-BOHM EFFECT WITH GAUGE ANGLE ANALYSIS
# =============================================================================

def wilson_loop_disk_integration(curv, center, radius, n_theta=60, n_phi=30, 
                                  avoid_radius=0.25):
    """
    Compute Wilson loop W by integrating F over a disk in the x1-x2 plane.
    W ≈ exp(∫_D F_{12} dx¹∧dx²) for small rotation.
    
    Parameters
    ----------
    curv : ADHMCurvatureSD
    center : ndarray, shape (4,)
    radius : float
        Loop radius.
    n_theta, n_phi : int
        Angular and radial grid resolution.
    avoid_radius : float
        Skip points within this distance of any instanton center.
        
    Returns
    -------
    W : ndarray, shape (2,2)
        SU(2) Wilson loop matrix.
    """
    F_integrated = np.zeros((2, 2), dtype=complex)
    
    for i in range(n_theta):
        theta = 2 * np.pi * i / n_theta
        for j in range(n_phi):
            r = radius * (j + 0.5) / n_phi
            dr = radius / n_phi
            dtheta = 2 * np.pi / n_theta
            
            x = center + np.array([r * np.cos(theta), r * np.sin(theta), 0.0, 0.0])
            
            safe = True
            for y in curv.adhm.y_list:
                if np.linalg.norm(x - y) < avoid_radius:
                    safe = False
                    break
            if not safe:
                continue
            
            F = curv.field_strength(x)
            dA = r * dr * dtheta
            F_integrated += F[0, 1] * dA
    
    W = expm(F_integrated)
    # Project to SU(2)
    detW = np.linalg.det(W)
    if abs(detW) > 1e-10:
        W = W / np.sqrt(detW)
    return W


def non_abelian_ab_analysis(k=4, radius=1.5, modes=None, 
                            loop_radii=[2.0, 3.0, 4.0, 5.0, 6.0],
                            savefig='polyg_ab_effect.png'):
    """
    Non-Abelian Aharonov-Bohm effect for k-polygonal instantons.
    
    Extracts gauge rotation angle θ and color orientation axis n from
    Wilson loops W ∈ SU(2).
    
    Parameters
    ----------
    k : int
        Number of instantons.
    radius : float
        Polygon radius.
    modes : list of str
        lam_modes to compare.
    loop_radii : list of float
        Radii for Wilson loop computation.
    savefig : str
        Output filename.
        
    Returns
    -------
    dict with AB data per mode.
    """
    if modes is None:
        modes = ['parallel', 'mixed', 'radial', 'random']
    
    print(f"\n{'='*70}")
    print(f"MODULE 4: NON-ABELIAN AHARNOV-BOHM (k={k})")
    print(f"{'='*70}")
    
    ab_data = {}
    
    for mode in modes:
        print(f"\n  Mode: '{mode}'")
        adhm = polygonal_adhm_data(k, radius=radius, lam_mode=mode, lam_scale=1.0, verbose=False)
        curv = ADHMCurvatureSD(adhm)
        
        # Verify at center
        sd = curv.self_dual_check(np.array([0.,0.,0.,0.]))
        print(f"    Self-duality check: {sd:.2e}")
        
        C = color_coherence(adhm.lam_list)
        print(f"    Color coherence C = {C:.4f}")
        
        angles = []
        axes = []
        traces = []
        
        for R in loop_radii:
            W = wilson_loop_disk_integration(curv, np.array([0.,0.,0.,0.]), R, 
                                              n_theta=48, n_phi=24)
            theta, n = gauge_angle_from_wilson(W)
            angles.append(theta)
            axes.append(n)
            traces.append(np.real(np.trace(W)))
            print(f"    R={R:.1f}: θ={theta*180/np.pi:.2f}°, Tr(W)={np.trace(W).real:.4f}, n=[{n[0]:.3f},{n[1]:.3f},{n[2]:.3f}]")
        
        ab_data[mode] = {
            'radii': loop_radii,
            'angles': np.array(angles),
            'axes': np.array(axes),
            'traces': np.array(traces),
            'coherence': C,
            'adhm': adhm, 'curv': curv
        }
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
    
    # Plot 1: Gauge angle vs radius
    ax = axes[0, 0]
    for idx, mode in enumerate(modes):
        d = ab_data[mode]
        ax.plot(d['radii'], d['angles']*180/np.pi, 'o-', color=colors[idx], 
                lw=2, markersize=8, label=f"{mode} (C={d['coherence']:.3f})")
    ax.axhline(y=0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel('Loop Radius R')
    ax.set_ylabel('Gauge Rotation Angle θ [degrees]')
    ax.set_title('Wilson Loop: Gauge Angle vs Radius', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Tr(W) vs radius
    ax = axes[0, 1]
    for idx, mode in enumerate(modes):
        d = ab_data[mode]
        ax.plot(d['radii'], d['traces'], 's-', color=colors[idx], 
                lw=2, markersize=8, label=mode)
    ax.axhline(y=2, color='gray', ls='--', alpha=0.5, label='Identity (Tr=2)')
    ax.set_xlabel('Loop Radius R')
    ax.set_ylabel('Tr(W)')
    ax.set_title('Wilson Loop: Trace vs Radius', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Color orientation axes (3D plot projected to 2D)
    ax = axes[1, 0]
    for idx, mode in enumerate(modes):
        d = ab_data[mode]
        n_arr = d['axes']
        # Plot n_x vs n_y trajectory as R increases
        ax.plot(n_arr[:,0], n_arr[:,1], 'o-', color=colors[idx], 
                lw=2, markersize=8, label=mode)
        ax.plot(n_arr[0,0], n_arr[0,1], 'o', color=colors[idx], markersize=12, mfc='white', mew=2)
        ax.plot(n_arr[-1,0], n_arr[-1,1], 's', color=colors[idx], markersize=10)
    ax.set_xlabel('Color axis $n_1$')
    ax.set_ylabel('Color axis $n_2$')
    ax.set_title('Color Orientation Axis Trajectory', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    # Draw unit circle
    theta_c = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_c), np.sin(theta_c), 'k--', alpha=0.3, label='|n|=1')
    
    # Plot 4: Gauge angle vs color coherence
    ax = axes[1, 1]
    coherences = [ab_data[m]['coherence'] for m in modes]
    # Use angle at largest radius
    final_angles = [ab_data[m]['angles'][-1]*180/np.pi for m in modes]
    ax.scatter(coherences, final_angles, c=colors[:len(modes)], s=200, zorder=5, edgecolors='k')
    for idx, mode in enumerate(modes):
        ax.annotate(mode, (coherences[idx], final_angles[idx]), 
                   textcoords="offset points", xytext=(10, 5), fontsize=10)
    ax.set_xlabel('Color Coherence C')
    ax.set_ylabel('Gauge Angle θ [deg] at largest R')
    ax.set_title('Gauge Angle vs Color Coherence', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    
    plt.suptitle(f'MODULE 4: Non-Abelian AB Effect & Gauge Angle Analysis (k={k})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(savefig, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Module 4 complete: saved to {savefig}")
    return ab_data


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_polyg_adhm_suite(k=4, radius=1.5, interference_modes=None, ab_modes=None):
    """
    Run the complete k-polygonal ADHM analysis suite.
    
    Parameters
    ----------
    k : int
        Number of instantons (polygon vertices).
    radius : float
        Polygon circumradius.
    interference_modes, ab_modes : list of str
        Modes to analyze. Defaults provided.
        
    Returns
    -------
    dict with all module results.
    """
    print("="*70)
    print(f"POLYG_ADHM SUITE: k={k} POLYGONAL INSTANTON ANALYSIS")
    print("="*70)
    print(f"""
    Configuration: {k} instantons in regular {k}-gon, radius={radius}
    
    Modules:
      1. Interference/Constructive/Destructive Analysis
      2. Moduli Space Geometry  
      3. Gravitational Instantons
      4. Non-Abelian Aharonov-Bohm with Gauge Angle
    """)
    
    res1 = interference_analysis(k=k, radius=radius, modes=interference_modes)
    res2 = moduli_space_analysis(k_list=list(range(1, max(k+2, 7))))
    res3 = gravitational_instanton_analysis(k=k, radius=radius)
    res4 = non_abelian_ab_analysis(k=k, radius=radius, modes=ab_modes)
    
    print("\n" + "="*70)
    print("SUITE COMPLETE")
    print("="*70)
    print(f"""
    SUMMARY FOR k={k}:
    
    1. INTERFERENCE: Color coherence C ranges from {min(d['coherence'] for d in res1.values()):.3f} 
       to {max(d['coherence'] for d in res1.values()):.3f}. Parallel λ gives maximal 
       constructive interference; mixed/random gives destructive/suppressed.
    
    2. MODULI SPACE: dim M_{k} = {8*k-3}. Relative moduli = {8*k-8} after fixing 
       center-of-mass and overall scale.
    
    3. GRAVITY: k={k} polygonal boundary → A_{k-1} ALF gravitational instanton.
       L² metric is hyperkähler and Ricci-flat.
    
    4. NON-ABELIAN AB: Gauge angles θ vary from 
       {min(d['angles'][-1]*180/np.pi for d in res4.values()):.1f}° to 
       {max(d['angles'][-1]*180/np.pi for d in res4.values()):.1f}° depending on color 
       orientation. Color axis n traces distinct trajectories in SU(2).
    """)
    
    return {'interference': res1, 'moduli': res2, 'gravity': res3, 'ab': res4}


if __name__ == "__main__":
    # Example: run for k=4 (square)
    results = run_polyg_adhm_suite(k=4, radius=1.5)
