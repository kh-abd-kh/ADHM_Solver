import numpy as np
from scipy.optimize import root

# ===================================================================
# 1. Quaternionic Arithmetic (2x2 Complex Matrices Representation)
# ===================================================================
I = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

e_0 = I
e_1 = 1j * sigma_x
e_2 = 1j * sigma_y
e_3 = 1j * sigma_z

def to_quaternion(v):
    """Converts a 4-vector into a 2x2 complex matrix representation of a quaternion."""
    v = np.asarray(v, dtype=complex)
    return v[0]*e_0 + v[1]*e_1 + v[2]*e_2 + v[3]*e_3

def quaternion_conj(q):
    """Returns the conjugate transpose of a quaternion matrix."""
    return q.conj().T

def quaternion_vec_parts(q):
    """Extracts the three vector components (imaginary parts) of a quaternion matrix."""
    return np.array([
        0.5 * np.real(np.trace(q @ quaternion_conj(e_1))),
        0.5 * np.real(np.trace(q @ quaternion_conj(e_2))),
        0.5 * np.real(np.trace(q @ quaternion_conj(e_3)))
    ])

# ===================================================================
# 2. ADHM Constraint Solver
# ===================================================================
def solve_adhm_constraints(y_list, lam_list, fixed_real_val=0.3):
    """
    Solves the ADHM algebraic constraints for arbitrary instanton charge k.
    
    Parameters:
        y_list (list of arrays): Centers of the k instantons.
        lam_list (list of arrays): Scales/orientations of the k instantons.
        fixed_real_val (float): Fixed real component of off-diagonal T_ij matrices 
                                to break the gauge invariance.
                                
    Returns:
        mu_map (dict): Solved T_ij components mapped as {(i, j): array([real, im_i, im_j, im_k])}.
        success (bool): True if the optimizer converged.
    """
    k = len(y_list)
    num_pairs = k * (k - 1) // 2
    
    if num_pairs == 0:
        return {}, True
        
    def unpack_mu(p):
        mu_map = {}
        idx = 0
        for i in range(k):
            for j in range(i+1, k):
                val = np.array([fixed_real_val, p[idx], p[idx+1], p[idx+2]])
                mu_map[(i, j)] = val
                mu_map[(j, i)] = val
                idx += 3
        return mu_map

    def equations(p):
        mu_map = unpack_mu(p)
        q_y = [to_quaternion(y) for y in y_list]
        q_lam = [to_quaternion(lam) for lam in lam_list]
        
        eqs = []
        for i in range(k):
            for j in range(i+1, k):
                term_lam = quaternion_conj(q_lam[i]) @ q_lam[j]
                term_T = np.zeros((2, 2), dtype=complex)
                for m in range(k):
                    T_mi = q_y[i] if m == i else to_quaternion(mu_map[(min(m, i), max(m, i))])
                    T_mj = q_y[j] if m == j else to_quaternion(mu_map[(min(m, j), max(m, j))])
                    term_T += quaternion_conj(T_mi) @ T_mj
                
                C_ij = term_lam + term_T
                eqs.append(quaternion_vec_parts(C_ij))
        return np.concatenate(eqs)

    p_init = np.zeros(3 * num_pairs)
    sol = root(equations, p_init, method='lm')
    if not sol.success:
        sol = root(equations, p_init)
    return unpack_mu(sol.x), sol.success

# ===================================================================
# 3. Physical Field Calculations (Superpotentials and Densities)
# ===================================================================
def superpotential_thooft(coords, y_list, lam_list):
    """
    Computes the 't Hooft superpotential W_Hooft(x) = ln Phi(x) at a 4D coordinate.
    """
    coords = np.asarray(coords, dtype=float)
    k = len(y_list)
    phi = 1.0
    for i in range(k):
        d_2 = np.sum((coords - y_list[i])**2)
        rho_2 = np.sum(lam_list[i]**2)
        if d_2 < 1e-9:
            return 1e10
        phi += rho_2 / d_2
    return np.log(phi)

def superpotential_adhm(coords, y_list, lam_list, mu_map):
    """
    Computes the ADHM superpotential W_ADHM(x) = 0.5 * ln det(Delta_dag @ Delta) at a 4D coordinate.
    """
    coords = np.asarray(coords, dtype=float)
    k = len(y_list)
    
    q_x = to_quaternion(coords)
    q_y = [to_quaternion(y) for y in y_list]
    q_lam = [to_quaternion(lam) for lam in lam_list]
    
    blocks = []
    blocks.append(q_lam)
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(q_y[i] + q_x)
            else:
                row.append(to_quaternion(mu_map[(min(i, j), max(i, j))]))
        blocks.append(row)
        
    Delta = np.block(blocks)
    Delta_dag = Delta.conj().T
    D_dag_D = Delta_dag @ Delta
    
    det_val = np.real(np.linalg.det(D_dag_D))
    return 0.5 * np.log(det_val) if det_val > 1e-15 else -100.0

def compute_bilaplacian_4D(f_func, coords, h=0.1):
    """Computes the 4D numerical bi-Laplacian of a scalar function f_func at coords."""
    coords = np.asarray(coords, dtype=float)
    e = np.eye(4)
    def laplacian(c):
        val = f_func(c)
        lap = -8.0 * val
        for mu in range(4):
            lap += f_func(c + h * e[mu]) + f_func(c - h * e[mu])
        return lap / (h**2)
    val_lap = laplacian(coords)
    bilap = -8.0 * val_lap
    for mu in range(4):
        bilap += laplacian(coords + h * e[mu]) + laplacian(coords - h * e[mu])
    return bilap / (h**2)

def energy_density_thooft(coords, y_list, lam_list, h=0.1):
    """Computes the 't Hooft topological charge density Q_Hooft(x) at coordinates."""
    f_func = lambda x: superpotential_thooft(x, y_list, lam_list)
    bilap = compute_bilaplacian_4D(f_func, coords, h)
    return -bilap / (16.0 * np.pi**2)

def energy_density_adhm(coords, y_list, lam_list, mu_map, h=0.1):
    """Computes the ADHM topological charge density Q_ADHM(x) at coordinates."""
    f_func = lambda x: superpotential_adhm(x, y_list, lam_list, mu_map)
    bilap = compute_bilaplacian_4D(f_func, coords, h)
    return -bilap / (16.0 * np.pi**2)

# ===================================================================
# 4. Grid and Slice Generators
# ===================================================================
def generate_plane_coords(plane='x0_x1', limits=(-3, 3), grid_size=30, offset=0.25):
    """
    Generates grid coordinate arrays for 2D slices within 4D space.
    
    Allowed planes: 'x0_x1', 'x0_x2', 'x0_x3', 'x1_x2', 'x1_x3', 'x2_x3'
    """
    u = np.linspace(limits[0], limits[1], grid_size)
    v = np.linspace(limits[0], limits[1], grid_size)
    U, V = np.meshgrid(u, v)
    
    coords_grid = np.full((grid_size, grid_size, 4), offset)
    
    mapping = {
        'x0_x1': (0, 1),
        'x0_x2': (0, 2),
        'x0_x3': (0, 3),
        'x1_x2': (1, 2),
        'x1_x3': (1, 3),
        'x2_x3': (2, 3),
    }
    
    if plane not in mapping:
        raise ValueError(f"Invalid plane: {plane}. Choose from {list(mapping.keys())}")
        
    idx_u, idx_v = mapping[plane]
    coords_grid[..., idx_u] = U
    coords_grid[..., idx_v] = V
    return u, v, coords_grid

# ===================================================================
# 5. Default Presets for k = 2, 3, 4, 5
# ===================================================================
def get_preset_configuration(k):
    """
    Returns default coordinate (y_list) and scale/orientation (lam_list) presets 
    for instanton charge k = 2, 3, 4, or 5.
    """
    if k == 2:
        y = [
            np.array([-1.0, -0.5, 0.0, 0.0]), 
            np.array([ 1.0,  0.5, 0.0, 0.0])
        ]
        lam = [
            np.array([1.0, 0.0, 0.0, 0.0]), 
            np.array([0.8, 0.2, 0.0, 0.0])
        ]
    elif k == 3:
        y = [
            np.array([-1.5, -0.5, 0.0, 0.0]), 
            np.array([ 0.0,  1.0, 0.0, 0.0]), 
            np.array([ 1.5, -0.5, 0.0, 0.0])
        ]
        lam = [
            np.array([1.0, 0.0, 0.0, 0.0]), 
            np.array([0.8, 0.2, 0.0, 0.0]), 
            np.array([0.9, 0.0, 0.3, 0.0])
        ]
    elif k == 4:
        y = [
            np.array([-1.5, -0.5, 0.0, 0.0]), 
            np.array([-0.5,  1.0, 0.0, 0.0]), 
            np.array([ 0.5, -1.0, 0.0, 0.0]), 
            np.array([ 1.5,  0.5, 0.0, 0.0])
        ]
        lam = [
            np.array([1.0, 0.0, 0.0, 0.0]), 
            np.array([0.8, 0.2, 0.0, 0.0]), 
            np.array([0.9, 0.0, 0.3, 0.0]), 
            np.array([0.7, 0.1, 0.1, 0.1])
        ]
    elif k == 5:
        y = [
            np.array([-2.0,  0.0, 0.0, 0.0]), 
            np.array([-1.0,  1.0, 0.0, 0.0]), 
            np.array([ 0.0, -1.0, 0.0, 0.0]), 
            np.array([ 1.0,  1.0, 0.0, 0.0]), 
            np.array([ 2.0,  0.0, 0.0, 0.0])
        ]
        lam = [
            np.array([1.0, 0.0, 0.0, 0.0]), 
            np.array([0.9, 0.1, 0.0, 0.0]), 
            np.array([0.8, 0.0, 0.2, 0.0]), 
            np.array([0.7, 0.1, 0.1, 0.0]), 
            np.array([0.6, 0.2, 0.0, 0.1])
        ]
    else:
        raise ValueError(f"Presets are only defined for k = 2, 3, 4, 5. Received k = {k}")
    return y, lam

# ===================================================================
# 6. Object-Oriented Interface Wrapper
# ===================================================================
class InstantonSystem:
    """
    State container for an SU(2) multi-instanton configuration of charge k.
    
    Provides convenient unified methods for fields calculations, grid generation, 
    and slice visualization.
    """
    def __init__(self, y_list, lam_list, fixed_real_val=0.3):
        self.y_list = [np.asarray(y, dtype=float) for y in y_list]
        self.lam_list = [np.asarray(lam, dtype=float) for lam in lam_list]
        self.k = len(y_list)
        self.fixed_real_val = fixed_real_val
        self.mu_map, self.solver_success = solve_adhm_constraints(self.y_list, self.lam_list, self.fixed_real_val)
        
    @classmethod
    def from_preset(cls, k, fixed_real_val=0.3):
        """Initializes an InstantonSystem from a built-in default configuration for charge k."""
        y_list, lam_list = get_preset_configuration(k)
        return cls(y_list, lam_list, fixed_real_val)
        
    def superpotential_adhm(self, coords):
        return superpotential_adhm(coords, self.y_list, self.lam_list, self.mu_map)
        
    def superpotential_thooft(self, coords):
        return superpotential_thooft(coords, self.y_list, self.lam_list)
        
    def energy_density_adhm(self, coords, h=0.1):
        return energy_density_adhm(coords, self.y_list, self.lam_list, self.mu_map, h)
        
    def energy_density_thooft(self, coords, h=0.1):
        return energy_density_thooft(coords, self.y_list, self.lam_list, h)

    def compute_plane_data(self, plane='x0_x1', limits=(-3, 3), grid_size_Q=12, grid_size_W=30, offset=0.25, h=0.1):
        """Evaluates superpotentials and energy densities over a grid slice."""
        u_Q, v_Q, coords_Q = generate_plane_coords(plane, limits, grid_size_Q, offset)
        Q_adhm = np.zeros((grid_size_Q, grid_size_Q))
        Q_thooft = np.zeros((grid_size_Q, grid_size_Q))
        
        for j in range(grid_size_Q):
            for i in range(grid_size_Q):
                Q_adhm[j, i] = self.energy_density_adhm(coords_Q[j, i], h=h)
                Q_thooft[j, i] = self.energy_density_thooft(coords_Q[j, i], h=h)
                
        u_W, v_W, coords_W = generate_plane_coords(plane, limits, grid_size_W, offset)
        W_adhm = np.zeros((grid_size_W, grid_size_W))
        W_thooft = np.zeros((grid_size_W, grid_size_W))
        
        for j in range(grid_size_W):
            for i in range(grid_size_W):
                W_adhm[j, i] = self.superpotential_adhm(coords_W[j, i])
                W_thooft[j, i] = self.superpotential_thooft(coords_W[j, i])
                
        return {
            'u_Q': u_Q, 'v_Q': v_Q, 'Q_adhm': Q_adhm, 'Q_thooft': Q_thooft,
            'u_W': u_W, 'v_W': v_W, 'W_adhm': W_adhm, 'W_thooft': W_thooft
        }

    def plot_slice(self, plane='x0_x1', label_name=None, limits=(-3, 3), grid_size_Q=12, grid_size_W=30, offset=0.25, h=0.1, dpi=150):
        """Generates and saves a comparative plot for a single slice."""
        import matplotlib.pyplot as plt
        if label_name is None:
            label_name = f"k{self.k}_{plane}"
            
        data = self.compute_plane_data(plane, limits, grid_size_Q, grid_size_W, offset, h)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Row 1: Energy Density
        ax = axes[0, 0]
        im = ax.imshow(data['Q_thooft'], extent=[limits[0], limits[1], limits[0], limits[1]], origin='lower', cmap='plasma', interpolation='bicubic')
        ax.set_title(f"'t Hooft Q ({plane})")
        fig.colorbar(im, ax=ax)
        
        ax = axes[0, 1]
        im = ax.imshow(data['Q_adhm'], extent=[limits[0], limits[1], limits[0], limits[1]], origin='lower', cmap='plasma', interpolation='bicubic')
        ax.set_title(f"ADHM Q ({plane})")
        fig.colorbar(im, ax=ax)
        
        # Row 2: Superpotential
        ax = axes[1, 0]
        im = ax.contourf(data['u_W'], data['v_W'], data['W_thooft'], 20, cmap='viridis')
        ax.set_title(f"'t Hooft W ({plane})")
        fig.colorbar(im, ax=ax)
        
        ax = axes[1, 1]
        im = ax.contourf(data['u_W'], data['v_W'], data['W_adhm'], 20, cmap='viridis')
        ax.set_title(f"ADHM W ({plane})")
        fig.colorbar(im, ax=ax)
        
        fig.tight_layout()
        fig.savefig(f"instanton_{label_name}.png", dpi=dpi)
        return fig

    def plot_standard_comparison(self, label_name=None, limits=(-3, 3), grid_size_Q=12, grid_size_W=30, offset=0.25, h=0.1, dpi=150):
        """Recreates the original comparative figures over three standard planes."""
        import matplotlib.pyplot as plt
        if label_name is None:
            label_name = f"k{self.k}"
            
        planes = ['x0_x1', 'x0_x2', 'x1_x2']
        fig_Q, axes_Q = plt.subplots(3, 2, figsize=(10, 12))
        fig_W, axes_W = plt.subplots(3, 2, figsize=(10, 12))
        
        for idx, plane in enumerate(planes):
            data = self.compute_plane_data(plane, limits, grid_size_Q, grid_size_W, offset, h)
            
            # Energy Density
            ax_t = axes_Q[idx, 0]
            im_t = ax_t.imshow(data['Q_thooft'], extent=[limits[0], limits[1], limits[0], limits[1]], origin='lower', cmap='plasma', interpolation='bicubic')
            ax_t.set_title(f"'t Hooft Q ({plane})")
            fig_Q.colorbar(im_t, ax=ax_t)
            
            ax_a = axes_Q[idx, 1]
            im_a = ax_a.imshow(data['Q_adhm'], extent=[limits[0], limits[1], limits[0], limits[1]], origin='lower', cmap='plasma', interpolation='bicubic')
            ax_a.set_title(f"ADHM Q ({plane})")
            fig_Q.colorbar(im_a, ax=ax_a)
            
            # Superpotential
            ax_W_t = axes_W[idx, 0]
            im_W_t = ax_W_t.contourf(data['u_W'], data['v_W'], data['W_thooft'], 20, cmap='viridis')
            ax_W_t.set_title(f"'t Hooft Superpotential ({plane})")
            fig_W.colorbar(im_W_t, ax=ax_W_t)
            
            ax_W_a = axes_W[idx, 1]
            im_W_a = ax_W_a.contourf(data['u_W'], data['v_W'], data['W_adhm'], 20, cmap='viridis')
            ax_W_a.set_title(f"ADHM Superpotential ({plane})")
            fig_W.colorbar(im_W_a, ax=ax_W_a)
            
        fig_Q.tight_layout()
        fig_Q.savefig(f"instanton_{label_name}_energy.png", dpi=dpi)
        
        fig_W.tight_layout()
        fig_W.savefig(f"instanton_{label_name}_superpotential.png", dpi=dpi)
        plt.close('all')
        return fig_Q, fig_W
