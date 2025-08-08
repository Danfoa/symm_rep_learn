"""
Thomas' Cyclically Symmetric Attractor Implementation

The Thomas attractor is a 3D strange attractor described by the differential equations:
    dx/dt = sin(y) - b*x
    dy/dt = sin(z) - b*y
    dz/dt = sin(x) - b*z

where b is a parameter that controls the system's behavior.

Reference: Thomas, René (1999). "Deterministic chaos seen in terms of feedback circuits"
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

# Optional plotting dependencies
try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    if TYPE_CHECKING:
        import plotly.graph_objects as go

# Default state domain box for initial conditions
# State vector is [x, y, z]
DEFAULT_STATE_DOMAIN = np.array(
    [
        [-5.0, 5.0],  # x position
        [-5.0, 5.0],  # y position
        [-5.0, 5.0],  # z position
    ]
)


class ThomasAttractor:
    """
    Thomas' Cyclically Symmetric Attractor

    A 3D strange attractor with cyclical symmetry in x, y, and z variables.
    Can be viewed as the trajectory of a frictionally dampened particle
    moving in a 3D lattice of forces.
    """

    def __init__(self, b: float = 0.19, noise_scale: float = 0.0):
        """
        Initialize the Thomas attractor with parameter b and noise scale.

        Parameters:
        -----------
        b : float, default=0.19
            Dissipation parameter. Controls the system's behavior:
            - b > 1: Single stable equilibrium at origin
            - b ≈ 0.32899: Hopf bifurcation (limit cycle)
            - b ≈ 0.208186: Onset of chaos
            - b = 0.19: Well within chaotic regime (good default)
        noise_scale : float, default=0.0
            Scale of Brownian noise added to the dynamics.
            Set to 0.0 for deterministic behavior.
        """
        self.b = b
        self.noise_scale = noise_scale

    def dynamics(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Thomas attractor differential equations with optional Brownian noise.

        Parameters:
        -----------
        t : float
            Time (not used in autonomous system)
        state : np.ndarray
            State vector [x, y, z]

        Returns:
        --------
        np.ndarray
            Derivative vector [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state

        dxdt = np.sin(y) - self.b * x
        dydt = np.sin(z) - self.b * y
        dzdt = np.sin(x) - self.b * z

        derivatives = np.array([dxdt, dydt, dzdt])

        # Add Brownian noise if specified
        if self.noise_scale > 0:
            noise = np.random.normal(0, self.noise_scale, size=3)
            derivatives += noise

        return derivatives

    def simulate(
        self,
        initial_state: np.ndarray,
        t_span: Tuple[float, float],
        dt: float = 0.01,
        method: str = "RK45",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a single trajectory of the Thomas attractor.

        Parameters:
        -----------
        initial_state : np.ndarray
            Initial conditions [x0, y0, z0]
        t_span : Tuple[float, float]
            Time span (t_start, t_end)
        dt : float, default=0.01
            Time step for output
        method : str, default='RK45'
            Integration method for solve_ivp
        rtol, atol : float
            Relative and absolute tolerance for integration

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (time_points, trajectory) where trajectory is shape (n_points, 3)
        """

        assert initial_state.shape == (3,), "Initial state must be a 3D vector [x, y, z]"
        # Create time points
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

        # Solve the differential equation
        sol = solve_ivp(fun=self.dynamics, t_span=t_span, y0=initial_state, t_eval=t_eval, method=method)

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        return sol.t, sol.y.T

    def generate_random_initial_conditions(
        self, n_trajectories: int, state_domain: Optional[np.ndarray] = None, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate random initial conditions within a specified domain.

        Parameters:
        -----------
        n_trajectories : int
            Number of initial conditions to generate
        state_domain : np.ndarray, optional
            Domain bounds as [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            If None, uses DEFAULT_STATE_DOMAIN
        seed : int, optional
            Random seed for reproducibility

        Returns:
        --------
        np.ndarray
            Initial conditions array of shape (n_trajectories, 3)
        """
        if seed is not None:
            np.random.seed(seed)

        if state_domain is None:
            state_domain = DEFAULT_STATE_DOMAIN

        # Generate random points within the domain
        initial_conditions = np.random.uniform(
            low=state_domain[:, 0], high=state_domain[:, 1], size=(n_trajectories, 3)
        )

        return initial_conditions

    def get_state_symmetry_rep(self):  # -> "escnn.representation.Representation":
        """
        Get the symmetry representation of the Thomas attractor state.

        The Thomas attactor has Cyclic group 3 symmety for any parameter b.

        Returns:
        --------
        escnn.representation.Representation
            The symmetry representation of the Thomas attractor state.
        """

        from escnn.group import CyclicGroup

        G = CyclicGroup(3)

        rep_S = G.regular_representation

        # Ensure the dynamics are G-equivariant.
        # z = np.random.uniform(-5.0, 5.0, size=(3,))
        # z_dot = self.dynamics(0.0, z)

        # for g in G.elements:
        #     print(f"rep_S({g}):\n ", rep_S(g))
        #     g_z = np.einsum("ij,j->i", rep_S(g), z)
        #     g_z_dot = np.einsum("ij,j->i", rep_S(g), z_dot)
        #     g_z_dot_gt = self.dynamics(0.0, g_z)
        #     assert np.allclose(g_z_dot, g_z_dot_gt), (
        #         f"Thomas attractor dynamics are not G-equivariant for group element {g}: {g_z_dot} != {g_z_dot_gt}"
        #     )
        return rep_S

    def generate_dataset(
        self,
        n_trajectories: int,
        trajectory_length: float,
        dt: float = 0.01,
        state_domain: Optional[np.ndarray] = None,
        initial_conditions: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        method: str = "RK45",
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate a dataset of multiple trajectories with random initial conditions.

        Parameters:
        -----------
        n_trajectories : int
            Number of trajectories to generate
        trajectory_length : float
            Length of each trajectory in time units
        dt : float, default=0.01
            Time step
        state_domain : np.ndarray, optional
            Domain for initial conditions
        seed : int, optional
            Random seed for reproducibility
        method : str, default='RK45'
            Integration method

        Returns:
        --------
        Tuple[List[np.ndarray], List[np.ndarray]]
            (time_arrays, trajectory_arrays) where each trajectory is shape (n_points, 3)
        """
        assert initial_conditions is None or (initial_conditions.ndim == 2 and initial_conditions.shape[1] == 3)

        if initial_conditions is None:
            # Generate random initial conditions
            initial_conditions = self.generate_random_initial_conditions(n_trajectories, state_domain, seed)

        time_arrays = []
        trajectory_arrays = []

        for i, ic in enumerate(initial_conditions):
            # Simulate from initial condition for the requested trajectory length
            t, traj = self.simulate(initial_state=ic, t_span=(0, trajectory_length), dt=dt, method=method)

            time_arrays.append(t)
            trajectory_arrays.append(traj)

        return time_arrays, trajectory_arrays

    def plot_trajectories_3d(
        self,
        time_arrays: Optional[List[np.ndarray]] = None,
        trajectory_arrays: Optional[List[np.ndarray]] = None,
        n_trajectories: int = 10,
        trajectory_length: float = 10.0,
        dt: float = 0.1,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot 3D trajectories of the Thomas attractor using plotly.

        Parameters:
        -----------
        time_arrays : List[np.ndarray], optional
            Time arrays for trajectories. If None, generates new trajectories.
        trajectory_arrays : List[np.ndarray], optional
            Trajectory arrays. If None, generates new trajectories.
        n_trajectories : int, default=10
            Number of trajectories to generate if none provided
        trajectory_length : float, default=10.0
            Length of trajectories in time units if generating new ones
        dt : float, default=0.1
            Time step if generating new trajectories
        title : str, optional
            Plot title

        Returns:
        --------
        go.Figure
            Plotly figure object
        """
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for plotting. Install with: pip install plotly")

        # Generate trajectories if not provided
        if time_arrays is None or trajectory_arrays is None:
            time_arrays, trajectory_arrays = self.generate_dataset(
                n_trajectories=n_trajectories, trajectory_length=trajectory_length, dt=dt, seed=42
            )

        fig = go.Figure()

        # Generate colors for trajectories
        n_traj = len(trajectory_arrays)
        color_palette = [f"hsl({i * 360 / n_traj}, 70%, 50%)" for i in range(n_traj)]

        for i, (t_array, traj_array) in enumerate(zip(time_arrays, trajectory_arrays)):
            # Extract position coordinates
            x_pos = traj_array[:, 0]
            y_pos = traj_array[:, 1]
            z_pos = traj_array[:, 2]

            color = color_palette[i % len(color_palette)]

            # Plot trajectory as transparent line
            fig.add_trace(
                go.Scatter3d(
                    x=x_pos,
                    y=y_pos,
                    z=z_pos,
                    mode="lines",
                    line=dict(color=color, width=2),
                    opacity=0.7,
                    name=f"Trajectory {i + 1}",
                    showlegend=False,
                )
            )

            # Plot initial condition as sphere marker (first point of trajectory)
            fig.add_trace(
                go.Scatter3d(
                    x=[x_pos[0]],
                    y=[y_pos[0]],
                    z=[z_pos[0]],
                    mode="markers",
                    marker=dict(color=color, size=4, symbol="circle", opacity=1.0),
                    name=f"IC {i + 1}",
                    showlegend=False,
                )
            )

        # Update layout
        noise_str = f", noise={self.noise_scale}" if self.noise_scale > 0 else ""
        fig.update_layout(
            title=title or f"Thomas Attractor Trajectories (b={self.b}{noise_str})",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
            margin=dict(l=0, r=0, b=0, t=40),
            showlegend=False,
        )

        return fig


def example_usage():
    """
    Example usage of the ThomasAttractor class.
    """
    np.set_printoptions(precision=3, suppress=True)
    # Create Thomas attractor instance with noise
    thomas = ThomasAttractor(b=0.19, noise_scale=0.5)
    rep_S = thomas.get_state_symmetry_rep()

    # Generate a single trajectory
    initial_state = np.array([0.1, 0.1, 0.1])
    print(f"Simulating trajectory with initial state: {initial_state}")
    t, trajectory = thomas.simulate(initial_state=initial_state, t_span=(0, 50), dt=0.05)

    print(f"Generated trajectory with {len(t)} points")
    print(f"Trajectory shape: {trajectory.shape}")

    # Generate a dataset using the original method
    T = 20
    N = 70

    initial_states = thomas.generate_random_initial_conditions(
        n_trajectories=N, state_domain=DEFAULT_STATE_DOMAIN, seed=42
    )
    # Add group orbit of initial states:
    G_init = []
    for g in rep_S.group.elements:
        G_init.append(np.einsum("ij,...j->...i", rep_S(g), initial_states))
    G_init = np.concatenate(G_init, axis=0)

    time_arrays, trajectory_arrays = thomas.generate_dataset(
        n_trajectories=N, trajectory_length=T, dt=0.05, seed=42, initial_conditions=G_init
    )
    print(f"Generated {len(time_arrays)} trajectories with shape {trajectory_arrays[0].shape}")


    # # Create and show 3D plot
    fig = thomas.plot_trajectories_3d(
        time_arrays=time_arrays, trajectory_arrays=trajectory_arrays, title="Thomas Attractor with Noise"
    )
    fig.show(renderer="browser")  # Force browser renderer



if __name__ == "__main__":
    example_usage()
