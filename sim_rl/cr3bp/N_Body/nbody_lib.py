"""N-body simulation library for inertial and rotating reference frames.

This module provides a small but flexible toolkit for gravitational
N-body simulations, including:

- :class:`Body` – container for a single celestial body,
- :class:`NBodySystem` – dynamical system in inertial or rotating frame,
- :class:`Simulator` – wrapper around :func:`scipy.integrate.solve_ivp`,
- :class:`Visualizer` – simple 2D/3D plotting and animation utilities,
- :class:`Analysis` – basic conservation-law diagnostics,
- :func:`plot_full_comparison_dashboard` – helper for comparing two solvers.

The code is designed to be lightweight and educational, while still
being useful for research-grade experiments (e.g. CR3BP studies).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (imported for side effects)
from matplotlib.animation import FuncAnimation


# ==============================================================================
# CLASS 1: Body
# ==============================================================================


class Body:
    """Container for a single celestial body.

    Parameters
    ----------
    mass : float
        Mass of the body.
    position : numpy.ndarray
        Initial position vector of shape ``(dims,)``.
    velocity : numpy.ndarray
        Initial velocity vector of shape ``(dims,)``.
    name : str, optional
        Optional name for labeling plots and diagnostics.
    radius : float, optional
        Effective radius used for collision detection.
    """

    def __init__(
        self,
        mass: float,
        position: np.ndarray,
        velocity: np.ndarray,
        name: str = "",
        radius: float = 0.0,
    ) -> None:
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.name = name
        self.radius = radius


# ==============================================================================
# CLASS 2: NBodySystem
# ==============================================================================


class NBodySystem:
    """Dynamical system for N-body gravity in inertial or rotating frames.

    The system can be evolved either in an inertial reference frame
    (standard Newtonian N-body problem) or in a rotating frame
    suitable for the circular restricted three-body problem (CR3BP).

    Parameters
    ----------
    bodies : list[Body]
        List of all bodies in the simulation.
    G : float, optional
        Gravitational constant used in the equations of motion.
    softening_factor : float, optional
        Softening parameter added to the squared distances to avoid
        numerical singularities when two bodies get very close.
    frame : {"inertial", "rotating"}, optional
        Reference frame for the dynamics. In the rotating case, additional
        parameters must be provided.
    omega : float, optional
        Angular velocity of the rotating frame (if ``frame == "rotating"``).
    primary_masses : list[float], optional
        Masses of the two primary bodies in the CR3BP setting.
        Required if ``frame == "rotating"``.
    primary_positions : list[numpy.ndarray], optional
        Positions of the two primary bodies in the rotating frame.
        Required if ``frame == "rotating"``.
    """

    def __init__(
        self,
        bodies: list[Body],
        G: float = 6.67430e-11,
        softening_factor: float = 0.0,
        frame: str = "inertial",
        omega: float = 0.0,
        primary_masses: list | None = None,
        primary_positions: list | None = None,
    ) -> None:
        if not bodies:
            raise ValueError("The list of bodies cannot be empty.")

        self.bodies = bodies
        self.G = G
        self.softening_factor = softening_factor
        self.dims = bodies[0].position.shape[0]
        self.n_bodies = len(bodies)
        self.masses = np.array([b.mass for b in self.bodies]).reshape(-1, 1)

        self.frame = frame
        if self.frame == "rotating":
            if omega == 0.0 or primary_masses is None or primary_positions is None:
                raise ValueError(
                    "For 'rotating' frame, omega, primary_masses, and primary_positions "
                    "must be provided."
                )
            self.omega = omega
            self.primary_masses = primary_masses
            self.primary_positions = primary_positions

    # ------------------------------------------------------------------
    # Differential equations
    # ------------------------------------------------------------------

    def _differential_equation_inertial(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the N-body ODE in an inertial frame.

        Parameters
        ----------
        t : float
            Current simulation time (unused but required by `solve_ivp`).
        y : numpy.ndarray
            Flattened state vector with positions followed by velocities.

        Returns
        -------
        numpy.ndarray
            Flattened time derivative of the state vector.
        """
        positions = y[: self.n_bodies * self.dims].reshape(self.n_bodies, self.dims)
        velocities = y[self.n_bodies * self.dims :].reshape(
            self.n_bodies,
            self.dims,
        )

        diffs = (
            positions.reshape(1, self.n_bodies, self.dims)
            - positions.reshape(self.n_bodies, 1, self.dims)
        )
        dist_sq = np.sum(diffs**2, axis=2) + self.softening_factor**2
        np.fill_diagonal(dist_sq, np.inf)
        inv_r3 = dist_sq**(-1.5)

        accelerations = self.G * np.sum(
            self.masses.T.reshape(1, self.n_bodies, 1)
            * diffs
            * inv_r3.reshape(self.n_bodies, self.n_bodies, 1),
            axis=1,
        )

        return np.concatenate(
            [
                velocities.flatten(),
                accelerations.flatten(),
            ]
        )

    def _differential_equation_rotating(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the CR3BP-like ODE in a rotating frame.

        The dynamics includes gravitational forces from two primaries as
        well as Coriolis and centrifugal terms induced by the rotating
        reference frame.

        Parameters
        ----------
        t : float
            Current simulation time (unused but required by `solve_ivp`).
        y : numpy.ndarray
            Flattened state vector with positions followed by velocities.

        Returns
        -------
        numpy.ndarray
            Flattened time derivative of the state vector.
        """
        positions = y[: self.n_bodies * self.dims].reshape(self.n_bodies, self.dims)
        velocities = y[self.n_bodies * self.dims :].reshape(
            self.n_bodies,
            self.dims,
        )
        accelerations = np.zeros_like(positions)

        r1, r2 = self.primary_positions
        m1, m2 = self.primary_masses

        for i, r in enumerate(positions):
            v = velocities[i]
            delta1 = r1 - r
            delta2 = r2 - r

            a_grav1 = self.G * m1 * delta1 / (
                np.linalg.norm(delta1) ** 3 + self.softening_factor**2
            )
            a_grav2 = self.G * m2 * delta2 / (
                np.linalg.norm(delta2) ** 3 + self.softening_factor**2
            )

            if self.dims == 2:
                centrifugal = self.omega**2 * r
                coriolis = 2 * self.omega * np.array([-v[1], v[0]])
                accelerations[i] = a_grav1 + a_grav2 + centrifugal + coriolis
            else:
                omega_vec = np.array([0.0, 0.0, self.omega])
                centrifugal = -np.cross(omega_vec, np.cross(omega_vec, r))
                coriolis = -2 * np.cross(omega_vec, v)
                accelerations[i] = a_grav1 + a_grav2 + centrifugal + coriolis

        return np.concatenate(
            [
                velocities.flatten(),
                accelerations.flatten(),
            ]
        )

    def differential_equation(self, t: float, y: np.ndarray) -> np.ndarray:
        """Return the appropriate right-hand side based on the frame.

        Parameters
        ----------
        t : float
            Current simulation time.
        y : numpy.ndarray
            Flattened state vector.

        Returns
        -------
        numpy.ndarray
            Flattened time derivative of the state vector.

        Raises
        ------
        ValueError
            If an unknown frame type is configured.
        """
        if self.frame == "inertial":
            return self._differential_equation_inertial(t, y)
        if self.frame == "rotating":
            return self._differential_equation_rotating(t, y)
        raise ValueError(f"Unknown frame type: {self.frame}")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def collision_event(self, t: float, y: np.ndarray) -> float:
        """Event function for collision detection between bodies.

        The event is defined as the minimum distance between any two
        bodies minus the sum of their radii. When this quantity becomes
        zero or negative, a collision is detected.

        Parameters
        ----------
        t : float
            Current simulation time (unused).
        y : numpy.ndarray
            Flattened state vector.

        Returns
        -------
        float
            Minimum clearance between any two bodies.
        """
        positions = y[: self.n_bodies * self.dims].reshape(self.n_bodies, self.dims)
        diffs = (
            positions.reshape(1, self.n_bodies, self.dims)
            - positions.reshape(self.n_bodies, 1, self.dims)
        )
        distances = np.linalg.norm(diffs, axis=2)

        radii = np.array([b.radius for b in self.bodies])
        radii_sum = radii.reshape(-1, 1) + radii.reshape(1, -1)

        clearance = distances - radii_sum
        np.fill_diagonal(clearance, np.inf)
        min_clearance = float(np.min(clearance))
        return min_clearance


# Configure event properties for SciPy's solve_ivp
NBodySystem.collision_event.terminal = True
NBodySystem.collision_event.direction = -1


# ==============================================================================
# CLASS 3: Simulator
# ==============================================================================


class Simulator:
    """Wrapper around :func:`scipy.integrate.solve_ivp` for N-body systems.

    Parameters
    ----------
    system : NBodySystem
        The dynamical system to be integrated.
    """

    def __init__(self, system: NBodySystem) -> None:
        self.system = system

    def run(
        self,
        t_span: tuple[float, float],
        solver_name: str = "DOP853",
        num_steps: int = 1000,
        time_unit: str = "seconds",
        rtol: float = 1e-9,
        atol: float = 1e-9,
        max_step: float = np.inf,
        events=None,
    ):
        """Integrate the system over a time span.

        Parameters
        ----------
        t_span : tuple[float, float]
            Start and end time of the integration interval.
        solver_name : str, optional
            Name of the SciPy ODE solver (e.g. ``"DOP853"``, ``"RK45"``).
        num_steps : int, optional
            Number of evaluation points in the interval.
        time_unit : {"seconds", "days", "years"}, optional
            Unit in which ``t_span`` is specified. The values are converted
            internally to seconds.
        rtol : float, optional
            Relative tolerance for the ODE solver.
        atol : float, optional
            Absolute tolerance for the ODE solver.
        max_step : float, optional
            Maximum allowed step size for the solver.
        events : callable or list[callable], optional
            Event functions to be passed to :func:`solve_ivp`.

        Returns
        -------
        scipy.integrate.OdeResult
            The solution object returned by :func:`solve_ivp`.
        """
        conversion_factors = {
            "seconds": 1.0,
            "days": 86400.0,
            "years": 31557600.0,
        }
        factor = conversion_factors.get(time_unit, 1.0)

        t0_sec = t_span[0] * factor
        t1_sec = t_span[1] * factor
        t_eval = np.linspace(t0_sec, t1_sec, num_steps)

        y0 = np.concatenate(
            [
                np.array([b.position for b in self.system.bodies]).flatten(),
                np.array([b.velocity for b in self.system.bodies]).flatten(),
            ]
        )

        solution = solve_ivp(
            fun=self.system.differential_equation,
            t_span=(t0_sec, t1_sec),
            y0=y0,
            method=solver_name,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            events=events,
        )
        return solution


# ==============================================================================
# CLASS 4: Visualizer
# ==============================================================================


class Visualizer:
    """Produce plots and animations from an N-body simulation result.

    Parameters
    ----------
    system : NBodySystem
        The system that was simulated.
    solution :
        The :class:`~scipy.integrate.OdeResult` returned by
        :func:`scipy.integrate.solve_ivp`.
    """

    def __init__(self, system: NBodySystem, solution) -> None:
        self.system = system
        self.solution = solution

        num_steps = len(self.solution.t)
        self.positions = self.solution.y.T[
            :, : self.system.n_bodies * self.system.dims
        ].reshape(num_steps, self.system.n_bodies, self.system.dims)

    def plot_trajectories(self, title: str = "N-Body Simulation") -> None:
        """Create a static plot of all trajectories.

        The method automatically switches between 2D and 3D visualization
        depending on the dimensionality of the system.
        """
        fig = plt.figure(figsize=(9, 9))

        if self.system.dims == 2:
            ax = fig.add_subplot(111)
            for i in range(self.system.n_bodies):
                x = self.positions[:, i, 0]
                y = self.positions[:, i, 1]
                label = self.system.bodies[i].name or f"Body {i + 1}"
                ax.plot(x, y, label=label)
                ax.plot(x[0], y[0], "o", color=ax.lines[-1].get_color())

            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.set_aspect("equal", "box")

        elif self.system.dims == 3:
            ax = fig.add_subplot(111, projection="3d")
            for i in range(self.system.n_bodies):
                x = self.positions[:, i, 0]
                y = self.positions[:, i, 1]
                z = self.positions[:, i, 2]
                label = self.system.bodies[i].name or f"Body {i + 1}"
                ax.plot(x, y, z, label=label)
                ax.plot([x[0]], [y[0]], [z[0]], "o", color=ax.lines[-1].get_color())

            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.set_zlabel("Z position")

        else:
            raise ValueError(
                f"Visualization for {self.system.dims} dimensions is not supported."
            )

        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        plt.show()

    def create_animation(
        self,
        title: str = "N-Body Simulation",
        time_unit: str = "s",
        xlim=None,
        ylim=None,
        zlim=None,
        elev: float | None = None,
        azim: float | None = None,
    ):
        """Create an animated trajectory plot.

        Parameters
        ----------
        title : str, optional
            Title of the animation figure.
        time_unit : str, optional
            Label for the time axis in the annotation.
        xlim, ylim, zlim : tuple[float, float], optional
            Axis limits for the respective dimensions. If ``None``, limits
            are chosen automatically based on the data.
        elev : float, optional
            Elevation angle in the z plane (3D only).
        azim : float, optional
            Azimuth angle in the x,y plane (3D only).

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The created animation object.
        """
        fig = plt.figure(figsize=(9, 9))

        if self.system.dims == 2:
            ax = fig.add_subplot(111)

            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim(
                    self.positions[..., 0].min() * 1.1,
                    self.positions[..., 0].max() * 1.1,
                )

            if ylim is not None:
                ax.set_ylim(ylim)
            else:
                ax.set_ylim(
                    self.positions[..., 1].min() * 1.1,
                    self.positions[..., 1].max() * 1.1,
                )

            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.set_aspect("equal", "box")

            points = [
                ax.plot([], [], "o", label=b.name or f"Body {i + 1}")[0]
                for i, b in enumerate(self.system.bodies)
            ]
            lines = [
                ax.plot([], [], "-", alpha=0.5, color=p.get_color())[0]
                for p in points
            ]
            time_text = ax.text(
                0.05,
                0.95,
                "",
                transform=ax.transAxes,
                verticalalignment="top",
            )

            def update(frame: int):
                for i, (p, l) in enumerate(zip(points, lines)):
                    p.set_data(
                        [self.positions[frame, i, 0]],
                        [self.positions[frame, i, 1]],
                    )
                    l.set_data(
                        self.positions[: frame + 1, i, 0],
                        self.positions[: frame + 1, i, 1],
                    )
                current_time = self.solution.t[frame] / (
                    31557600.0 if time_unit == "years" else 1.0
                )
                time_text.set_text(f"Time: {current_time:.2f} {time_unit}")
                return points + lines + [time_text]

            anim = FuncAnimation(
                fig,
                update,
                frames=len(self.solution.t),
                blit=False,
                interval=20,
            )

        elif self.system.dims == 3:
            ax = fig.add_subplot(111, projection="3d")

            all_pos = self.positions
            mid_x = (all_pos[..., 0].min() + all_pos[..., 0].max()) / 2.0
            mid_y = (all_pos[..., 1].min() + all_pos[..., 1].max()) / 2.0
            mid_z = (all_pos[..., 2].min() + all_pos[..., 2].max()) / 2.0

            max_range = np.max(
                [
                    all_pos[..., 0].max() - all_pos[..., 0].min(),
                    all_pos[..., 1].max() - all_pos[..., 1].min(),
                    all_pos[..., 2].max() - all_pos[..., 2].min(),
                ]
            ) * 1.1
            if max_range == 0:
                max_range = 1.0

            ax.set_xlim(
                xlim if xlim is not None else (mid_x - max_range / 2, mid_x + max_range / 2)
            )
            ax.set_ylim(
                ylim if ylim is not None else (mid_y - max_range / 2, mid_y + max_range / 2)
            )
            ax.set_zlim(
                zlim if zlim is not None else (mid_z - max_range / 2, mid_z + max_range / 2)
            )

            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.set_zlabel("Z position")

            if elev is not None and azim is not None:
                ax.view_init(elev=elev, azim=azim)

            points = [
                ax.plot([], [], [], "o", label=b.name or f"Body {i + 1}")[0]
                for i, b in enumerate(self.system.bodies)
            ]
            lines = [
                ax.plot([], [], [], "-", alpha=0.5, color=p.get_color())[0]
                for p in points
            ]
            time_text = ax.text2D(
                0.05,
                0.95,
                "",
                transform=ax.transAxes,
                verticalalignment="top",
            )

            def update(frame: int):
                for i, (p, l) in enumerate(zip(points, lines)):
                    p.set_data_3d(
                        [self.positions[frame, i, 0]],
                        [self.positions[frame, i, 1]],
                        [self.positions[frame, i, 2]],
                    )
                    l.set_data_3d(
                        self.positions[: frame + 1, i, 0],
                        self.positions[: frame + 1, i, 1],
                        self.positions[: frame + 1, i, 2],
                    )
                current_time = self.solution.t[frame] / (
                    31557600.0 if time_unit == "years" else 1.0
                )
                time_text.set_text(f"Time: {current_time:.2f} {time_unit}")
                return points + lines + [time_text]

            anim = FuncAnimation(
                fig,
                update,
                frames=len(self.solution.t),
                blit=False,
                interval=20,
            )

        else:
            raise ValueError(
                f"Animation for {self.system.dims} dimensions is not supported."
            )

        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        plt.close(fig)
        return anim


# ==============================================================================
# CLASS 5: Analysis
# ==============================================================================


class Analysis:
    """Compute and plot basic conservation laws from a simulation result.

    The analysis is currently implemented for inertial-frame simulations.
    For rotating frames, only a warning is issued and no quantities are
    computed.

    Parameters
    ----------
    system : NBodySystem
        The system that was simulated.
    solution :
        The :class:`~scipy.integrate.OdeResult` returned by
        :func:`scipy.integrate.solve_ivp`.
    """

    def __init__(self, system: NBodySystem, solution) -> None:
        self.system = system
        self.solution = solution
        self.times = solution.t

        num_steps = len(self.times)
        y_transposed = self.solution.y.T

        self.positions = y_transposed[
            :, : self.system.n_bodies * self.system.dims
        ].reshape(num_steps, self.system.n_bodies, self.system.dims)
        self.velocities = y_transposed[
            :, self.system.n_bodies * self.system.dims :
        ].reshape(num_steps, self.system.n_bodies, self.system.dims)

        if self.system.frame == "inertial":
            self._calculate_momentum()
            self._calculate_angular_momentum()
            self._calculate_energy()
            self._calculate_center_of_mass()
        else:
            print(
                "Analysis of conservation laws is only implemented for the "
                "'inertial' frame."
            )

    # -------------------- internal computations ----------------------

    def _calculate_momentum(self) -> None:
        """Compute total linear momentum over time."""
        momenta = self.system.masses * self.velocities
        self.total_momentum = np.sum(momenta, axis=1)

    def _calculate_angular_momentum(self) -> None:
        """Compute total angular momentum over time."""
        momenta = self.system.masses * self.velocities

        if self.system.dims == 2:
            angular_momenta = (
                self.positions[:, :, 0] * momenta[:, :, 1]
                - self.positions[:, :, 1] * momenta[:, :, 0]
            )
            self.total_angular_momentum = np.sum(angular_momenta, axis=1)
        elif self.system.dims == 3:
            angular_momenta = np.cross(self.positions, momenta)
            self.total_angular_momentum = np.sum(angular_momenta, axis=1)

    def _calculate_energy(self) -> None:
        """Compute kinetic, potential, and total energy over time."""
        sq_velocities = np.sum(self.velocities**2, axis=2)
        self.kinetic_energy = 0.5 * np.sum(
            self.system.masses.T * sq_velocities,
            axis=1,
        )

        p1 = np.expand_dims(self.positions, axis=2)
        p2 = np.expand_dims(self.positions, axis=1)
        diffs = p1 - p2
        distances = np.linalg.norm(diffs, axis=3)
        distances[distances == 0] = np.inf

        mass_matrix = self.system.masses @ self.system.masses.T
        potential_energies = (
            -0.5
            * self.system.G
            * np.sum(mass_matrix / distances, axis=(1, 2))
        )
        self.potential_energy = potential_energies
        self.total_energy = self.kinetic_energy + self.potential_energy

    def _calculate_center_of_mass(self) -> None:
        """Compute the center-of-mass position over time."""
        weighted_positions = self.positions * self.system.masses
        total_mass = np.sum(self.system.masses)

        if total_mass > 0.0:
            self.center_of_mass_position = (
                np.sum(weighted_positions, axis=1) / total_mass
            )
        else:
            self.center_of_mass_position = np.zeros_like(
                self.positions[:, 0, :]
            )

    # ---------------------- public plotting API ----------------------

    def plot_energy(self, title: str = "Energy Conservation") -> None:
        """Plot kinetic, potential and total energy and their drift."""
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(10, 8),
            sharex=True,
        )

        ax1.plot(self.times, self.kinetic_energy, label="Kinetic energy (K)")
        ax1.plot(
            self.times,
            self.potential_energy,
            label="Potential energy (U)",
        )
        ax1.plot(
            self.times,
            self.total_energy,
            label="Total energy (E)",
            linewidth=2,
        )
        ax1.set_ylabel("Energy")
        ax1.legend()
        ax1.grid(True)
        ax1.set_title(title)

        energy_drift = (
            self.total_energy - self.total_energy[0]
        ) / self.total_energy[0]
        ax2.plot(self.times, energy_drift)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Relative energy deviation (E - E₀) / E₀")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_momentum(self, title: str = "Momentum Conservation") -> None:
        """Plot the components of the total linear momentum."""
        plt.figure(figsize=(10, 4))

        labels = ["X", "Y", "Z"][: self.system.dims]
        for i, label in enumerate(labels):
            plt.plot(
                self.times,
                self.total_momentum[:, i],
                label=f"Momentum ({label}-component)",
            )

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Total momentum")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_center_of_mass(
        self,
        title: str = "Center-of-mass motion (drift)",
    ) -> None:
        """Plot the drift of the center-of-mass position."""
        plt.figure(figsize=(10, 4))

        initial_position = self.center_of_mass_position[0]
        drift = self.center_of_mass_position - initial_position

        labels = ["X", "Y", "Z"][: self.system.dims]
        for i, label in enumerate(labels):
            plt.plot(
                self.times,
                drift[:, i],
                label=f"Drift ({label}-component)",
            )

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Deviation from starting point")
        plt.grid(True)
        plt.legend()
        plt.show()


# ==============================================================================
# FUNCTIONS for comparing simulations
# ==============================================================================


def plot_full_comparison_dashboard(
    system: NBodySystem,
    sol1_ref,
    sol1_pert,
    sol2_ref,
    sol2_pert,
    label1: str = "Solver 1",
    label2: str = "Solver 2",
) -> None:
    """Create a comparison dashboard for two solvers and their perturbations.

    This helper visualizes the growth of perturbations for two different
    numerical solvers. For each solver, a perturbed and a reference
    simulation are compared.

    Parameters
    ----------
    system : NBodySystem
        Underlying system that was simulated.
    sol1_ref, sol1_pert, sol2_ref, sol2_pert :
        :class:`~scipy.integrate.OdeResult` objects for the two solvers
        (reference and perturbed runs).
    label1 : str, optional
        Label for the first solver.
    label2 : str, optional
        Label for the second solver.
    """
    t_eval = sol1_ref.t
    n_bodies = system.n_bodies
    dims = system.dims

    def get_positions(sol):
        return sol.y[: n_bodies * dims].reshape(
            n_bodies,
            dims,
            len(sol.t),
        )

    pos_pert1 = get_positions(sol1_pert)
    pos_pert2 = get_positions(sol2_pert)
    pos_ref1 = get_positions(sol1_ref)
    pos_ref2 = get_positions(sol2_ref)

    diff1 = np.linalg.norm(pos_pert1 - pos_ref1, axis=1)
    diff2 = np.linalg.norm(pos_pert2 - pos_ref2, axis=1)

    # 2x2 dashboard: trajectories and linear perturbation growth
    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(221, projection="3d")
    for i in range(n_bodies):
        ax1.plot(
            pos_pert1[i, 0],
            pos_pert1[i, 1],
            pos_pert1[i, 2],
            label=system.bodies[i].name,
        )
    ax1.set_title(f"{label1} – perturbed orbit")
    ax1.legend()

    ax2 = fig.add_subplot(222, projection="3d")
    for i in range(n_bodies):
        ax2.plot(
            pos_pert2[i, 0],
            pos_pert2[i, 1],
            pos_pert2[i, 2],
            linestyle="--",
            label=system.bodies[i].name,
        )
    ax2.set_title(f"{label2} – perturbed orbit")
    ax2.legend()

    ax3 = fig.add_subplot(223)
    for i in range(n_bodies):
        ax3.plot(
            t_eval,
            diff1[i],
            ".",
            markersize=0.5,
            label=system.bodies[i].name,
        )
    ax3.set_title(f"Perturbation growth – {label1} (linear)")
    ax3.grid(True)
    ax3.legend()

    ax4 = fig.add_subplot(224)
    for i in range(n_bodies):
        ax4.plot(
            t_eval,
            diff2[i],
            ".",
            markersize=0.5,
            label=system.bodies[i].name,
        )
    ax4.set_title(f"Perturbation growth – {label2} (linear)")
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.show()

    # 1x2 dashboard: logarithmic error growth
    fig2, (log_ax1, log_ax2) = plt.subplots(
        1,
        2,
        figsize=(18, 6),
        sharey=True,
    )

    for i in range(n_bodies):
        log_ax1.plot(
            t_eval,
            np.log10(np.maximum(diff1[i], 1e-20)),
            label=system.bodies[i].name,
        )
    log_ax1.set_title(f"{label1} – logarithmic error progression")
    log_ax1.grid(True)
    log_ax1.legend()

    for i in range(n_bodies):
        log_ax2.plot(
            t_eval,
            np.log10(np.maximum(diff2[i], 1e-20)),
            label=system.bodies[i].name,
        )
    log_ax2.set_title(f"{label2} – logarithmic error progression")
    log_ax2.grid(True)
    log_ax2.legend()

    plt.tight_layout()
    plt.show()
