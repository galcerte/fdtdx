import time

import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger
import fdtdx


def irradiance(
    E0_0,
    freq,
    electric_conductivity,
    eps_r,
    mu_r,
    z,
):
    """
    Time-averaged power-flux density ⟨S⟩·k̂ = I(z) of an attenuating plane wave.

    Parameters
    ----------
    E0_0 : complex or float
        Complex electric-field amplitude at z = 0 (peak value, V/m).
    freq : float
        Frequency of plane wave
    electric_conductivity : array_like or float
        Conductivity as used in the vector form of Ohm's law. [S/m].
    eps_r : float, optional
        Relative permittivity ε_r (default 1.0).
    mu_r : float, optional
        Relative permeability μ_r (default 1.0).
    z : array_like or scalar
        Distance(s) from the reference plane (m), measured **along k̂**.

    Returns
    -------
    I : ndarray or scalar
        Irradiance I(z) = |E₀(z)|² / (2 η)  (W m⁻²).
    """
    omega = 2.0 * jnp.pi * freq
    eps = fdtdx.constants.eps0 * eps_r
    mu = fdtdx.constants.mu0 * mu_r
    eta = jnp.sqrt(mu / eps)
    ratio = electric_conductivity / (omega * eps)

    root = jnp.sqrt(1.0 + ratio**2)
    alpha = omega * jnp.sqrt(mu * eps / 2.0) * jnp.sqrt(root - 1.0)
    E0_z = E0_0 * jnp.exp(-alpha * z)  # field amplitude vs. z
    return jnp.square(jnp.abs(E0_z)) / (2.0 * eta)


def main():
    exp_logger = fdtdx.Logger(
        experiment_name="simulate_lossy_material",
    )
    key = jax.random.PRNGKey(seed=42)

    # Field/wave properties
    wavelength = 7.4e-3
    period = fdtdx.constants.wavelength_to_period(wavelength)
    frequency = 1 / period
    field_initial_amp = 1.0
    # Material properties
    # Relative permittivity (real)
    eps_r = 2.
    # Relative permeability (real)
    mu_r = 1.
    # Conductivity (real)
    sigma = 5.

    config = fdtdx.SimulationConfig(
        time=100e-11,
        resolution=1e-4,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    gradient_config = fdtdx.GradientConfig(
        recorder=fdtdx.Recorder(
            modules=[
                fdtdx.DtypeConversion(dtype=jnp.bfloat16),
            ]
        )
    )
    config = config.aset("gradient_config", gradient_config)

    constraints = []

    volume = fdtdx.SimulationVolume(
        partial_real_shape=(12.0e-3, 12e-3, 48e-3),
        material=fdtdx.Material(  # Background material
            permittivity=eps_r,
            permeability=mu_r,
            electric_conductivity=sigma,
        ),
    )
    # Set boundary conditions
    # Periodic boundary conditions on min X, max X, min Y and max Y
    # Perfectly Matched Layer (PML) on min Z and max Z.
    thickness = 10  # In grid units
    kappa_start = 1.0
    kappa_end = 1.5
    bound_cfg = fdtdx.BoundaryConfig(
        boundary_type_minx="periodic",
        boundary_type_maxx="periodic",
        boundary_type_miny="periodic",
        boundary_type_maxy="periodic",
        boundary_type_minz="pml",
        boundary_type_maxz="pml",
        thickness_grid_minx=thickness,
        thickness_grid_maxx=thickness,
        thickness_grid_miny=thickness,
        thickness_grid_maxy=thickness,
        thickness_grid_minz=thickness,
        thickness_grid_maxz=thickness,
        kappa_start_minx=kappa_start,
        kappa_end_minx=kappa_end,
        kappa_start_maxx=kappa_start,
        kappa_end_maxx=kappa_end,
        kappa_start_miny=kappa_start,
        kappa_end_miny=kappa_end,
        kappa_start_maxy=kappa_start,
        kappa_end_maxy=kappa_end,
        kappa_start_minz=kappa_start,
        kappa_end_minz=kappa_end,
        kappa_start_maxz=kappa_start,
        kappa_end_maxz=kappa_end,
    )
    _, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    constraints.extend(c_list)

    source = fdtdx.UniformPlaneSource(
        amplitude=field_initial_amp,
        partial_grid_shape=(None, None, 1),
        partial_real_shape=(None, None, None),
        fixed_E_polarization_vector=(1, 0, 0),
        # partial_grid_shape=(1, None, None),
        # partial_real_shape=(None, 10e-6, 10e-6),
        # fixed_E_polarization_vector=(0, 1, 0),
        # partial_grid_shape=(None, 1, None),
        # partial_real_shape=(10e-6, None, 10e-6),
        # fixed_E_polarization_vector=(1, 0, 0),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        direction="-",
    )
    constraints.extend(
        [
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(0, 0, 0),
                other_positions=(0, 0, 1),
                grid_margins=(0, 0, -(bound_cfg.thickness_grid_maxz + 4)),
            ),
        ]
    )

    video_energy_detector = fdtdx.EnergyDetector(
        name="Energy Video",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(interval=20),
        # if set to positive integer, makes plotting much faster, but can also
        # cause instabilities
        num_video_workers=8,
    )
    constraints.extend(video_energy_detector.same_position_and_size(volume))

    backwards_video_energy_detector = fdtdx.EnergyDetector(
        name="Backwards Energy Video",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(interval=20),
        inverse=True,
        # if set to positive integer, makes plotting much faster, but can also
        # cause instabilities
        num_video_workers=8,
    )
    constraints.extend(backwards_video_energy_detector.same_position_and_size(volume))

    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        volume=volume,
        config=config,
        constraints=constraints,
        key=subkey,
    )
    logger.info(tc.tree_summary(arrays, depth=1))
    print(tc.tree_diagram(config, depth=4))

    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        fdtdx.plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=[
                backwards_video_energy_detector,
                video_energy_detector,
            ],
        ),
    )

    def sim_fn(
        params: fdtdx.ParameterContainer,
        arrays: fdtdx.ArrayContainer,
        key: jax.Array,
    ):
        arrays, new_objects, info = fdtdx.apply_params(arrays, objects, params, key)

        final_state = fdtdx.run_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        _, arrays = final_state

        _, arrays = fdtdx.full_backward(
            state=final_state,
            objects=new_objects,
            config=config,
            key=key,
            record_detectors=True,
            reset_fields=True,
        )

        new_info = {
            **info,
        }
        return arrays, new_info

    compile_start_time = time.time()
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    jitted_loss = jax.jit(sim_fn, donate_argnames=["arrays"]).lower(params, arrays, key).compile()
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)

    run_start_time = time.time()
    key, subkey = jax.random.split(key)
    arrays, info = jitted_loss(params, arrays, subkey)

    runtime_delta = time.time() - run_start_time
    info["runtime"] = runtime_delta
    info["compile time"] = compile_delta_time

    logger.info(f"{compile_delta_time=}")
    logger.info(f"{runtime_delta=}")

    # videos
    exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)


if __name__ == "__main__":
    main()
