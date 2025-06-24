import time

import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger
import fdtdx


def irradiance(E0_0, eta, alpha, z):
    """
    Time-averaged power-flux density ⟨S⟩·k̂ = I(z) of an attenuating plane wave.

    Parameters
    ----------
    E0_0 : complex or float
        Complex electric-field amplitude at z = 0  (peak value, V/m).
    eta : float
        Intrinsic impedance of the propagation medium (Ω).
        •  vacuum → η ≈ 376.730 313 668 Ω
        •  air (20 °C) ≈ 377 Ω
        •  seawater, tissue, etc. → use |√(μ/ε)| at the frequency of interest
    alpha : float
        Power-attenuation coefficient α [Np m⁻¹].
        •  α = 0 → loss-free medium (I independent of z)
        •  α > 0 → exponential decay  e^{-2αz}
    z : array_like or scalar
        Distance(s) from the reference plane (m), measured **along k̂**.

    Returns
    -------
    I : ndarray or scalar
        Irradiance I(z) = |E₀(z)|² / (2 η)  (W m⁻²).
    """
    E0_z = E0_0 * jnp.exp(-alpha * z)          # field amplitude vs. z
    return jnp.square(jnp.abs(E0_z)) / (2.0 * eta)


def poynting_vector(E0_0_vec, eta, alpha, z, k_hat=jnp.array([0.0, 0.0, 1.0])):
    """
    Full time-averaged Poynting vector ⟨S⟩ for an arbitrarily polarized plane wave.

    Parameters
    ----------
    E0_0_vec : array_like, shape (3,)
        Complex electric-field vector amplitude at z = 0 (V/m).
        Must satisfy k̂ · E₀ = 0  for a true plane wave.
    k_hat : array_like, shape (3,), optional
        Unit propagation vector (defaults to +ẑ).

    Returns
    -------
    S_vec : ndarray
        ⟨S⟩(z)(W m⁻²) in 3-vector form, pointing along k̂.
    """
    E0_z_vec = jnp.asarray(E0_0_vec) * jnp.exp(-alpha * z)
    I = jnp.square(jnp.abs(E0_z_vec)).sum(axis=-1) / (2.0 * eta)
    return I[..., None] * jnp.asarray(k_hat)


def main():
    exp_logger = fdtdx.Logger(
        experiment_name="simulate_lossy_material",
    )
    key = jax.random.PRNGKey(seed=42)

    wavelength = 7.4e-3
    period = fdtdx.constants.wavelength_to_period(wavelength)
    frequency = 1 / period
    field_initial_amp = 1.

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
            electric_conductivity=2.0, # In Siemens/meter
        )
    )

    periodic = True
    if periodic:
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="periodic")
    else:
        bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=10, boundary_type="pml")
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
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
        switch=fdtdx.OnOffSwitch(interval=3),
        exact_interpolation=True,
        # if set to positive integer, makes plotting much faster, but can also cause instabilities
        num_video_workers=8,
    )
    constraints.extend(video_energy_detector.same_position_and_size(volume))

    backwards_video_energy_detector = fdtdx.EnergyDetector(
        name="Backwards Energy Video",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(interval=3),
        inverse=True,
        exact_interpolation=True,
        # if set to positive integer, makes plotting much faster, but can also cause instabilities
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
