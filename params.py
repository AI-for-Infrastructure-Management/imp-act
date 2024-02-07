import jax.numpy as jnp
from flax import struct


# Environment parameters
@struct.dataclass
class EnvParams:
    # Horizon parameters
    max_timesteps: int = 50
    discount_factor: float = 1.0

    # Reward parameters
    travel_time_reward_factor: float = -0.01

    # Graph parameters
    num_vertices: int = 4
    edges: jnp.array = jnp.array([(0, 1), (1, 3), (2, 0), (3, 2)])
    num_edges: int = 4
    edge_segments_numbers: jnp.array = jnp.array([2] * num_edges)

    shortest_path_max_iterations: int = 500
    traffic_assignment_max_iterations: int = 15
    traffic_assignment_convergence_threshold: float = 0.01
    traffic_assignment_update_weight: float = 0.5
    traffic_alpha: float = 0.15
    traffic_beta: float = 4.0

    # Road Network parameters
    trips: jnp.array = jnp.array(
        [[0, 200, 0, 0], [0, 0, 0, 200], [200, 0, 0, 0], [0, 0, 200, 0]],
        dtype=jnp.int32,
    )

    btt_table: jnp.array = (
        jnp.array(
            [
                [1.00, 1.10, 1.40, 1.60],
                [1.00, 1.10, 1.40, 1.60],
                [1.00, 1.05, 1.15, 1.45],
                [1.50, 1.50, 1.50, 1.50],
            ]
        )
        * 50.0
    )

    capacity_table: jnp.array = (
        jnp.array(
            [
                [1.00, 1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00],
                [0.80, 0.80, 0.80, 0.80],
                [0.50, 0.50, 0.50, 0.50],
            ]
        )
        * 500.0
    )

    # Damage parameters
    num_dam_states: int = 4
    initial_dam_state: int = 0
    initial_obs: int = 0
    initial_belief: jnp.array = jnp.array([1.0, 0.0, 0.0, 0.0])

    deterioration_table: jnp.array = jnp.array(
        [
            [  # Action 0: do-nothing
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 0.9, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # Action 1: inspect
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 0.9, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # Action 2: minor repair
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.7, 0.2, 0.1, 0.0],
            ],
            [  # Action 3: major repair (replacement)
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    observation_table: jnp.array = jnp.array(
        [
            [  # Action 0: do-nothing
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # Action 1: inspect
                [0.8, 0.2, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.0, 0.1, 0.9, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # Action 2: minor repair
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # Action 3: major repair (replacement)
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    rewards_table: jnp.array = jnp.array(
        [
            [0, -1, -20, -150],
            [0, -1, -25, -150],
            [0, -1, -30, -150],
            [0, -1, -40, -150],
        ]
    )
