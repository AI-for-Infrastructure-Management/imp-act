from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patches as patches

from imp_act.environments.recorder import Recorder


class RolloutPlotter:
    def __init__(self, env):

        if isinstance(env, Recorder):
            self.env = env.env
            self.recorded_env = env
        else:
            self.env = env
            self.recorded_env = Recorder(env)

        obs = self.env.reset()
        self.num_components = sum(
            [len(sublist) for sublist in obs["edge_observations"]]
        )
        _shape = self.env.graph.es[0]["road_edge"].segments[0].deterioration_table.shape
        self.num_damage_states = _shape[-1]
        self.num_component_actions = _shape[0]
        self.base_total_travel_time = self.env.base_total_travel_time
        self.initial_edge_volumes = np.array(self.env.graph.es["volume"])
        self.max_timesteps = self.env.max_timesteps

    def flatten(self, lst):
        return list(chain(*lst))

    def _preprocess_episode_data(self, episode_data):
        """
        N: number of components
        T: max timesteps
        S: number of damage states
        E: number of edges
        """

        # stencil for the plotting data
        plot_data = {
            "time_step": None,  # shape: (T,)
            "edge_states": None,  # shape: (T+1, N)
            "edge_observations": None,  # shape: (T+1, N)
            "edge_beliefs": None,  # shape: (T+1, S, N)
            "action": None,  # shape: (T, N)
            "applied_actions": None,  # shape: (T, N)
            "component_failures": np.zeros(
                (self.max_timesteps + 1, self.num_components)
            ),
            "total_travel_time": None,  # shape: (T,)
            "travel_times": None,  # shape: (T, E)
            "reward": None,  # shape: (T,)
            "travel_time_reward": np.zeros(self.max_timesteps),  # shape: (T,)
            "maintenance_reward": np.zeros(self.max_timesteps),  # shape: (T,)
            "budget_remaining": None,  # shape: (T+1,)
            "budget_constraints_applied": None,  # shape: (T,)
            "forced_replace_constraint_applied": None,  # shape: (T,)
            "traffic_volumes": None,  # shape: (T, E)
            "episode_cost": 0,
        }

        for key in plot_data.keys():
            if key in episode_data:
                plot_data[key] = np.array(episode_data[key]).squeeze()

        for t in range(self.max_timesteps):
            _x = episode_data["reward_elements"][t]
            plot_data["maintenance_reward"][t] = _x["maintenance_reward"]
            plot_data["travel_time_reward"][t] = _x["travel_time_reward"]

        for t in range(self.max_timesteps + 1):
            # if damage state is self.num_damage_states, then component has failed
            plot_data["component_failures"][t, :] = plot_data["edge_states"][t, :] == (
                self.num_damage_states - 1
            )

        plot_data["episode_cost"] = -np.sum(plot_data["reward"])

        return plot_data

    def plot(self, episode_data, save_kwargs=None):

        # preprocess plot data for plotting
        data = self._preprocess_episode_data(episode_data)

        self._plot_deterioration(data, save_kwargs=save_kwargs)
        self._plot_travel_time_and_rewards(data, save_kwargs=save_kwargs)
        self._plot_traffic_volume_and_travel_times(data, save_kwargs=save_kwargs)

    def _plot_deterioration(self, plot_data, save_kwargs=None):

        fig, _ax = plt.subplots(6, 2, figsize=(14, 10), sharex=True, sharey=True)

        # ticks and labels: actions
        time_horizon_ticks = np.arange(0, self.max_timesteps + 1, 10)
        action_markers = [".", "s", "<", ">", "^"]
        action_labels = [
            "do-nothing",
            "inspect",
            "minor-repair",
            "major-repair",
            "replace",
        ]
        action_colors = ["gray", "orange", "blue", "dodgerblue", "darkviolet"]
        action_markersize = 5

        for c in range(self.num_components):
            ax = _ax[c // 2, c % 2]

            # state
            (h_true_state,) = ax.plot(
                plot_data["time_step"],
                plot_data["edge_states"][:, c],
                "-",
                label="true state",
                color="tab:green",
                markersize=2,
                alpha=0.5,
            )

            # observation
            (h_obs,) = ax.plot(
                plot_data["time_step"],
                plot_data["edge_observations"][:, c],
                "-o",
                label="observation",
                color="tab:blue",
                markersize=2,
                alpha=0.8,
            )

            # belief
            ax.pcolormesh(
                plot_data["time_step"],
                np.arange(self.num_damage_states),
                plot_data["edge_beliefs"][:, c, :].T,
                shading="nearest",
                cmap="binary",  # _r for reversed
                alpha=0.2,
                vmin=0,
                vmax=1,
                edgecolors="face",
            )

            # draw vertical lines when component fails
            if plot_data["component_failures"][:, c].any():
                for t in np.where(plot_data["component_failures"][:, c])[0]:
                    ax.axvline(t, color="red", linestyle="--", alpha=0.5)

            # Highlight the last timestep with hatching
            last_timestep_start = plot_data["time_step"][-1] - 0.5
            last_timestep_width = (
                plot_data["time_step"][-1] - plot_data["time_step"][-2]
            )
            rect = patches.Rectangle(
                (last_timestep_start, -0.5),  # Lower left corner of the rectangle
                last_timestep_width,  # Width of the rectangle (covers last timestep)
                self.num_damage_states + 1,  # Height of the rectangle
                facecolor="none",  # No fill color
                hatch="\\" * 8,  # Hatching pattern
                edgecolor="black",  # Edge color
                alpha=0.1,  # Transparency
                label="terminal state",
            )
            ax.add_patch(rect)

            ## Plot agent actions
            for a in range(self.num_component_actions):
                _x = np.where(plot_data["applied_actions"][:, c] == a)
                ax.plot(
                    _x,
                    2,
                    action_markers[a],
                    markersize=action_markersize,
                    label=action_labels[a],
                    color=action_colors[a],
                )

            ax.set_xlim([-0.5, self.max_timesteps + 0.5])
            ax.set_ylim([-0.5, self.num_damage_states - 0.5])
            ax.set_xticks(time_horizon_ticks)
            ax.set_yticks(np.arange(self.num_damage_states))
            ax.set_xlabel("time", fontsize=12)
            ax.set_ylabel("damage state", fontsize=8)
            ax.set_title(f"Component {c}", fontsize=12, weight="bold")

            # create legend handles
            legend_handles = []
            for a in range(self.num_component_actions):
                legend_handles += [
                    Line2D(
                        [],
                        [],
                        marker=action_markers[a],
                        markersize=action_markersize,
                        label=action_labels[a],
                        color=action_colors[a],
                        linestyle="None",
                    )
                ]

        legend_handles += [h_true_state, h_obs, rect]
        pcolormesh_proxy = patches.Patch(
            facecolor="gray", alpha=0.2, label="Edge Beliefs"
        )
        legend_handles += [pcolormesh_proxy]
        if plot_data["component_failures"].any():
            legend_handles += [
                Line2D([], [], color="red", linestyle="--", label="unsafe state")
            ]

        # Move the legend outside the plot
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=4,
        )

        fig.suptitle("Deterioration Process", fontsize=14, weight="bold")
        fig.tight_layout()
        plt.show()

        if save_kwargs is not None:
            fig.savefig(**save_kwargs)

    def _plot_travel_time_and_rewards(self, plot_data, save_kwargs=None):

        time = plot_data["time_step"][:-1]

        fig, _ax = plt.subplots(1, 3, figsize=(18, 5))

        # plot total travel time
        ax = _ax[0]
        ax.plot(
            time,
            plot_data["total_travel_time"],
            label="total travel time",
            color="black",
        )
        ax.axhline(
            self.base_total_travel_time,
            linestyle="--",
            color="red",
            label="base travel time",
        )
        ax.set_ylabel("total travel time", fontsize=12)
        ax.set_title("Total Travel Time", fontsize=14)

        # plot rewards
        ax = _ax[1]
        ax.plot(time, plot_data["reward"], label="total reward")
        ax.plot(
            time,
            plot_data["travel_time_reward"],
            label="travel time reward",
        )
        ax.plot(
            time,
            plot_data["maintenance_reward"],
            label="maintenance reward",
        )
        ax.set_ylabel("reward", fontsize=12)
        ax.set_title("Reward Components", fontsize=14)

        for ax in _ax[:-1]:
            ax.set_xlabel("time", fontsize=12)
            ax.set_xlim([-0.5, self.max_timesteps + 0.5])
            ax.set_xticks(np.arange(0, self.max_timesteps + 1, 10))
            ax.grid()
            ax.legend()

        # plot reward pie chart
        ax = _ax[2]
        x1 = -plot_data["travel_time_reward"].sum()
        x2 = -plot_data["maintenance_reward"].sum()
        ax.pie(
            [x1, x2],
            labels=["travel time", "maintenance"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Reward Distribution", fontsize=14)

        plt.show()

        if save_kwargs is not None:
            fig.savefig(**save_kwargs)

    def _plot_traffic_volume_and_travel_times(self, plot_data, save_kwargs=None):

        time = plot_data["time_step"][:-1]

        fig, _ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        # Define labels for edges
        labels = [f"edge {i}" for i in range(self.env.graph.ecount())]

        colors = plt.cm.get_cmap("tab10", self.env.graph.ecount())

        for i, label in enumerate(labels):
            _ax[0].plot(
                time,
                plot_data["traffic_volumes"][:, i],
                label=label,
                color=colors(i),  # Use distinct color for each line
                marker="o",  # Optional: add markers to each line
                linestyle="-",  # Keep lines solid
                linewidth=1.5,  # Thicker lines for better visibility
            )

            _ax[1].plot(
                time,
                plot_data["travel_times"][:, i],
                color=colors(i),  # Use distinct color for each line
                marker="x",  # Optional: add markers to each line
                linestyle="-",  # Keep lines solid
                linewidth=1.5,  # Thicker lines for better visibility
            )

        _ax[0].set_ylabel("Traffic Volume", fontsize=12)
        _ax[1].set_ylabel("Travel Time", fontsize=12)

        for ax in _ax:
            ax.set_xlim([-0.5, self.max_timesteps + 0.5])
            ax.set_xticks(np.arange(0, self.max_timesteps + 1, 1))
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Time (s)", fontsize=12)

        fig.suptitle("Traffic Data", fontsize=14)
        fig.legend(loc="center right", bbox_to_anchor=(1.1, 0.5), fontsize=10)
        fig.tight_layout()

        if save_kwargs:
            fig.savefig(**save_kwargs)

        plt.show()
