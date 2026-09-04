from types import SimpleNamespace

import pytest
import torch
from holosoma.envs.wbt import wbt_manager
from holosoma.replay import run_reference_replay


class _PhysicsStepMustNotRun:
    def step(self) -> None:
        raise AssertionError("reference replay must not advance physics")


def test_run_reference_replay_does_not_advance_physics() -> None:
    class FakeEnv:
        def __init__(self) -> None:
            self.simulator = SimpleNamespace(sim=_PhysicsStepMustNotRun())
            self.frames = 0

        def step_visualize_motion(self, actions) -> bool:
            assert actions is None
            self.frames += 1
            return self.frames == 3

    env = FakeEnv()
    run_reference_replay(env)

    assert env.frames == 3


def test_debug_visualization_is_skipped_without_viewer() -> None:
    manager = wbt_manager.WholeBodyTrackingManager.__new__(wbt_manager.WholeBodyTrackingManager)
    manager.viewer = None
    manager.simulator = SimpleNamespace(
        get_simulator_type=lambda: (_ for _ in ()).throw(AssertionError("simulator type must not be queried"))
    )

    manager._draw_debug_vis()


def test_visualize_motion_writes_state_before_single_render(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeMotionCommand:
        def __init__(self) -> None:
            self.motion = SimpleNamespace(fps=torch.tensor([50.0]), has_object=False, time_step_total=10)
            self.time_steps = torch.zeros(1, dtype=torch.long)
            self.root_pos_w = torch.tensor([[1.0, 2.0, 3.0]])
            self.root_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
            self.body_lin_vel_w = torch.tensor([[[0.1, 0.2, 0.3]]])
            self.body_ang_vel_w = torch.tensor([[[0.4, 0.5, 0.6]]])
            self.joint_pos = torch.tensor([[0.7, 0.8]])
            self.joint_vel = torch.tensor([[0.9, 1.0]])

        def step(self) -> None:
            calls.append("motion_step")
            self.time_steps += 1

    class FakeSimulationContext:
        def forward(self) -> None:
            calls.append("forward")

        def render(self) -> None:
            calls.append("render")

        def step(self) -> None:
            raise AssertionError("step_visualize_motion must not advance physics")

    class FakeSimulator:
        def __init__(self) -> None:
            self.dof_pos = torch.zeros(1, 2)
            self.dof_vel = torch.zeros(1, 2)
            self.dof_state = torch.zeros(1, 2, 2)
            self.robot_root_states = torch.zeros(1, 13)
            self.all_root_states = self.robot_root_states
            self.scene = SimpleNamespace(write_data_to_sim=lambda: calls.append("write"))
            self.sim = FakeSimulationContext()

        def set_actor_root_state_tensor(self, env_ids, root_states) -> None:
            calls.append("set_root")

        def set_dof_state_tensor(self, env_ids, dof_states) -> None:
            calls.append("set_dof")

        def refresh_sim_tensors(self) -> None:
            calls.append("refresh")

    motion_command = FakeMotionCommand()
    manager = wbt_manager.WholeBodyTrackingManager.__new__(wbt_manager.WholeBodyTrackingManager)
    manager.command_manager = SimpleNamespace(get_state=lambda _name: motion_command)
    manager.simulator = FakeSimulator()
    manager.num_envs = 1
    manager.device = "cpu"
    manager._draw_debug_vis = lambda: calls.append("draw")

    clock_values = iter((10.0, 10.006))
    sleeps: list[float] = []
    monkeypatch.setattr(wbt_manager.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(wbt_manager.time, "sleep", sleeps.append)

    done = manager.step_visualize_motion(None)

    assert not done
    assert calls == ["motion_step", "set_root", "set_dof", "write", "forward", "refresh", "draw", "render"]
    assert sleeps == [pytest.approx(0.014)]
    torch.testing.assert_close(manager.simulator.dof_pos, motion_command.joint_pos)
    torch.testing.assert_close(manager.simulator.dof_vel, motion_command.joint_vel)
    torch.testing.assert_close(manager.simulator.robot_root_states[:, :3], motion_command.root_pos_w)
    torch.testing.assert_close(manager.simulator.robot_root_states[:, 3:7], motion_command.root_quat_w)
