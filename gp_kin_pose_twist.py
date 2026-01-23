import asyncio
import ctypes
import time
from ctypes import c_bool, c_int32
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import iceoryx2 as iox2
import jax.numpy as jnp
import numpy as np
import tomli_w
import tomllib
from gosselin_platform import GPR9, GPSE3SO23, GPDimension
from jax.flatten_util import ravel_pytree
from jax_dataclasses import pytree_dataclass
from jaxlie import SE3, SO2


# -----------------------------
# Config / data structures
# -----------------------------

@pytree_dataclass(frozen=True)
class GPOrcaInitialPosition:
    """Same as your two scripts."""
    x0: GPSE3SO23
    rho0: GPR9

    @property
    def ravel_1d_array(self):
        return ravel_pytree(self)[0]

    def to_toml(self, path: Path) -> None:
        arr = self.ravel_1d_array.tolist()
        with path.open("wb") as f:
            f.write(tomli_w.dumps({"initial_position": arr}).encode("utf-8"))

    @staticmethod
    def from_toml(path: Path) -> "GPOrcaInitialPosition":
        with path.open("rb") as f:
            data = tomllib.load(f)
        arr = jnp.array(data["initial_position"])
        return GP_ORCA_INIT_POS_UNRAVEL_FN(arr)


GP_ORCA_INIT_POS_UNRAVEL_FN = ravel_pytree(
    GPOrcaInitialPosition(
        x0=GPSE3SO23(SE3.identity(), SO2.identity((3,))), rho0=GPR9(jnp.zeros(9))
    )
)[1]


class MotorCommandData(ctypes.Structure):
    _fields_ = [("position_um", c_int32 * 9)]

    @staticmethod
    def type_name() -> str:
        return "MotorCommandData"


class MotorState(ctypes.Structure):
    _fields_ = [
        ("position_um", c_int32),
        ("force_mn", c_int32),
        ("power_w", ctypes.c_uint16),
        ("temperature_c", ctypes.c_uint8),
        ("voltage_mv", ctypes.c_uint16),
        ("error", ctypes.c_uint16),
    ]

    @staticmethod
    def type_name() -> str:
        return "MotorState"


class MotorStates(ctypes.Structure):
    _fields_ = [("states", MotorState * 9)]

    @staticmethod
    def type_name() -> str:
        return "MotorStates"


class Pose(ctypes.Structure):
    _fields_ = [
        ("qw", ctypes.c_float),
        ("qx", ctypes.c_float),
        ("qy", ctypes.c_float),
        ("qz", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
    ]

    @staticmethod
    def type_name() -> str:
        return "Pose"


class Twist(ctypes.Structure):
    _fields_ = [
        ("vx", ctypes.c_float),
        ("vy", ctypes.c_float),
        ("vz", ctypes.c_float),
        ("wx", ctypes.c_float),
        ("wy", ctypes.c_float),
        ("wz", ctypes.c_float),
    ]

    @staticmethod
    def type_name() -> str:
        return "Twist"


def clip_by_norm(x, max_norm, axis=-1, eps=1e-12):
    x = jnp.asarray(x)
    norms = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    scale = jnp.minimum(1.0, max_norm / (norms + eps))
    return x * scale


@dataclass
class LatestCommand:
    last_pose: Optional[SE3] = None
    last_pose_t: float = -1.0

    last_twist: Optional[np.ndarray] = None  # shape (6,)
    last_twist_t: float = -1.0

# -----------------------------
# Main combined controller
# -----------------------------

async def main():
    # twist mode limits (optional safety)
    CLIP_TWIST_MODE = True

    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore

    rr_connection_client = (
        node.service_builder(iox2.ServiceName.new("orca-motor/connect"))  # type: ignore
        .request_response(c_bool, c_bool)
        .open_or_create()
        .client_builder()
        .create()
    )

    # Connect to motor
    async with asyncio.timeout(5):
        pending_response = rr_connection_client.loan_uninit().write_payload(c_bool(True)).send()
        while True:
            maybe_connected = pending_response.receive()
            if maybe_connected is not None:
                is_motor_connected = bool(maybe_connected.payload().contents)
                print(f"Connected: {is_motor_connected}")
                break
            await asyncio.sleep(100e-3)

    if not is_motor_connected:
        raise RuntimeError("Failed to connect to motor.")

    # Load configuration
    DIMENSION = GPDimension.from_toml(Path("./dimension-capstone-revised.toml"))
    INIT_POS = GPOrcaInitialPosition.from_toml(Path("./initial-position.toml"))
    x = INIT_POS.x0

    # JIT warm up
    print("Warming up JIT...")
    for _ in range(100):
        (_, _loss), x = DIMENSION.damped_newton_step_fn(
            (x, 0.0), INIT_POS.x0.pose, factor=1e-2
        )
    print("JIT warm up done.")

    # IO endpoints
    motor_position_publisher = (
        node.service_builder(iox2.ServiceName.new("orca-motor/position_command_um"))  # type: ignore
        .publish_subscribe(MotorCommandData)
        .open_or_create()
        .publisher_builder()
        .create()
    )

    motor_state_subscriber = (
        node.service_builder(iox2.ServiceName.new("orca-motor/state"))  # type: ignore
        .publish_subscribe(MotorStates)
        .open_or_create()
        .subscriber_builder()
        .create()
    )

    pose_subscriber = (
        node.service_builder(iox2.ServiceName.new("/pose"))  # type: ignore
        .publish_subscribe(Pose)
        .open_or_create()
        .subscriber_builder()
        .create()
    )

    twist_subscriber = (
        node.service_builder(iox2.ServiceName.new("/twist"))  # type: ignore
        .publish_subscribe(Twist)
        .open_or_create()
        .subscriber_builder()
        .create()
    )

    cmd = LatestCommand()

    try:
        # Move to initial position (same logic)
        actuator_commands_m_init = DIMENSION.ik(x).rho - INIT_POS.rho0.rho
        for frac in jnp.linspace(0, 1, int((_init_time := 5) / (init_dt := 25e-3))):
            motor_command_um_frac = actuator_commands_m_init * 1e6 * frac
            motor_position_publisher.loan_uninit().write_payload(
                MotorCommandData(
                    position_um=tuple(motor_command_um_frac.astype(np.int32).tolist())
                )
            ).send()
            await asyncio.sleep(init_dt)

            maybe_states = motor_state_subscriber.receive()
            if maybe_states is not None:
                states: MotorStates = maybe_states.payload().contents
                positions = [states.states[i].position_um for i in range(9)]
                print(f"Motor positions (um): {positions}")


        CONTROL_DT = 0.025
        next_tick = time.perf_counter()
        last_print_t = time.perf_counter()
        PRINT_DT = 0.2

        twist_cmd = np.zeros(6, dtype=np.float64)
        ALPHA = 0.2

        # Control loop
        target_pose = x.pose  # used in pose mode
        last_update_instant = time.perf_counter()

        while True:
            now = time.perf_counter()
            if now < next_tick:
                await asyncio.sleep(next_tick - now)
            now = time.perf_counter()

            dt = CONTROL_DT
            next_tick = now + CONTROL_DT

            # drain latest pose
            maybe_pose = None
            while True:
                temp = pose_subscriber.receive()
                if temp is None:
                    break
                maybe_pose = temp
            if maybe_pose is not None:
                p = maybe_pose.payload().contents
                cmd.last_pose = SE3(jnp.array([p.qw, p.qx, p.qy, p.qz, p.x, p.y, p.z]))
                cmd.last_pose_t = now

            # drain latest twist
            maybe_twist = None
            while True:
                temp = twist_subscriber.receive()
                if temp is None:
                    break
                maybe_twist = temp
            if maybe_twist is not None:
                t = maybe_twist.payload().contents
                cmd.last_twist = np.array([t.vx, t.vy, t.vz, t.wx, t.wy, t.wz], dtype=np.float64)
                cmd.last_twist_t = now

            # mode
            pose_ok = (cmd.last_pose is not None)
            twist_ok = (now - cmd.last_twist_t) <= 0.25

            if twist_ok:
                use_twist, use_pose = True, False
            elif pose_ok:
                use_pose, use_twist = True, False
            else:
                use_pose = use_twist = False

            # compute step
            if use_pose:
                target_pose = cmd.last_pose
                se3_err = (x.pose.inverse() @ target_pose).log()
                twist_est = se3_err / dt

                tlin = clip_by_norm(twist_est[:3], 0.2)
                trot = clip_by_norm(twist_est[3:], jnp.deg2rad(30.0))
                twist_clipped = jnp.concatenate([tlin, trot], axis=0)

                # low-pass filter (optional but recommended)
                twist_np = np.array(twist_clipped, dtype=np.float64)
                twist_cmd = (1 - ALPHA) * twist_cmd + ALPHA * twist_np

                se3_log = twist_cmd * dt

            elif use_twist:
                tw = cmd.last_twist
                # (your clipping here)
                se3_log = (tw * dt).astype(np.float64)
            else:
                se3_log = np.zeros(6, dtype=np.float64)

            (_, _loss), x = DIMENSION.damped_newton_step_fn((x, 0.0), x.pose @ SE3.exp(se3_log), factor=1e-2)

            actuator_commands_m = DIMENSION.ik(x).rho - INIT_POS.rho0.rho
            motor_command_um = (actuator_commands_m * 1e6).astype(int).tolist()
            motor_position_publisher.loan_uninit().write_payload(
                MotorCommandData(position_um=tuple(motor_command_um))
            ).send()

            if (now - last_print_t) >= PRINT_DT:
                last_print_t = now
            # motor state (optional print)
            maybe_states = None
            while True:
                temp = motor_state_subscriber.receive()
                if temp is None:
                    break
                else:
                    maybe_states = temp

            if maybe_states is not None:
                st: MotorStates = maybe_states.payload().contents
                positions = [st.states[i].position_um for i in range(9)]
                mode = "TWIST" if use_twist else ("POSE" if use_pose else "HOLD")
                print(f"[{mode}] Motor positions (um): {positions}")

            # Small sleep to avoid 100% CPU busy loop
            await asyncio.sleep(1e-3)


    finally:
        # Close motor connection
        async with asyncio.timeout(5):
            pending_response = (
                rr_connection_client.loan_uninit().write_payload(c_bool(False)).send()
            )
            while True:
                maybe_connected = pending_response.receive()
                if maybe_connected is not None:
                    print(f"Connected: {maybe_connected.payload().contents}")
                    break
                await asyncio.sleep(100e-3)


if __name__ == "__main__":
    asyncio.run(main())
