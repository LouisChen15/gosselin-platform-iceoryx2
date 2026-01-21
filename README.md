# Gosselin Platform on iceoryx2

Real‑time control stack for a 9‑actuator Gosselin platform using zero‑copy IPC with [iceoryx2](https://iceoryx.io/). The stack consists of:

- `orca-motor-server` (Rust): talks to 9 ORCA motors over serial, exposes connection control, consumes position commands, and publishes motor states.
- `gamepad-twist-publisher` (Rust): reads a gamepad and publishes either Twist or preset Pose commands for tele‑operation.
- `gp_kin_twist.py` / `gp_kin_pose.py` (Python): Kinematics controllers using the `gosselin-platform` library (JAX) that convert Twist/Pose into actuator position commands.

All components communicate over iceoryx2 publish/subscribe and request/response services.

## Prerequisites

- Install [pixi](https://pixi.sh/latest/) (the only manual dependency). Pixi installs Rust, Python, and the Python packages (including `iceoryx2` and `gosselin-platform`).

## Install

```bash
pixi install --all
```

> **Note for OAK-Camera (depthai python package):**
> On Windows, the `depthai` python wheel seems to have incorrect RECORD file. Currently, we have no workaround. If you need OAK-D support, consider using Linux or macOS, or manually resolve the `depthai` Python package issue.
> To install the rest of the packages without `depthai`, comment out the line with `depthai` in `oak-camera/pyproject.toml` before running `pixi install --all`.

## Configure

- Serial devices: edit [orca-serial-config.toml](orca-motor-server/orca-serial-config.toml) to match your system. Paths are validated at runtime.
- Platform geometry: [dimension-capstone-revised.toml](dimension-capstone-revised.toml) defines the mechanism dimensions used by the kinematics.
- Initial position: [initial-position.toml](initial-position.toml) defines the home pose and nominal prismatic lengths used by the controllers.

### Zenoh Configuration (optional, for distributed systems)

This is only for running over distributed systems. We have verified that Zenoh in client mode can run in UBC wireless network.

On one machine whose IP address is reachable by all participants:

- Install [zenohd](https://zenoh.io/docs/getting-started/installation/#installing-the-zenoh-router)
- Run `zenohd`
- Run `iox2 tunnel zenoh`

On each participant machine:

- Edit endpoint(s) in [zenoh-client-config.json](./zenoh-client-config.json) to include the IP address of the `zenohd` router server.
- Run `iox2 tunnel zenoh -z ./zenoh-client-config.json` to enable Zenoh communication.

> For `iox2` [installation](https://ekxide.github.io/iceoryx2-book/main/getting-started/robot-nervous-system/command-line-tools-and-debugging.html), you need rust toolchain, which is already provided in the pixi default environment. `cargo` command is available from `pixi shell`. Follow the instructions after `cargo` installed `iox2` to add it to your `PATH`.
>
> `clang` is also required to build `iox2` cli tool:
>
> ```bash
> # On Ubuntu/Debian
> sudo apt install clang
> ```
>
> ```bash
> # On Windows
> winget install llvm
> ```

## Run

`pixi run` to see available tasks.

### ORCA Motor Server

```bash
pixi run orca-motor-server
```

### Gamepad Twist/Pose Publisher

```bash
pixi run gamepad-twist-publisher
```

### Keyboard Twist Publisher

```bash
pixi run keyboard-twist-publisher
```

### Kinematics Controller (choose one)

Twist controller:

```bash
pixi run gp-kin-twist
```

Pose controller:

```bash
pixi run gp-kin-pose
```

Combined controller:

```bash
pixi run gp-kin-pose-twist
```

### MuJoCo simulation with twist controller:

```bash
pixi run -e mujoco-sim gp-kin-twist-mujoco-sim
```

Leave the server and publisher running; start/stop the controller as needed.

### OAK-D RGB‑D Camera Publisher and Subscriber

```bash
pixi run -e oak-camera oak-rgbd-publisher
```

```bash
pixi run -e oak-camera oak-rgbd-subscriber
```

```bash
pixi run -e oak-camera oak-pcl-subscriber
```

## Controls

From `gamepad-twist-publisher/src/main.rs`:

- Twist (topic `/twist`):

  - `vx, vy` ← left stick X/Y
  - `vz` ← right stick Y
  - `wx` ← D‑pad Down − Up
  - `wy` ← D‑pad Right − Left
  - `wz` ← − right stick X

- Pose presets (topic `/pose`):
  - South (A/Cross): set z = 0.30 m
  - North (Y/Triangle): set z = 0.40 m
  - East (B/Circle): set x = +0.10 m, z = 0.35 m
  - West (X/Square): set x = −0.10 m, z = 0.35 m

The Python controllers:

- `gp_kin_twist.py` subscribes to `/twist` and integrates the commanded twist into an SE3 target.
- `gp_kin_pose.py` subscribes to `/pose` and tracks a target SE3 pose with rate limits.
- `gp_kin_pose.py` subscribes to both `/pose` and `/twist` and run whatever command that comes earlier.

Both compute inverse kinematics and publish `MotorCommandData` in micrometers to the motor server.

## IPC Services

The system uses these iceoryx2 services (see the Rust/Python sources for struct layouts):

- Publish/Subscribe Messages
  - `/twist` → `Twist` (Rust struct; Python `ctypes.Structure`)
  - `/pose` → `Pose`
  - `orca-motor/position_command_um` → `MotorCommandData` (controller → server)
  - `orca-motor/state` → `MotorStates` (server → any listener)
  - `oak-camera/rgbd` → [`OakRGBDMessage`](./oak-camera/oak_camera_const.py)
  - `oak-camera/pcl` → [`OakPointCloudMessage`](./oak-camera/oak_camera_const.py)
- Request/Response
  - `orca-motor/connect` → `bool` request to (dis)connect motors; `bool` response indicates current connection state
- Event
  - `orca-motor/error_event` → notifies when any motor reports a non‑zero error code

## Notes and Tips

- Always run inside `pixi shell`; it provides Rust and Python toolchains and packages.
- First runs of the Python controllers perform JAX JIT warm‑up and may take a few seconds.
- On Linux/macOS, serial devices look like `/dev/tty*`; on Windows, use `COM*` in `orca-serial-config.toml`.
- Safety: verify a clear workspace before sending commands; the controllers move real hardware.
