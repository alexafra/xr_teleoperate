# Reproduce the RGB-D image server on a new robot

This procedure installs the same TeleImager RGB-D server used on the original
robot. It deliberately pins the known-good server and preserves its camera
contract. For a replacement RealSense unit, change only the configured serial
number unless the validation below finds a real hardware/calibration mismatch.

## Pinned source

- Repository: `https://github.com/alexafra/teleimager.git`
- Tag: `teleimager-rgbd-v1.1-2026-08-06`
- Commit: `bdf7dc959b743b5fb7c83796880827c301d4861f`

Do not install from TeleImager's default branch: it does not contain the pinned
RGB, aligned-depth, and raw-depth server used by this project.

On the new robot, make a fresh clone so an existing checkout cannot be
overwritten accidentally:

```bash
git clone \
  --branch teleimager-rgbd-v1.1-2026-08-06 \
  --single-branch \
  https://github.com/alexafra/teleimager.git \
  "$HOME/teleimager-rgbd-v1.1"
cd "$HOME/teleimager-rgbd-v1.1"
test "$(git rev-parse HEAD)" = bdf7dc959b743b5fb7c83796880827c301d4861f
```

Cloning this repository's `research/g1-data-collection` branch with
`--recurse-submodules` is also valid: its TeleImager gitlink resolves to the
same commit. The direct clone above is preferable when PC2 only needs the image
server.

## Preserve the original environment when it is still accessible

The repository pins the server source, but it has no dependency lockfile. If
the old PC2 is still readable, capture its service and Python environment before
recreating anything:

```bash
sudo systemctl cat teleimager.service
python -m pip freeze --all > "$HOME/teleimager-old-pip-freeze.txt"
```

Use the interpreter named by the old service for the second command. Keep these
files with the migration notes; do not change the old robot while collecting
them.

## Environment

The pinned package requires Python 3.8--3.10. Match the original deployment
with Python 3.10:

```bash
conda create -n teleimager python=3.10 -y
conda activate teleimager
sudo apt-get update
sudo apt-get install -y libusb-1.0-0-dev libturbojpeg-dev
python -m pip install -e ".[server]"
python -m pip install psutil
```

`psutil` and `pyrealsense2` are used by the server but are not declared by the
pinned package metadata. Before installing or rebuilding librealsense, first
check whether the robot image already provides a compatible Python binding:

```bash
python -c 'import psutil, pyrealsense2 as rs; print("pyrealsense2", getattr(rs, "__version__", "unknown"), rs.__file__)'
```

On Jetson/aarch64, do not blindly run `pip install --upgrade pyrealsense2`,
upgrade camera firmware, or replace librealsense. Reuse the robot image's
working binding when possible. If it is absent, build the version documented by
the pinned server's aarch64 error guidance (`librealsense` tag `v2.50.0`)
against this exact Python 3.10 environment, then repeat the import check.

The full `.[server]` install is required even though WebRTC is disabled: the
image-server module imports those dependencies at startup.

Before touching the camera, verify every runtime import in the selected
interpreter:

```bash
python - <<'PY'
modules = (
    "pyrealsense2", "psutil", "cv2", "numpy", "yaml", "zmq",
    "logging_mp", "aiohttp", "aiortc", "av", "uvc",
)
for name in modules:
    module = __import__(name)
    print("OK", name, getattr(module, "__file__", "built-in"))
PY
```

## Identify the replacement RealSense without starting a stream

Stop any camera consumer first. Prefer the SDK query below over
`teleimager-server --cf --rs`: TeleImager's discovery path reloads the
`uvcvideo` kernel module.

For a compact identity check, if the librealsense command-line utility is
installed:

```bash
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  rs-enumerate-devices -s
else
  echo "rs-enumerate-devices is not installed; use the Python SDK query below"
fi
```

For the exact serial, firmware, depth scale, required stream profiles, and
color intrinsics:

```bash
python - <<'PY'
import pyrealsense2 as rs

ctx = rs.context()
devices = list(ctx.query_devices())
if not devices:
    raise SystemExit("No RealSense device found")

for index, dev in enumerate(devices):
    def info(key):
        return dev.get_info(key) if dev.supports(key) else "unsupported"

    print(f"device[{index}]")
    print("  name:", info(rs.camera_info.name))
    print("  serial:", info(rs.camera_info.serial_number))
    print("  firmware:", info(rs.camera_info.firmware_version))
    print("  recommended_firmware:", info(rs.camera_info.recommended_firmware_version))
    print("  product_line:", info(rs.camera_info.product_line))
    print("  product_id:", info(rs.camera_info.product_id))
    print("  usb_type:", info(rs.camera_info.usb_type_descriptor))
    print("  depth_scale_m_per_unit:", dev.first_depth_sensor().get_depth_scale())

    wanted = {
        (rs.stream.color, rs.format.bgr8): None,
        (rs.stream.depth, rs.format.z16): None,
    }
    for sensor in dev.query_sensors():
        for profile in sensor.get_stream_profiles():
            key = (profile.stream_type(), profile.format())
            if key not in wanted or profile.fps() != 30:
                continue
            video = profile.as_video_stream_profile()
            if video.width() == 640 and video.height() == 480:
                wanted[key] = video

    for (stream, fmt), video in wanted.items():
        label = f"{stream}/{fmt} 640x480@30"
        print(" ", label, "SUPPORTED" if video is not None else "MISSING")
        if video is not None:
            intr = video.get_intrinsics()
            print("    intrinsics:", {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "cx_ppx": intr.ppx,
                "cy_ppy": intr.ppy,
                "model": str(intr.model),
                "coeffs": list(intr.coeffs),
            })
PY
```

If `rs-enumerate-devices` is available, save its full calibration record as
well:

```bash
if command -v rs-enumerate-devices >/dev/null 2>&1; then
  rs-enumerate-devices > "$HOME/realsense_new_camera_calibration.txt"
fi
```

Require the intended camera to be uniquely identifiable and require both
`BGR8 640x480@30` and `Z16 640x480@30`. Record, but do not automatically apply,
the SDK's recommended firmware version.

## Change only the serial in the server configuration

Edit `cam_config_server.yaml` and replace the old value of
`head_camera.serial_number` (`242322076480`) with the new serial. Keep it quoted.

Then inspect the diff:

```bash
git diff -- cam_config_server.yaml
```

The intended server contract remains:

| Setting | Required value |
| --- | --- |
| Camera type | `realsense` |
| Image shape | `[480, 640]` (height, width) |
| Rate | `30` fps |
| RGB ZMQ | enabled, port `5555` |
| Aligned uint16 depth ZMQ | enabled, port `5558` |
| Raw uint16 depth ZMQ | enabled, port `5559` |
| Config reply | port `60000` (server default) |
| WebRTC | disabled |
| Wrist cameras | disabled |

The full image server dynamically adds `depth_scale_m_per_unit` from the
connected depth sensor to the configuration returned to clients. Do not use
`realsense_zmq_publisher.py`; that helper is RGB-only and would not reproduce
the recording contract above.

## Foreground validation before autostart

The server startup itself attempts to reload `uvcvideo`, so ensure no other
camera process is running and obtain a sudo ticket before the foreground test.
Do not run `setup_uvc.sh` pre-emptively: RealSense also uses `uvcvideo`, but the
script makes persistent udev, group-membership, and passwordless-sudo changes,
while the installed service runs as root. If the foreground test reports an
actual device-permission or `sudo modprobe` failure, review the pinned script
before choosing to apply those changes. Then run the pinned server in the
foreground:

```bash
sudo -v
conda activate teleimager
cd "$HOME/teleimager-rgbd-v1.1"
teleimager-server --rs
```

Confirm that it initializes the selected serial at 640x480/30 and reports a
positive depth scale. From the workstation, confirm that the configuration is
received on port 60000 and that RGB, aligned depth, and raw depth all advance on
ports 5555, 5558, and 5559 before enabling boot startup.

The recent original-robot baseline was 683 RGB, 683 aligned-depth, and 683
raw-depth frames at 29.85 Hz, with a depth scale near 0.001 metres per raw unit.
Treat missing/unequal streams or a materially lower rate as a failed migration,
not as an acceptable camera-ID difference.

Keep the original PC2 address (`192.168.123.164`) if the rest of the deployment
is meant to remain unchanged. If the address changes, it is a second deployment
change and every `--img-server-ip`/`--image-host` invocation must be updated.

## Enable the existing boot service last

After the foreground smoke test passes:

```bash
bash setup_autostart.sh
```

Choose the `teleimager` Conda environment and answer yes for RealSense. The
existing script installs and restarts `teleimager.service`; it runs the same
checkout with `--rs` and `Restart=always`.

Verify it after a reboot:

```bash
sudo systemctl status teleimager.service --no-pager
sudo journalctl -u teleimager.service -b --no-pager
```

Do not enable the service until the foreground run is clean. A restart loop can
otherwise repeatedly reload the camera driver and obscure the actual setup
failure.

## Calibration boundary for surface-normal policies

Changing only the serial is sufficient for the existing RGB, aligned-depth,
and raw-depth transport. It does not make a physically different camera's
optics identical. The current surface-normal preprocessing uses the original
D435I color intrinsics (`fx=605.4215`, `fy=605.5905`, `ppx=321.8568`,
`ppy=242.2497`) at 640x480.

Compare the new intrinsics printed above before collecting or deploying a
surface-normal policy. If they differ materially, keep the server migration
unchanged and update the shared offline/live surface-normal calibration in a
separate reviewed change. Do not silently retrain or deploy an old
surface-normal checkpoint under a different camera calibration.
