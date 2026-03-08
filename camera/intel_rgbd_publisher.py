import ctypes
import time
import numpy as np

import pyrealsense2 as rs
import iceoryx2 as iox2
from numpy.typing import NDArray
from oak_camera_const import (
    RESOLUTION,
    OakPointCloudMessage,
    OakRGBDMessage,
)
import cv2

FPS = 25

if __name__ == "__main__":
    # -----------------------------
    # Start Intel RealSense pipeline
    # -----------------------------
    pipeline = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(
        rs.stream.color,
        RESOLUTION[0],
        RESOLUTION[1],
        rs.format.bgr8,
        FPS,
    )
    cfg.enable_stream(
        rs.stream.depth,
        RESOLUTION[0],
        RESOLUTION[1],
        rs.format.z16,
        FPS,
    )

    profile = pipeline.start(cfg)

    # Align depth to color frame
    align = rs.align(rs.stream.color)

    # Optional manual exposure / gain
    try:
        color_sensor = profile.get_device().first_color_sensor()
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.exposure, 400)
        color_sensor.set_option(rs.option.gain, 128)
    except Exception as e:
        print(f"[WARN] Could not set color sensor options: {e}")

    # Optional: print depth scale
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"[INFO] RealSense depth scale: {depth_scale}")
    except Exception as e:
        print(f"[WARN] Could not query depth scale: {e}")

    # Warm-up frames
    for _ in range(10):
        pipeline.wait_for_frames()

    # -----------------------------
    # Create IPC publisher
    # -----------------------------
    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
    intel_cam_rgbd_publisher = (
        node.service_builder(iox2.ServiceName.new("intel-camera/rgbd")) 
        .publish_subscribe(OakRGBDMessage)
        .open_or_create()
        .publisher_builder()
        .create()
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            print(time.perf_counter())

            # RealSense color arrives as BGR
            rgb_bgr = np.asanyarray(color_frame.get_data()).copy()

            # Convert to RGB so it matches your existing subscriber expectations
            rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

            # Raw depth uint16, aligned to color
            depth_raw16 = np.asanyarray(depth_frame.get_data()).copy()

            # Make sure memory is contiguous before memmove
            rgb_rgb = np.ascontiguousarray(rgb_rgb, dtype=np.uint8)
            depth_raw16 = np.ascontiguousarray(depth_raw16, dtype=np.uint16)

            request_uninit = intel_cam_rgbd_publisher.loan_uninit()
            payload_ptr = request_uninit.payload()

            # Copy RGB data into payload
            ctypes.memmove(
                ctypes.addressof(payload_ptr.contents.rgb_data_uint8_flat_array),
                rgb_rgb.ctypes.data,
                rgb_rgb.nbytes,
            )

            # Copy depth data into payload
            ctypes.memmove(
                ctypes.addressof(payload_ptr.contents.depth_data_uint16_flat_array),
                depth_raw16.ctypes.data,
                depth_raw16.nbytes,
            )

            # Publish
            request_uninit.assume_init().send()

    finally:
        pipeline.stop()