import ctypes
import time
from datetime import timedelta

import depthai as dai
import iceoryx2 as iox2
from numpy.typing import NDArray
from oak_camera_const import (
    RESOLUTION,
    STEREO_SIZE,
    FPS,
    OakPointCloudMessage,
    OakRGBDMessage,
)

# This file uses another method to align workflow to align the RGB camera images and Depth images.

if __name__ == "__main__":
    # Create pipeline
    pipeline = dai.Pipeline()
    platform = pipeline.getDefaultDevice().getPlatform()

    cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    sync = pipeline.create(dai.node.Sync)

    if platform == dai.Platform.RVC4:
        align = pipeline.create(dai.node.ImageAlign)
    else:
        align = None

    stereo.setExtendedDisparity(True)
    stereo.setLeftRightCheck(True)
    sync.setSyncThreshold(timedelta(seconds=1 / (2 * FPS)))

    rgb_out = cam_rgb.requestOutput(
        size=RESOLUTION,
        fps=FPS,
        enableUndistortion=True,
    )
    left_out = cam_left.requestOutput(size=STEREO_SIZE, fps=FPS)
    right_out = cam_right.requestOutput(size=STEREO_SIZE, fps=FPS)

    rgb_out.link(sync.inputs["rgb"])
    left_out.link(stereo.left)
    right_out.link(stereo.right)

    if platform == dai.Platform.RVC4:
        stereo.depth.link(align.input)
        rgb_out.link(align.inputAlignTo)
        align.outputAligned.link(sync.inputs["depth_aligned"])
    else:
        rgb_out.link(stereo.inputAlignTo)
        stereo.depth.link(sync.inputs["depth_aligned"])

    rgbd_queue = sync.out.createOutputQueue()

    pipeline.start()

    node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore
    oak_cam_rgbd_publisher = (
        node.service_builder(iox2.ServiceName.new("oak-camera/rgbd"))  # type: ignore
        .publish_subscribe(OakRGBDMessage)
        .open_or_create()
        .publisher_builder()
        .create()
    )

    while pipeline.isRunning():
        rgbd_data: dai.RGBDData
        packets = rgbd_queue.getAll()
        rgbd_data = packets[-1]
        print(time.perf_counter())

        rgb_image_frame = rgbd_data["rgb"]
        # print(rgb_image_frame.getType())
        rgb_888i_mat = rgb_image_frame.getCvFrame()
        # print(rgb_888i_mat.shape)

        depth_image_frame = rgbd_data["depth_aligned"]
        # print(depth_image_frame.getType())
        depth_raw16_mat = depth_image_frame.getFrame()
        # print(depth_raw16_mat.shape)

        request_uninit = oak_cam_rgbd_publisher.loan_uninit()
        payload_ptr = request_uninit.payload()

        # Copy data into the payload
        assert rgb_888i_mat.flags.c_contiguous
        ctypes.memmove(
            ctypes.addressof(payload_ptr.contents.rgb_data_uint8_flat_array),
            rgb_888i_mat.ctypes.data,
            rgb_888i_mat.nbytes,
        )
        assert depth_raw16_mat.flags.c_contiguous
        ctypes.memmove(
            ctypes.addressof(payload_ptr.contents.depth_data_uint16_flat_array),
            depth_raw16_mat.ctypes.data,
            depth_raw16_mat.nbytes,
        )

        # Publish the data
        request_uninit.assume_init().send()
