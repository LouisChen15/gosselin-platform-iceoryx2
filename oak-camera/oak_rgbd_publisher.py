import ctypes
import time

import depthai as dai
import iceoryx2 as iox2
from numpy.typing import NDArray
from oak_camera_const import (
    RESOLUTION,
    OakPointCloudMessage,
    OakRGBDMessage,
)

if __name__ == "__main__":
    # Create pipeline
    with dai.Pipeline() as p:
        color_cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left_cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right_cam = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = p.create(dai.node.StereoDepth).build(
            left=left_cam.requestOutput(RESOLUTION),
            right=right_cam.requestOutput(RESOLUTION),
        )
        rgbd = p.create(dai.node.RGBD).build()

        # These settings are copied from depthai examples
        stereo.setRectifyEdgeFillColor(0)
        stereo.enableDistortionCorrection(True)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.initialConfig.postProcessing.thresholdFilter.maxRange = 10000
        rgbd.setDepthUnits(dai.StereoDepthConfig.AlgorithmControl.DepthUnit.METER)

        color_out = color_cam.requestOutput(
            RESOLUTION, dai.ImgFrame.Type.RGB888i, dai.ImgResizeMode.CROP, fps=25
        )
        color_out.link(stereo.inputAlignTo)
        color_out.link(rgbd.inColor)

        stereo.depth.link(rgbd.inDepth)

        rgbd_queue = rgbd.rgbd.createOutputQueue()
        pcl_queue = rgbd.pcl.createOutputQueue()

        p.start()

        node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore
        oak_cam_rgbd_publisher = (
            node.service_builder(iox2.ServiceName.new("oak-camera/rgbd"))  # type: ignore
            .publish_subscribe(OakRGBDMessage)
            .open_or_create()
            .publisher_builder()
            .create()
        )
        oak_cam_point_cloud_publisher = (
            node.service_builder(iox2.ServiceName.new("oak-camera/pcl"))  # type: ignore
            .publish_subscribe(OakPointCloudMessage)
            .open_or_create()
            .publisher_builder()
            .create()
        )

        while p.isRunning():
            rgbd_data: dai.RGBDData
            for rgbd_data in rgbd_queue.getAll():  # type: ignore
                print(time.perf_counter())
                rgb_image_frame = rgbd_data.getRGBFrame()
                # print(rgb_image_frame.getType())
                rgb_888i_mat = rgb_image_frame.getFrame()
                # print(rgb_888i_mat.shape)

                depth_image_frame = rgbd_data.getDepthFrame()
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

            pcl_data: dai.PointCloudData
            for pcl_data in pcl_queue.getAll():  # type: ignore
                points: NDArray
                rgba: NDArray
                points, rgba = pcl_data.getPointsRGB()
                assert points.flags.c_contiguous
                assert rgba.flags.c_contiguous
                request_uninit = oak_cam_point_cloud_publisher.loan_uninit()
                payload_ptr = request_uninit.payload()
                # Copy data into the payload
                ctypes.memmove(
                    ctypes.addressof(payload_ptr.contents.xyz),
                    points.ctypes.data,
                    points.nbytes,
                )
                ctypes.memmove(
                    ctypes.addressof(payload_ptr.contents.rgba),
                    rgba.ctypes.data,
                    rgba.nbytes,
                )
                # Publish the data
                request_uninit.assume_init().send()
