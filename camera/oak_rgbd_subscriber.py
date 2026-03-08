import ctypes
import time

import cv2
import iceoryx2 as iox2
import numpy as np
from oak_camera_const import (
    RESOLUTION,
    OakRGBDMessage,
)

if __name__ == "__main__":
    iox2_node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore

    oak_cam_subscriber = (
        iox2_node.service_builder(iox2.ServiceName.new("oak-camera/rgbd"))  # type: ignore
        .publish_subscribe(OakRGBDMessage)
        .open_or_create()
        .subscriber_builder()
        .create()
    )

    while True:
        maybe_data = oak_cam_subscriber.receive()
        if maybe_data is None:
            time.sleep(0.001)
            continue

        payload_ptr = maybe_data.payload()

        rgb_888i_mat_zero_copy = np.ctypeslib.as_array(
            payload_ptr.contents.rgb_data_uint8_flat_array
        ).reshape((RESOLUTION[1], RESOLUTION[0], 3))
        depth_raw16_mat_zero_copy = np.ctypeslib.as_array(
            payload_ptr.contents.depth_data_uint16_flat_array
        ).reshape((RESOLUTION[1], RESOLUTION[0]))

        assert rgb_888i_mat_zero_copy.ctypes.data == ctypes.addressof(
            payload_ptr.contents.rgb_data_uint8_flat_array
        )
        assert depth_raw16_mat_zero_copy.ctypes.data == ctypes.addressof(
            payload_ptr.contents.depth_data_uint16_flat_array
        )

        cv2.cvtColor(
            rgb_888i_mat_zero_copy, cv2.COLOR_RGB2BGR, rgb_888i_mat_zero_copy
        )  # adapt for openCV BGR display
        cv2.imshow("OAK Camera RGB Recovered", rgb_888i_mat_zero_copy)
        cv2.imshow("OAK Camera Depth Recovered", depth_raw16_mat_zero_copy)
        if cv2.waitKey(1) == ord("q"):
            break
