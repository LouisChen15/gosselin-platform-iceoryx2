import ctypes
import time

import iceoryx2 as iox2
import numpy as np
import rerun as rr
from oak_camera_const import (
    RESOLUTION,
    OakPointCloudMessage,
)

if __name__ == "__main__":
    iox2_node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)  # type: ignore

    oak_cam_subscriber = (
        iox2_node.service_builder(iox2.ServiceName.new("oak-camera/pcl"))  # type: ignore
        .publish_subscribe(OakPointCloudMessage)
        .open_or_create()
        .subscriber_builder()
        .create()
    )

    rr.init("", spawn=True)
    rr.log("world", rr.ViewCoordinates.RDF)
    rr.log("world/ground", rr.Boxes3D(half_sizes=[3.0, 3.0, 0.00001]))

    while True:
        maybe_data = oak_cam_subscriber.receive()
        if maybe_data is None:
            time.sleep(0.001)
            continue

        payload_ptr = maybe_data.payload()

        # Demonstrate zero copy recovery of data from the payload
        xyz_mat_zero_copy = np.ctypeslib.as_array(payload_ptr.contents.xyz).reshape(
            (RESOLUTION[1] * RESOLUTION[0], 3)
        )
        rgba_mat_zero_copy = np.ctypeslib.as_array(payload_ptr.contents.rgba).reshape(
            (RESOLUTION[1] * RESOLUTION[0], 4)
        )

        assert xyz_mat_zero_copy.ctypes.data == ctypes.addressof(
            payload_ptr.contents.xyz
        )
        assert rgba_mat_zero_copy.ctypes.data == ctypes.addressof(
            payload_ptr.contents.rgba
        )

        rr.log(
            "world/pcl",
            rr.Points3D(xyz_mat_zero_copy, colors=rgba_mat_zero_copy, radii=[0.01]),
        )
