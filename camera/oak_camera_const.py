import ctypes

RESOLUTION = (640, 480)


class OakPointCloudMessage(ctypes.Structure):
    _fields_ = [
        ("xyz", ctypes.c_float * (RESOLUTION[0] * RESOLUTION[1] * 3)),
        ("rgba", ctypes.c_uint8 * (RESOLUTION[0] * RESOLUTION[1] * 4)),
    ]


class OakRGBDMessage(ctypes.Structure):
    _fields_ = [
        (
            "rgb_data_uint8_flat_array",
            ctypes.c_uint8 * (RESOLUTION[0] * RESOLUTION[1] * 3),
        ),
        (
            "depth_data_uint16_flat_array",
            ctypes.c_uint16 * (RESOLUTION[0] * RESOLUTION[1]),
        ),
    ]

    @staticmethod
    def type_name() -> str:
        return "OAKCameraData"
