"""
Intel RealSense depth camera wrapper.

Encapsulates pipeline configuration, stream alignment, frame retrieval,
and exposure of color stream intrinsics for 3D deprojection.
"""

import numpy as np
import pyrealsense2 as rs


class DepthCamera:
    """
    Wraps the Intel RealSense pipeline for aligned RGB-D capture.

    Streams:
        - Depth: 640x480 @ 30 fps (Z16)
        - Color: 1920x1080 @ 30 fps (BGR8)

    Depth frames are aligned to the color frame so that each color pixel
    maps to a valid depth value, enabling accurate deprojection.
    """

    def __init__(self) -> None:
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        config.resolve(pipeline_wrapper)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # Store color stream intrinsics for rs2_deproject_pixel_to_point
        color_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().intrinsics

    def get_frame(self) -> tuple[bool, np.ndarray | None, rs.depth_frame | None]:
        """
        Wait for the next aligned frame pair.

        Returns:
            (success, color_image, depth_frame)
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None, None

        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image, depth_frame

    def deproject(self, x: int, y: int, depth: float) -> tuple[float, float, float]:
        """
        Convert a 2D pixel + depth to a 3D point in camera space (metres).

        Uses the color stream intrinsics and the RealSense SDK deprojection,
        which accounts for focal length, principal point, and lens distortion.

        Args:
            x, y: pixel coordinates in the color image.
            depth: depth at that pixel in metres (from depth_frame.get_distance).

        Returns:
            (X, Y, Z) in metres in camera coordinate space.
        """
        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(x), float(y)], depth)
        return point[0], point[1], point[2]

    def release(self) -> None:
        """Stop the RealSense pipeline and free hardware resources."""
        self.pipeline.stop()