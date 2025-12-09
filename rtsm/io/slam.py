from __future__ import annotations
from typing import Tuple, Optional
import rclpy
from rclpy.duration import Duration
import tf2_ros
from builtin_interfaces.msg import Time

class PoseSource:
    def __init__(self, node, target_frame="map", source_frame="camera_link", tf_cache_sec: float = 20.0):
        self.node = node
        self.target = target_frame
        self.source = source_frame
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=tf_cache_sec))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

    def get_pose_at(self, stamp: Time, timeout_sec: float = 0.03) -> Tuple[Optional[list], bool]:
        try:
            tf = self.tf_buffer.lookup_transform(self.target, self.source, stamp,
                                                 timeout=Duration(seconds=timeout_sec))
        except Exception:
            return None, False
        # turn TransformStamped into a 4x4 (or (q, t)); here we return (q_xyzw, t_xyz)
        t = tf.transform.translation
        q = tf.transform.rotation
        return ([q.x, q.y, q.z, q.w], [t.x, t.y, t.z]), True
