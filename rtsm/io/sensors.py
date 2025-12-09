from __future__ import annotations
from typing import Callable, Optional
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from builtin_interfaces.msg import Time

from rtsm.stores.frame_window import TTLBuf, t_to_float
from rtsm.io.slam import PoseSource
from rtsm.datamodel import FramePacket

class IngestNode(Node):
    def __init__(
        self,
        on_packet_ready: Callable[[FramePacket], None],
        rgb_topic="/camera/rgb/image_rect",
        depth_topic="/camera/depth/image_rect_raw",
        rgb_info_topic="/camera/rgb/camera_info",
        depth_info_topic="/camera/depth/camera_info",
        slop_sec: float = 0.03,
        ttl_sec: float = 3.0,
    ):
        super().__init__("rtsm_ingest")
        self.group = ReentrantCallbackGroup()
        self.on_packet_ready = on_packet_ready
        self.slop = slop_sec

        # tiny TTL caches (only used if info arrives out-of-sync)
        self.rgb_info_buf = TTLBuf(ttl_sec=ttl_sec)
        self.depth_info_buf = TTLBuf(ttl_sec=ttl_sec)

        # pose source (TF lookup)
        self.pose = PoseSource(self, target_frame="map", source_frame="camera_link")

        # subscribe to camera infos (optional)
        self.create_subscription(CameraInfo, rgb_info_topic, self._on_rgb_info, 10, callback_group=self.group)
        self.create_subscription(CameraInfo, depth_info_topic, self._on_depth_info, 10, callback_group=self.group)

        # RGB + Depth sync
        self.rgb_sub   = Subscriber(self, Image, rgb_topic, callback_group=self.group)
        self.depth_sub = Subscriber(self, Image, depth_topic, callback_group=self.group)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=20, slop=slop_sec)
        self.sync.registerCallback(self._on_rgb_depth)  # <-- this is the callback syntax

    # ---------- info listeners ----------
    def _on_rgb_info(self, msg: CameraInfo):
        self.rgb_info_buf.put(msg.header.stamp, msg)

    def _on_depth_info(self, msg: CameraInfo):
        self.depth_info_buf.put(msg.header.stamp, msg)

    # ---------- main pair callback ----------
    def _on_rgb_depth(self, rgb_msg: Image, depth_msg: Image):
        # canonical time = RGB stamp
        t_img: Time = rgb_msg.header.stamp

        # 1) pose at t_img
        (quat_xyzw, t_xyz), ok = self.pose.get_pose_at(t_img, timeout_sec=0.03)
        if not ok:
            return  # quietly skip; SLAM lost or TF not yet in buffer

        # # 2) grab camera infos (nearest within slop)
        # rgb_info  = self.rgb_info_buf.get_nearest(t_to_float(t_img), self.slop)
        # depth_info= self.depth_info_buf.get_nearest(t_to_float(t_img), self.slop)

        # 3) build packet
        pkt = FramePacket(
            stamp=t_img,
            rgb=rgb_msg,
            depth=depth_msg,
            pose_map_T_cam=(quat_xyzw, t_xyz),
            # rgb_info=rgb_info,
            # depth_info=depth_info,
        )

        # 4) emit to your pipeline (the user-provided callback)
        self.on_packet_ready(pkt)
