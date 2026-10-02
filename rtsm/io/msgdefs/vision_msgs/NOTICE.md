# vision_msgs message definitions

The `.msg` files in `ros2/` and `ros1/` are copied verbatim from
[ros-perception/vision_msgs](https://github.com/ros-perception/vision_msgs)
(branches `ros2` and `noetic-devel`, fetched 2026-10-02), licensed under the
Apache License, Version 2.0 (the same licence as this repository; the full text
is in the repository's `LICENSE`). They are redistributed unchanged so that
`rtsm` can read `Detection2DArray` / `Detection3DArray` topics from bags that do
not embed their message definitions, without a ROS installation.
