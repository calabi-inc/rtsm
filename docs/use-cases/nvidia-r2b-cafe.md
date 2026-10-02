# NVIDIA r2b_cafe — what a bag needs, and why RTSM refuses rather than guesses

**The recording.** `r2b_cafe` from the NVIDIA r2b Dataset 2023 (NGC `nvidia/isaac/r2bdataset2023`, version 1): a 5-second rosbag2 (sqlite3 storage, zstd-compressed messages, no embedded message definitions) from an autonomous-robot sensor rig — a RealSense D455 (1280×720 RGB at 30 Hz, `32FC1` depth), its IR pair, a HAWK stereo pair and a lidar — in a cafe. Licence CC BY 4.0. Fetch it with

```bash
B=https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2023/versions/1/files
mkdir -p r2b_cafe
curl -L -o r2b_cafe/metadata.yaml $B/r2b_cafe/metadata.yaml
curl -L -o r2b_cafe/r2b_cafe_0.db3 $B/r2b_cafe/r2b_cafe_0.db3
```

(1.2 GB). The dataset was published for the Isaac ROS perception stack; it has no pose stream.

**The command:**

```bash
rtsm eval r2b_cafe
```

## What happened

The reader probed the bag, found two things it cannot work around, and stopped before any model loaded (one second, no GPU):

```text
rtsm eval: error: bag refused: no_pose_source: the TF chain base_link -> D455_1:rgb has only static hops: no moving pose in this bag (tf None, odometry None); unaligned_depth: depth K scaled to the RGB size (654.16, 654.16, 651.64, 360.35) vs rgb K (641.42, 640.66, 647.89, 359.72): max rel diff 2.1 % (frames rgb 'D455_1:rgb' / depth 'D455_1:depth'); set io.bag.assume_aligned: true only if the depth IS registered to the RGB
```

Discovery itself worked: the RGB topic `d455_1_rgb_image`, the depth topic `d455_1_depth_image`, both `CameraInfo` topics and `/tf_static` were found without overrides (the topic names carry no leading slash; the bag has no typedefs, so the stock ROS 2 typestore is used). What is missing is not a topic name.

## Why each refusal is right

- **No pose.** RTSM is a spatial memory: every observation is placed in the world through the camera pose at the image stamp. This bag carries only `/tf_static` — `base_link → D455_1 → D455_1:rgb` are fixed calibration hops — and no odometry or pose topic. There is no moving hop anywhere in the chain, so there is no pose to place anything with. The reader could have assumed a static camera; on a robot driving through a cafe that would silently put every object at the first frame's location. It refuses instead and names the chain it inspected.
- **Unregistered depth.** The depth `CameraInfo` (fx 654.2, cx 651.6 after scaling to the RGB size) does not match the RGB one (fx 641.4, cx 647.9): 2.1 % apart, on two different optical frames (`D455_1:rgb` vs `D455_1:depth`). The D455 was recorded without `align_depth`, so a depth pixel does not sit under the RGB pixel of the same index. Using it anyway gives each object a position taken from a neighbouring surface — the 2 % focal-length difference alone is about 13 px at the image edge, before the offset between the two optical frames. The threshold is 1 %; the message prints both K matrices so you can judge the gap.

Both refusals are decided from the metadata (TF topology, `CameraInfo`), not from the images, so the answer is the same on a laptop without a GPU, and it is the same answer `python -m rtsm --bag r2b_cafe` gives at startup.

## What a bag needs

| | needed | r2b_cafe has |
|---|---|---|
| RGB images | a `sensor_msgs/Image` or `CompressedImage` topic | yes, `rgb8` 1280×720 |
| depth, registered to the RGB | an aligned depth topic (`aligned_depth_to_color`, `depth_registered`, or K matrices that agree) | no — raw D455 depth on its own optical frame |
| `CameraInfo` | for the RGB (and the depth, for the registration check) | yes, both |
| **a moving pose** | `/tf` chaining the camera frame to a root through at least one moving hop, or `nav_msgs/Odometry` / `PoseStamped` | **no** — static TF only |
| stamps | header stamps on images and poses (the sensor clock pairs on them; log time is 36–45 ms late in this bag) | yes |

The overrides exist for bags whose discovery is wrong, not for bags whose data is absent: `io.bag.topics.tf` / `odom` point the reader at a pose topic it did not pick, `io.bag.world_frame` / `camera_frame` name the chain ends, `io.bag.assume_aligned: true` tells it the depth *is* registered despite differing calibrations. None of them applies here. To use a sequence like this one with RTSM, record it with the camera's poses (a SLAM or VIO node publishing `odom` or a `map → base_link` TF) and with depth aligned to colour; the [bag guide](../guides/bags.md) lists the roles, the refusals and the overrides in full.

What this page is not: a judgement of the r2b dataset, which was made for a different purpose (perception benchmarks of Isaac ROS, where a pose is not an input). It is the shape of the contract: RTSM tells you what the bag lacks before it spends a GPU minute on it.
