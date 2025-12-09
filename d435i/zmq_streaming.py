import argparse, time, json, zmq, cv2, numpy as np
import pyrealsense2 as rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub", default="tcp://0.0.0.0:5555")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--view", action="store_true", help="Show RGB and depth visualization windows")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(args.pub)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.w, args.h, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)
    prof = pipe.start(cfg)

    align = rs.align(rs.stream.color)
    depth_sensor = prof.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    cam = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = dict(fx=cam.fx, fy=cam.fy, cx=cam.ppx, cy=cam.ppy, width=cam.width, height=cam.height)

    print("Publishing on", args.pub, "depth_scale_m=", depth_scale)
    topic = b"camera.rgbd"

    try:
        while True:
            frames = pipe.wait_for_frames()
            frames = align.process(frames)
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if not cf or not df:
                continue

            ts_ns = int(time.time_ns())

            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())

            if args.view:
                cv2.imshow("RGB", color)
                depth_vis = np.clip(depth, 0, 5000).astype(np.float32)
                depth_vis = (depth_vis / 5000 * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imshow("Depth", depth_colored)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            _, jpg = cv2.imencode(".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            _, dpng = cv2.imencode(".png", depth)

            meta = dict(
                ts_ns=ts_ns,
                intrinsics=intr,
                depth_units_m=depth_scale,
                encoding=dict(rgb="jpeg", depth="png_u16"),
            )
            bmeta = json.dumps(meta).encode("utf-8")

            sock.send_multipart([topic, bmeta, np.frombuffer(jpg, dtype=np.uint8), np.frombuffer(dpng, dtype=np.uint8)])
    finally:
        pipe.stop()
        if args.view:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


