"""Quick sanity check for the Raspberry Pi Camera Module 3 NoIR."""

from pathlib import Path

from picamera2 import Picamera2
from PIL import Image

OUTPUT_PATH = Path(__file__).parent / "test_capture.jpg"


def main():
    cam = Picamera2()
    print("Camera properties:", cam.camera_properties)

    config = cam.create_still_configuration(main={"size": (640, 480)})
    cam.configure(config)
    cam.start()

    frame = cam.capture_array()
    print(f"Captured frame: shape={frame.shape}, dtype={frame.dtype}")

    Image.fromarray(frame).save(OUTPUT_PATH)
    print(f"Saved to {OUTPUT_PATH}")

    cam.stop()
    cam.close()
    print("Camera test passed!")


if __name__ == "__main__":
    main()
