import argparse
import sys

from roarm_rl.app import run
from roarm_rl.hardware import RoArmHardware, RoArmHardwareError


def parse_args():
    parser = argparse.ArgumentParser(description="RoArm-M2 3D GUI (sim + optional hardware mirror)")
    parser.add_argument("--hw", choices=["none", "serial", "http"], default="none",
                         help="connect to a real RoArm-M2 over serial or WiFi/HTTP (default: none)")
    parser.add_argument("--port", default="COM5", help="serial port, e.g. COM5 (Windows) or /dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--host", default="192.168.4.1", help="RoArm IP address for --hw http")
    return parser.parse_args()


def main():
    args = parse_args()

    hardware = None
    if args.hw != "none":
        hardware = RoArmHardware()
        try:
            if args.hw == "serial":
                hardware.connect_serial(args.port, args.baudrate)
            else:
                hardware.connect_http(args.host)
        except RoArmHardwareError as e:
            print(f"[hardware] {e}", file=sys.stderr)
            hardware = None

    run(hardware=hardware)


if __name__ == "__main__":
    main()
