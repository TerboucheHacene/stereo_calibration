from calibration import StereoCalibration
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo Camera Calibration")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data",
        help="Path to the folder containing calibration images",
    )
    parser.add_argument(
        "--chessboard_size",
        type=str,
        default="11,7",
        help="Size of the chessboard (width, height)",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=0.003,
        help="Size of the chessboard square in meters",
    )
    return parser.parse_args()


def main(
    input_path="data",
    chessboard_size=(11, 7),
    square_size=0.003,
):
    stereo_calibrator = StereoCalibration(
        input_path=input_path,
        chessboard_size=chessboard_size,
        square_size=square_size,
    )
    stereo_calibrator.create_chessboard_points()
    stereo_calibrator.calibrate()
    stereo_calibrator.save_stereo_calibration()
    stereo_calibrator.rectify_calibration_images()


if __name__ == "__main__":
    args = parse_args()
    chessboard_size = tuple(map(int, args.chessboard_size.split(",")))
    main(
        input_path=args.input_path,
        chessboard_size=chessboard_size,
        square_size=args.square_size,
    )
