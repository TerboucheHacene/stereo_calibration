from typing import List, Tuple
import cv2
import numpy as np
import os
from tqdm import tqdm


class StereoCalibration:
    """Stereo calibration class.

    Parameters
    ----------
    input_path : str
        path to input data
    chessboard_size : Tuple
        size of chessboard (rows, columns)
    square_size : float
        size of chessboard square in meters
    """

    def __init__(
        self,
        input_path: str,
        chessboard_size: Tuple,
        square_size: float,
    ) -> None:
        self.input_path = input_path
        self.image_path = os.path.join(input_path, "images")
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.corners_path = os.path.join(input_path, "corners")
        self._initialize()
        self._init_paths()

    def _init_paths(self) -> None:
        """Initialize paths to left and right images.

        The left and right images are stored in separate directories as follows:
        - images
            - left
                - left_0000.png
                - left_0001.png
                - ...
            - right
                - right_0000.png
                - right_0001.png
                - ...
        """
        self._left_image_path = os.path.join(self.image_path, "left")
        self._right_image_path = os.path.join(self.image_path, "right")
        self._left_image_names = os.listdir(self._left_image_path)
        self._right_image_names = os.listdir(self._right_image_path)
        self._left_image_names = [
            name for name in self._left_image_names if name.endswith(".png")
        ]
        self._right_image_names = [
            name for name in self._right_image_names if name.endswith(".png")
        ]
        self._left_image_names.sort()
        self._right_image_names.sort()
        self._left_image_paths = [
            os.path.join(self._left_image_path, name) for name in self._left_image_names
        ]
        self._right_image_paths = [
            os.path.join(self._right_image_path, name)
            for name in self._right_image_names
        ]

    def _initialize(self) -> None:
        self._chessboard_3d_left_points: List[np.ndarray] = []
        self._chessboard_2d_left_points: List[np.ndarray] = []
        self._chessboard_3d_right_points: List[np.ndarray] = []
        self._chessboard_2d_right_points: List[np.ndarray] = []

        self._stereo_map_left: Tuple[np.ndarray, np.ndarray] = []
        self._stereo_map_right: Tuple[np.ndarray, np.ndarray] = []

        self._left_image_size: Tuple[int, int] = None
        self._right_image_size: Tuple[int, int] = None

    def create_3d_chessboard_points(self) -> np.ndarray:
        """Create 3D chessboard points for a given chessboard size and square size.

        Returns
        -------
        np.ndarray
            Nx3 array of 3D points: N = number of squares in chessboard
        """

        # create Nx3 array of 3D points: N = number of squares in chessboard
        chessboard_points = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        # fill x, y coordinates with indices of chessboard points
        chessboard_points[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2)
        # scale coordinates by square size
        chessboard_points *= self.square_size
        return chessboard_points

    def create_2d_chessboard_points(self, image: np.ndarray) -> np.ndarray:
        """Create 2D chessboard points for a given image and chessboard size.

        Parameters
        ----------
        image : np.ndarray
            image of chessboard

        Returns
        -------
        np.ndarray
            Nx1x2 array of 2D points: N = number of corners found in the image
        """
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # find chessboard corners
        found, corners = cv2.findChessboardCorners(image, self.chessboard_size, None)
        if not found:
            return None
        # refine corner positions
        corners = cv2.cornerSubPix(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )
        return corners

    def plot_chessboard(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Plot chessboard corners on the image.

        Parameters
        ----------
        image : np.ndarray
            image of chessboard
        corners : np.ndarray
            Nx1x2 array of 2D points: N = number of corners found in the image
        """
        image = cv2.drawChessboardCorners(image, self.chessboard_size, corners, True)
        return image

    def create_chessboard_points(self) -> None:
        """Create 3D and 2D chessboard points for left and right images."""

        for left_image_path, right_image_path in tqdm(
            zip(self._left_image_paths, self._right_image_paths)
        ):
            left_image = cv2.imread(left_image_path)
            right_image = cv2.imread(right_image_path)
            left_corners = self.create_2d_chessboard_points(left_image)
            right_corners = self.create_2d_chessboard_points(right_image)

            if self.corners_path is not None:
                left_image = self.plot_chessboard(left_image, left_corners)
                right_image = self.plot_chessboard(right_image, right_corners)
                left_image_name = os.path.basename(left_image_path)
                right_image_name = os.path.basename(right_image_path)
                left_image_path = os.path.join(
                    self.corners_path, "left", left_image_name
                )
                right_image_path = os.path.join(
                    self.corners_path, "right", right_image_name
                )
                os.makedirs(os.path.dirname(left_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(right_image_path), exist_ok=True)
                cv2.imwrite(left_image_path, left_image)
                cv2.imwrite(right_image_path, right_image)

            if left_corners is not None and right_corners is not None:
                self._chessboard_3d_left_points.append(
                    self.create_3d_chessboard_points()
                )
                self._chessboard_2d_left_points.append(left_corners)
                self._chessboard_3d_right_points.append(
                    self.create_3d_chessboard_points()
                )
                self._chessboard_2d_right_points.append(right_corners)

        self._left_image_size = (left_image.shape[1], left_image.shape[0])
        self._right_image_size = (right_image.shape[1], right_image.shape[0])

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate stereo camera.

        The calibration process consists of the following steps:
        1. Calibrate left camera
        2. Calibrate right camera
        3. Stereo calibration
        4. Stereo rectification
        5. Stereo mapping


        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            stereo_map_left, stereo_map_right
        """
        # left camera calibration
        print("Calibrating left camera...")
        ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            self._chessboard_3d_left_points,
            self._chessboard_2d_left_points,
            (self._left_image_size),
            None,
            None,
        )
        print("Getting optimal new camera matrix for left camera...")
        newcameramtx_left, roi_left = cv2.getOptimalNewCameraMatrix(
            mtx_left, dist_left, self._left_image_size, 1, self._left_image_size
        )

        # right camera calibration
        print("Calibrating right camera...")
        ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = (
            cv2.calibrateCamera(
                self._chessboard_3d_right_points,
                self._chessboard_2d_right_points,
                (self._right_image_size),
                None,
                None,
            )
        )
        print("Getting optimal new camera matrix for right camera...")
        newcameramtx_right, roi_right = cv2.getOptimalNewCameraMatrix(
            mtx_right, dist_right, self._right_image_size, 1, self._right_image_size
        )

        # stereo calibration
        print("Calibrating stereo camera...")
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
        (
            ret_stereo,
            newcameramtx_left,
            dist_left,
            newcameramtx_right,
            dist_right,
            rot,
            trans,
            essential,
            fundamental,
        ) = cv2.stereoCalibrate(
            self._chessboard_3d_left_points,
            self._chessboard_2d_left_points,
            self._chessboard_2d_right_points,
            newcameramtx_left,
            dist_left,
            newcameramtx_right,
            dist_right,
            self._left_image_size,
            criteria=criteria,
            flags=flags,
        )
        print("translation vector", trans)
        print("rotation matrix", rot)

        # stereo rectification
        print("Rectifying stereo camera...")
        rect_left, rect_right, proj_left, proj_right, Q, roi_left, roi_right = (
            cv2.stereoRectify(
                newcameramtx_left,
                dist_left,
                newcameramtx_right,
                dist_right,
                self._left_image_size,
                rot,
                trans,
                1,
                (0, 0),
            )
        )

        # stereo mapping
        print("Mapping left and right cameras...")
        stereo_map_left = cv2.initUndistortRectifyMap(
            newcameramtx_left,
            dist_left,
            rect_left,
            proj_left,
            self._left_image_size,
            cv2.CV_16SC2,
        )
        stereo_map_right = cv2.initUndistortRectifyMap(
            newcameramtx_right,
            dist_right,
            rect_right,
            proj_right,
            self._right_image_size,
            cv2.CV_16SC2,
        )
        print("Calibration complete!")
        self._stereo_map_left = stereo_map_left
        self._stereo_map_right = stereo_map_right
        return stereo_map_left, stereo_map_right

    def save_stereo_calibration(self) -> None:
        """Save stereo calibration to XML file."""

        print("Saving stereo calibration...")
        xml_path = os.path.join(self.input_path, "params", "stereo_calibration.xml")
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_WRITE)
        fs.write("stereo_map_left_x", self._stereo_map_left[0])
        fs.write("stereo_map_left_y", self._stereo_map_left[1])
        fs.write("stereo_map_right_x", self._stereo_map_right[0])
        fs.write("stereo_map_right_y", self._stereo_map_right[1])
        fs.release()

    def read_stereo_calibration(self, xml_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read stereo calibration from XML file.

        Parameters
        ----------
        xml_path : str
            path to XML file

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            stereo_map_left, stereo_map_right
        """

        print("Reading stereo calibration...")
        # use cv2.FileStorage to read stereo calibration
        fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        stereo_map_left_x = fs.getNode("stereo_map_left_x").mat()
        stereo_map_left_y = fs.getNode("stereo_map_left_y").mat()
        stereo_map_right_x = fs.getNode("stereo_map_right_x").mat()
        stereo_map_right_y = fs.getNode("stereo_map_right_y").mat()
        fs.release()
        return (stereo_map_left_x, stereo_map_left_y), (
            stereo_map_right_x,
            stereo_map_right_y,
        )

    def rectify_images(
        self, left_image: np.ndarray, right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify left and right images using stereo mapping.

        Parameters
        ----------
        left_image : np.ndarray
            left image
        right_image : np.ndarray
            right image

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            left_rectified, right_rectified
        """
        # rectify images
        left_rectified = cv2.remap(
            left_image,
            self._stereo_map_left[0],
            self._stereo_map_left[1],
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )
        right_rectified = cv2.remap(
            right_image,
            self._stereo_map_right[0],
            self._stereo_map_right[1],
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )
        return left_rectified, right_rectified

    def rectify_calibration_images(self) -> None:
        """Rectify calibration images and save them to disk.
        
        The rectified images are stored in separate directories as follows:
        - input_path
            - rectified
                - left
                    - left_0000.png
                    - left_0001.png
                    - ...
                - right
                    - right_0000.png
                    - right_0001.png
                    - ...
        """

        for left_image_path, right_image_path in tqdm(
            zip(self._left_image_paths, self._right_image_paths)
        ):
            left_image = cv2.imread(left_image_path)
            right_image = cv2.imread(right_image_path)
            left_rectified, right_rectified = self.rectify_images(
                left_image, right_image
            )
            left_image_name = os.path.basename(left_image_path)
            right_image_name = os.path.basename(right_image_path)
            left_image_path = os.path.join(
                self.input_path, "rectified", "left", left_image_name
            )
            right_image_path = os.path.join(
                self.input_path, "rectified", "right", right_image_name
            )
            os.makedirs(os.path.dirname(left_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(right_image_path), exist_ok=True)
            cv2.imwrite(left_image_path, left_rectified)
            cv2.imwrite(right_image_path, right_rectified)
