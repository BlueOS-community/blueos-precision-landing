#!/usr/bin/env python3

"""
Image Correction Module for Precision Landing

This module handles camera calibration loading and image undistortion functionality.
Supports multiple camera types with their respective calibration files.
"""

import cv2
import numpy as np
import logging
import os
from typing import Optional

# Get logger
logger = logging.getLogger("precision-landing")

# Module-level variables for camera calibration
camera_type: Optional[str] = None
camera_matrix: Optional[np.ndarray] = None
distortion_coeffs: Optional[np.ndarray] = None
calibration_loaded: bool = False


# load camera calibration for a specific camera type
# .npz files for each camera type should be placed in the camera-calibration directory
def init_camera_calibration(camera_type_param: str) -> bool:
    """
    Initialize camera calibration for the specified camera type.

    Args:
        camera_type_param: String identifier for the camera type (e.g., "xfrobot-z1-mini")

    Returns:
        bool: True if calibration was loaded successfully, False otherwise
    """
    global camera_type, camera_matrix, distortion_coeffs, calibration_loaded

    try:
        # Reset previous calibration
        camera_type = None
        camera_matrix = None
        distortion_coeffs = None
        calibration_loaded = False

        # Construct calibration file path
        calibration_dir = os.path.join(os.path.dirname(__file__), "camera-calibration")
        calibration_file = os.path.join(calibration_dir, f"{camera_type_param}.npz")

        # Check if calibration file exists
        if not os.path.exists(calibration_file):
            logger.warning(f"image_correction: calibration file not found: {calibration_file}")
            return False

        # Load calibration data
        # acceptable keywords are "camera_matrix" and "distortion_coeffs" or "K" and "D"
        with np.load(calibration_file) as data:
            camera_matrix_found = False
            distortion_coeffs_found = False

            if 'camera_matrix' in data and 'distortion_coeffs' in data:
                # Standard naming convention
                camera_matrix = data['camera_matrix']
                distortion_coeffs = data['distortion_coeffs']
                camera_matrix_found = True
                distortion_coeffs_found = True
            elif 'K' in data and 'D' in data:
                # OpenCV calibration script naming convention (K=camera matrix, D=distortion coeffs)
                camera_matrix = data['K']
                distortion_coeffs = data['D']
                camera_matrix_found = True
                distortion_coeffs_found = True
            else:
                # Check what keys are actually available for debugging
                available_keys = list(data.keys())
                logger.error(f"image_correction: invalid calibration file format in {calibration_file}")
                logger.error(f"image_correction: available keys: {available_keys}")
                logger.error(f"image_correction: expected either (camera_matrix, distortion_coeffs) or (K, D)")
                return False

            if not (camera_matrix_found and distortion_coeffs_found):
                logger.error(f"image_correction: missing calibration data in {calibration_file}")
                return False
            camera_type = camera_type_param
            calibration_loaded = True

            logger.info(f"image_correction: loaded calibration for {camera_type_param}")
            logger.debug(f"image_correction: camera matrix shape: {camera_matrix.shape}")
            logger.debug(f"image_correction: distortion coeffs shape: {distortion_coeffs.shape}")

            return True

    except Exception as e:
        logger.error(f"image_correction: failed to load calibration for {camera_type_param}: {str(e)}")
        return False


# returns true if camera calibration has been loaded
def has_camera_calibration() -> bool:
    """
    Check if camera calibration is loaded and available.
    This is a fast check that callers can use to avoid unnecessary function calls.

    Returns:
        bool: True if calibration is loaded and ready for undistortion
    """
    return calibration_loaded and camera_matrix is not None and distortion_coeffs is not None


# undistort an image using the loaded camera calibration
def undistort_image(image: np.ndarray) -> np.ndarray:
    """
    Undistort an image using the loaded camera calibration.

    Args:
        image: Input image as numpy array (BGR format from OpenCV)

    Returns:
        np.ndarray: Undistorted image, or original image if no calibration available
    """
    # return original if no calibration
    if not has_camera_calibration():
        return image

    try:
        # Perform undistortion
        undistorted = cv2.undistort(image, camera_matrix, distortion_coeffs)

        logger.debug(f"image_correction: undistorted image shape: {undistorted.shape}")
        return undistorted

    except Exception as e:
        logger.error(f"image_correction: failed to undistort image: {str(e)}")
        # Return original image on error
        return image
