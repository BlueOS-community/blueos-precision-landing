#!/usr/bin/env python3

"""
Optical Flow based motion estimation module

This module handles Optical Flow estimation using a downward facing camera.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any
import base64

REV_FLOW = True  # Reverse flow calculation for additional filtering
# Get logger
logger = logging.getLogger("precision-landing")

# previous image
prev_image: np.ndarray = None
prev_image_time = None

def inBorder(corners: np.ndarray, width: int, height: int) -> bool:
    """
    Check if points are within the image borders.
    """
    return (corners[:, 0] >= 0) & (corners[:, 0] < width) & (corners[:, 1] >= 0) & (corners[:, 1] < height)

def get_optical_flow(curr_image: np.ndarray, capture_time, include_augmented_image: bool) -> Dict[str, Any]:
    """
    Estimate optical flow in the image using the Lucas-Kanade method and Shi-Tomasi corner detection.

    Args:
        image: Input image as numpy array (BGR format from OpenCV)
        prev_image: Previous image for optical flow calculation
        include_augmented_image: Whether to return augmented image with optical flow vectors drawn

    Returns:
        Dictionary containing:
        - success: bool indicating if detection was successful
        - flow_x: average flow value in x axis (None on failure)
        - flow_y: average flow value in y axis (None on failure)
        - dt: Time difference between current and previous image in seconds (None on failure)
        - message: Status message
        - image_base64: Base64 encoded image, None if not requested or could not be generated
    """

    # logging prefix for all messages from this function
    logging_prefix_str = "get_optical_flow:"

    flow_x = None
    flow_y = None
    height, width = curr_image.shape[:2]
    # logger.debug(f"{logging_prefix_str} Image dimensions: {width}x{height}")
    try:
        # variables
        global prev_image, prev_image_time

        # Convert new image to grayscale for corner detection
        if len(curr_image.shape) == 3:
            curr_image_grey = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        else:
            curr_image_grey = curr_image

        # if no previous image, backup current image to previous and return failure
        if prev_image is None:

            # backup current image to previous
            prev_image = curr_image_grey
            prev_image_time = capture_time

            # log and return failure
            logger.debug(f"{logging_prefix_str} No previous image available for optical flow calculation")
            return {
                "success": False,
                "message": "No previous image available",
                "flow_x" : None,
                "flow_y" : None,
                "dt": None,
                "image_base64": None
            }

        # Create Shi-Tomasi corner detector parameters
        feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=7, blockSize=7)

        # Create Lucas-Kanade optical flow parameters
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW

        lk_params = dict( winSize  = (21, 21),
                  maxLevel = 1, criteria = criteria)

        # Detect corners in the previous image
        corners0 = cv2.goodFeaturesToTrack(prev_image, mask=None,**feature_params) # TODO: Add initial guess for corners
        if corners0 is None or len(corners0) == 0:
            logger.warning(f"{logging_prefix_str} No corners detected in the previous image")
            return {
                "success": False,
                "message": "No corners detected in the previous image",
                "flow_x" : None,
                "flow_y" : None,
                "dt": None,
                "image_base64": None
            }

        # Calculate optical flow using Lucas-Kanade method
        corners1, st, err = cv2.calcOpticalFlowPyrLK(prev_image, curr_image_grey, corners0, None, **lk_params)

        # Try again if less than 10 points were tracked
        if np.sum(st) < 10:
            corners1, st, err = cv2.calcOpticalFlowPyrLK(prev_image, curr_image_grey, corners0, None, winSize=(21, 21), maxLevel=3)

        if REV_FLOW:
            rev_corners1, stRev, errRev = cv2.calcOpticalFlowPyrLK(
                curr_image_grey, prev_image, corners1, corners0, winSize=(21, 21), maxLevel=1, criteria=criteria, flags=flags)

            dist = np.linalg.norm(corners0 - rev_corners1, axis=1)
            st = st & stRev & (dist <= 0.5)

        # Use of previous image complete, backup current image to previous
        dt = capture_time - prev_image_time
        prev_image = curr_image_grey
        prev_image_time = capture_time

        # Filter out points that are outside the image borders
        mask = inBorder(corners1, width, height)
        st[~mask] = 0

        # Calculate flow vectors for all tracked points
        good_new = corners1[st.ravel() == 1]
        good_old = corners0[st.ravel() == 1]
        good_errors = err[st.ravel() == 1]

        # Sanity check shapes
        if good_new.shape != good_old.shape or good_new.shape[0] == 0:
            logger.error(f"{logging_prefix_str} Invalid points array shapes, new:{good_new.shape}, "
                     f"old:{good_old.shape}, errors:{good_errors.shape}")
            return {
                "success": False,
                "message": "Invalid points array shapes",
                "flow_x" : None,
                "flow_y" : None,
                "dt": None,
                "image_base64": None
            }

        # Compute flow vectors and reshape to (N, 2)
        flow_vectors = (good_new - good_old).reshape(-1, 2)

        # print out shape of flow vectors
        logger.debug(f"{logging_prefix_str} good_new shape: {good_new.shape}, "
                     f"good_old shape: {good_old.shape}, good_errors shape: {good_errors.shape}"
                     f", flow_vectors shape: {flow_vectors.shape}")

        # Check if any points were successfully tracked
        if len(flow_vectors) == 0:
            logger.warning(f"{logging_prefix_str} No points successfully tracked")
            return {
                "success": False,
                "message": "No points successfully tracked",
                "flow_x" : None,
                "flow_y" : None,
                "dt": None,
                "image_base64": None
            }

        # Calculate weighted average flow rates in x and y axis
        # Use inverse of error as weights (lower error = higher weight)
        # Add small epsilon to avoid division by zero
        weights = 1.0 / (good_errors.ravel() + 1e-6)
        flow_x = np.average(flow_vectors[:, 0], weights=weights) if len(flow_vectors) > 0 else 0
        flow_y = np.average(flow_vectors[:, 1], weights=weights) if len(flow_vectors) > 0 else 0

        # Create augmented image
        image_base64 = None
        if include_augmented_image:
            augmented_image = curr_image.copy()
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(augmented_image, (a, b), (c, d), (0, 255, 0), 2)
                cv2.circle(augmented_image, (a, b), 5, (0, 0, 255), -1)

            _, buffer = cv2.imencode('.jpg', augmented_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

        # return success
        return {
            "success": True,
            "message": "Success",
            "flow_x": flow_x,
            "flow_y": flow_y,
            "dt": dt,
            "image_base64": image_base64
        }

    except Exception as e:
        logger.exception(f"Error during Optical Flow calculation: {str(e)}")
        return {
            "success": False,
            "message": f"Optical Flow calculation failed: {str(e)}",
            "flow_x": flow_x,
            "flow_y": flow_y,
            "dt": None,
            "image_base64": None
        }
