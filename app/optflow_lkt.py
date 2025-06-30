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

REV_FLOW = False
# Get logger
logger = logging.getLogger("optical-flow")

# previous image
prev_image: np.ndarray = None
prev_image_time = None

def predict_next_frame(image: np.ndarray, flow: Dict[str, Any]) -> np.ndarray:
    """
    Predict the next frame features based on the current image and constant velocity model.
    """

    return

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
        - flow: Optical flow vectors and other information
        - image_base64: Base64 encoded image (augmented if include_augmented_image=True, original if False, empty if no image requested)
        - message: Status message
    """

    # logging prefix for all messages from this function
    logging_prefix_str = "get_optical_flow:"

    try:
        # variables
        global prev_image, prev_image_time

        # Convert to grayscale for corner detection
        if len(curr_image.shape) == 3:
            curr_image_grey = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        else:
            curr_image_grey = curr_image

        # Calculate dt
        dt = capture_time - prev_image_time

        # store current image as previous
        prev_image = curr_image_grey
        prev_image_time = capture_time

        # Create Shi-Tomasi corner detector parameters
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Create Lucas-Kanade optical flow parameters
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW

        lk_params = dict( winSize  = (21, 21),
                  maxLevel = 1, criteria = criteria)

        corners0 = cv2.goodFeaturesToTrack(
            prev_image,
            mask=None,
            **feature_params
        ) # TODO: Add initial guess for corners

        if corners0 is None or len(corners0) == 0:
            logger.warning(f"{logging_prefix_str} No corners detected in the previous image")
            return {
                "success": False,
                "message": "No corners detected in the previous image",
                "flow": None,
                "image_base64": ""
            }

        corners1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_image, curr_image_grey, corners0, None, **lk_params)

        # Filter out points
        total_succ = np.sum(st)

        if total_succ < 10:
            corners1, st, err = cv2.calcOpticalFlowPyrLK(prev_image, curr_image_grey, corners0, None, winSize=(21, 21), maxLevel=3)

        if REV_FLOW:
            rev_corners1, stRev, errRev = cv2.calcOpticalFlowPyrLK(
                curr_image_grey, prev_image, corners1, corners0, winSize=(21, 21), maxLevel=1, criteria=criteria, flags=flags)

            dist = np.linalg.norm(corners0 - rev_corners1, axis=1)
            st = st & stRev * (dist <= 0.5)

        if corners1 is not None:
            good_new = corners1[st.ravel() == 1]
            good_old = corners0[st.ravel() == 1]

            # Calculate flow vectors
            flow_vectors = good_new - good_old

            flow_info = {
                "num_points": len(good_new),
                "points": good_new.tolist(),
                "old_points": good_old.tolist(),
                "flow_vectors": flow_vectors.tolist()
            }

            image_base64 = ""
            if include_augmented_image:
                augmented_image = curr_image.copy()
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    cv2.line(augmented_image, (a, b), (c, d), (0, 255, 0), 2)
                    cv2.circle(augmented_image, (a, b), 5, (0, 0, 255), -1)

                _, buffer = cv2.imencode('.jpg', augmented_image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "success": True,
                "flow": flow_info,
                "image_base64": image_base64,
                "message": f"Optical flow calculated for {len(good_new)} points"
            }
        else:
            logger.warning(f"{logging_prefix_str} No good points found after optical flow calculation")
            return {
                "success": False,
                "message": "No good points found after optical flow calculation",
                "flow": None,
                "image_base64": ""
            }

    except Exception as e:
        logger.exception(f"Error during Optical Flow calculation: {str(e)}")
        return {
            "success": False,
            "message": f"Optical Flow calculation failed: {str(e)}",
            "flow": None,
            "image_base64": ""
        }
