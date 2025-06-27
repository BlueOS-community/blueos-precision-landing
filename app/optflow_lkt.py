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

# Get logger
logger = logging.getLogger("optical-flow")

def get_optical_flow(image: np.ndarray, prev_image: np.ndarray, include_augmented_image: bool) -> Dict[str, Any]:
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
        # Convert to grayscale for corner detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Convert prev_image to grayscale for corner detection
        if len(prev_image.shape) == 3:
            gray_prev = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_prev = prev_image

        # Create Shi-Tomasi corner detector parameters
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Create Lucas-Kanade optical flow parameters
        lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        corners0 = cv2.goodFeaturesToTrack(
            gray_prev,
            mask=None,
            **feature_params
        )
        if corners0 is None or len(corners0) == 0:
            logger.warning(f"{logging_prefix_str} No corners detected in the previous image")
            return {
                "success": False,
                "message": "No corners detected in the previous image",
                "flow": None,
                "image_base64": ""
            }

        corner1, st, err = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray, corners0, None, **lk_params
        )

        # select only good points
        if corner1 is not None:
            good_new = corner1[st.ravel() == 1]
            good_old = corners0[st.ravel() == 1]

            # Calculate flow vectors
            flow_vectors = good_new - good_old

            # flow_info dictionary to store additional information
            flow_info = {
                "num_points": len(good_new),
                "points": good_new.tolist(),
                "old_points": good_old.tolist(),
                "flow_vectors": flow_vectors.tolist()
            }

            image_base64 = ""
            if include_augmented_image:
                augmented_image = image.copy()
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
