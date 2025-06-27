#!/usr/bin/env python3

"""
OpticalFlow Calculation

This file calculates optical flow values for a pair of images
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any

# Get logger
logger = logging.getLogger("precision-landing")

# previous image
prev_image: np.ndarray = None
prev_image_time = None

def calc_opticalflow(curr_image: np.ndarray, capture_time) -> Dict[str, Any]:
    """
    Calculate the optical flow values given a pair of images

    Args:
        curr_image: latest image as numpy array (BGR format from OpenCV)

    Returns:
        Dictionary containing:
        - success: bool indicating if flow calculation was successful
        - flow_x: flow value on x-axis
        - flow_y: flow value on y-axis
        - dt: time difference in seconds between current and previous image
        - message: Status message
    """

    try:
        # variables
        global prev_image, prev_image_time
        dt = 0.0

        # Convert to grayscale
        if len(curr_image.shape) == 3:
            curr_image_grey = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        else:
            curr_image_grey = curr_image

        # calculate optical flow values
        if prev_image is None or prev_image_time is None:
            return {
                "success": False,
                "message": f"Optical flow calculation failed: {str(e)}",
                "flow_x" : 0.0,
                "flow_y" : 0.0,
                "dt": 0.0
            }

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_image, curr_image_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate average flow values
        flow_x = np.mean(flow[..., 0])
        flow_y = np.mean(flow[..., 1])

        # Calculate dt
        dt = capture_time - prev_image_time

        # store current image as previous
        prev_image = curr_image_grey
        prev_image_time = capture_time

        # return success or failure
        return {
            "success": True,
            "message": "success",
            "flow_x" : flow_x,
            "flow_y" : flow_y,
            "dt": dt
        }

    except Exception as e:
        logger.exception(f"Optical flow calculation failed: {str(e)}")
        return {
            "success": False,
            "message": f"Optical flow calculation failed: {str(e)}",
            "flow_x" : 0.0,
            "flow_y" : 0.0,
            "dt": 0.0
        }
