#!/usr/bin/env python3

"""
Image Capture Module for Precision Landing

This module handles RTSP camera connection and frame capture functionality.

Methods include:

- `test_rtsp_connection`: Tests RTSP connection and captures a frame.


Based on the working implementation from https://github.com/mzahana/siyi_sdk
"""

import cv2
import base64
import logging
import threading
import time
from typing import Dict, Any

# Get logger
logger = logging.getLogger("precision-landing")

# Import AprilTag detection module with error handling
try:
    from app import april_tags
    APRIL_TAGS_AVAILABLE = True
except ImportError as e:
    logger.error(f"AprilTag detection module not available: {str(e)}")
    APRIL_TAGS_AVAILABLE = False
    april_tags = None

# video capture objects
video_capture = None
video_capture_rtsp_url = None
video_capture_mutex = threading.Lock()


# captures a single frame from an RTSP stream
def capture_frame_from_stream(rtsp_url: str) -> Dict[str, Any]:
    """
    Capture a single frame from an RTSP stream.
    This is a simplified version focused on just getting one frame quickly.

    Args:
        rtsp_url: The RTSP URL to connect to

    Returns:
        Dictionary with success status and frame data (numpy array)
    """
    global rtsp_stream_reader

    # logging prefix for all messages from this function
    logging_prefix_str = "capture_frame_from_stream:"

    try:
        # Ensure video capture is thread-safe
        with video_capture_mutex:
            # Initialize reader if needed
            if not rtsp_stream_reader.start(rtsp_url):
                logger.error(f"{logging_prefix_str} failed to start RTSP stream {rtsp_url}")
                return {
                    "success": False,
                    "message": "Failed to start RTSP stream reader",
                    "error": "RTSP stream reader start failed"
                }

            # Get latest frame from the background thread
            ret, frame = rtsp_stream_reader.get_latest_frame()

        # check if frame was read successfully
        if not ret or frame is None:
            logger.error(f"{logging_prefix_str} no frame available from RTSP stream")
            return {
                "success": False,
                "message": "No frame available from video stream",
                "error": "Frame read failed"
            }

        # get dimensions and return frame
        height, width = frame.shape[:2]
        return {
            "success": True,
            "message": f"Frame captured successfully ({width}x{height})",
            "frame": frame,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height
        }

    except Exception as e:
        logger.exception(f"{logging_prefix_str} exception {str(e)}")
        return {
            "success": False,
            "message": f"Error capturing frame: {str(e)}",
            "error": str(e)
        }


# Clean up video capture resources
def cleanup_video_capture():
    """
    Clean up the video capture object and release resources.
    Should be called when stopping precision landing or when done.
    """
    global rtsp_stream_reader

    with video_capture_mutex:
        # Stop the RTSP stream reader
        rtsp_stream_reader.stop()
        logger.info("Cleaning up video capture")


# RTSPStreamReader class continuously reads frames from an RTSP stream, keeping only the latest frame
class RTSPStreamReader:
    def __init__(self):
        self.cap = None  # VideoCapture object
        self.rtsp_url = None  # Current RTSP URL
        self.frame_latest = None  # Latest frame
        self.frame_latest_time = 0.0  # Timestamp of latest frame
        self.frame_sem = threading.Semaphore(1)  # Protect frame access
        self.running = False
        self.thread = None
        self.logging_prefix_str = "RTSPStreamReader:"

    # start reading from the RTSP stream
    # returns True on success, False on failure
    def start(self, rtsp_url):

        # check if already running
        if self.running:

            # if the RTSP URL is the same, no need to restart
            if self.rtsp_url == rtsp_url:
                return True

            # if the RTSP URL has changed, stop the current reader
            self.stop()

        # initialise video capture with the new RTSP URL and timeout parameters
        self.cap = cv2.VideoCapture(rtsp_url,
                                    cv2.CAP_FFMPEG,
                                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                     cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])

        if self.cap.isOpened():
            # read one test frame
            test_ret, test_frame = self.cap.read()
            if not test_ret or test_frame is None:
                logger.error(f"{self.logging_prefix_str} failed to read test frame from {rtsp_url}")
                self.cap.release()
                return False

            # start the reader thread
            self.running = True
            self.rtsp_url = rtsp_url
            self.thread = threading.Thread(target=self._reader)
            self.thread.daemon = True
            self.thread.start()

            # allow time for the thread to start and return success
            time.sleep(0.5)
            logger.info(f"{self.logging_prefix_str} started reader for {rtsp_url}")
            return True
        else:
            logger.error(f"{self.logging_prefix_str} failed to open RTSP stream: {rtsp_url}")
            self.cap.release()
            return False

    # stop reading from the RTSP stream
    def stop(self):
        if self.running:
            logger.info(f"{self.logging_prefix_str} stopped")
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)  # Don't wait forever
            if self.cap:
                self.cap.release()
                self.cap = None
            self.rtsp_url = None

    # get the latest frame
    def get_latest_frame(self):
        with self.frame_sem:
            # Check if we have a recent frame (within last 0.2 seconds)
            current_time = time.time()
            if (self.frame_latest is not None and
                (current_time - self.frame_latest_time) < 0.2):
                return True, self.frame_latest
            else:
                return False, None

    # background thread that reads frames from the RTSP stream
    def _reader(self):
        logger.info(f"{self.logging_prefix_str} thread started")
        first_frame = True
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # update the latest frame with timestamp
                    with self.frame_sem:
                        self.frame_latest = frame.copy()
                        self.frame_latest_time = time.time()

                    if first_frame:
                        logger.debug(f"{self.logging_prefix_str} started receiving frames")
                        first_frame = False

                elif not self.running:
                    # Normal shutdown
                    break
                else:
                    # Failed to read frame, short sleep and retry
                    time.sleep(0.01)
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    logger.error(f"{self.logging_prefix_str} error in reader thread: {e}")
                    time.sleep(0.1)  # Short delay before retry

        logger.info(f"{self.logging_prefix_str} thread stopped")


# Global stream reader
rtsp_stream_reader = RTSPStreamReader()
