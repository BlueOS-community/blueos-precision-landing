#!/usr/bin/env python3

"""
Landing Target MAVLink Message Module for Precision Landing

This module handles sending LANDING_TARGET MAVLink messages to the vehicle
via BlueOS MAV2Rest API interface.
"""

import time
import logging
import math
import json
from typing import Dict, Any, Optional
import urllib.request

# Get logger
logger = logging.getLogger("precision-landing")

# MAV2Rest endpoint
MAV2REST_ENDPOINT = "http://host.docker.internal:6040"

# MAVLink component ID
MAV_COMP_ID_ONBOARD_COMPUTER = 191  # Component ID for onboard computer systems

# LANDING_TARGET message template
LANDING_TARGET_TEMPLATE = """{{
  "header": {{
    "system_id": {sysid},
    "component_id": {component_id},
    "sequence": 0
  }},
  "message": {{
    "type": "LANDING_TARGET",
    "time_usec": {time_usec},
    "target_num": {target_num},
    "frame": {{
      "type": "MAV_FRAME_{frame_name}"
    }},
    "angle_x": {angle_x},
    "angle_y": {angle_y},
    "distance": {distance},
    "size_x": {size_x},
    "size_y": {size_y},
    "x": {x},
    "y": {y},
    "z": {z},
    "q": [
      {q0},
      {q1},
      {q2},
      {q3}
    ],
    "position_type": {{
      "type": "LANDING_TARGET_TYPE_{position_type_name}"
    }}
  }}
}}"""

# COMMAND_LONG message template for SET_MESSAGE_INTERVAL
COMMAND_LONG_SET_MESSAGE_INTERVAL_TEMPLATE = """{{
  "header": {{
    "system_id": {sysid},
    "component_id": {component_id},
    "sequence": 0
  }},
  "message": {{
    "type": "COMMAND_LONG",
    "target_system": {target_system},
    "target_component": {target_component},
    "command": {{
      "type": "MAV_CMD_SET_MESSAGE_INTERVAL"
    }},
    "confirmation": 0,
    "param1": {message_id},
    "param2": {interval_us},
    "param3": {param3},
    "param4": {param4},
    "param5": {param5},
    "param6": {param6},
    "param7": {response_target}
  }}
}}"""

# OPTICAL_FLOW message template
OPTICAL_FLOW_TEMPLATE = """{{
  "header": {{
    "system_id": {sysid},
    "component_id": {component_id},
    "sequence": 0
  }},
  "message": {{
    "type": "OPTICAL_FLOW",
    "time_usec": {time_usec},
    "sensor_id": {sensor_id},
    "flow_x": {flow_x},
    "flow_y": {flow_y},
    "flow_comp_m_x": {flow_comp_m_x},
    "flow_comp_m_y": {flow_comp_m_y},
    "quality": {quality},
    "ground_distance": {ground_distance},
    "flow_rate_x": {flow_rate_x},
    "flow_rate_y": {flow_rate_y}
  }}
}}"""


# send mavlink message using MAV2Rest
def post_to_mav2rest(url: str, data: str) -> Optional[str]:
    """
    Sends a POST request to MAV2Rest with JSON data
    Returns response text if successful, None otherwise
    """
    try:
        jsondata = data.encode("ascii")  # data should be bytes
        req = urllib.request.Request(url, jsondata)
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read().decode()
    except Exception as error:
        logger.error(f"post_to_mav2rest: error : {url}: {error}")
        return None


# send LANDING_TARGET message based on AprilTag information
def send_landing_target(tag_id: int,
                        tag_center_x: float,
                        tag_center_y: float,
                        tag_width_pixels: float,
                        tag_height_pixels: float,
                        image_width: int,
                        image_height: int,
                        camera_hfov_deg: float,
                        camera_vfov_deg: float,
                        sysid: int) -> Dict[str, Any]:
    """
    Convert AprilTag detection to LANDING_TARGET MAVLink message and send it

    Args:
        tag_id: AprilTag ID number
        tag_center_x: X coordinate of tag center in pixels
        tag_center_y: Y coordinate of tag center in pixels
        tag_width_pixels: Width of the tag in pixels
        tag_height_pixels: Height of the tag in pixels
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        camera_hfov_deg: Horizontal field of view in degrees
        camera_vfov_deg: Vertical field of view in degrees
        sysid: System ID to send message to

    Returns:
        Dictionary with conversion and send results
    """

    # logging prefix for all messages from this function
    logging_prefix_str = "send_landing_target:"

    try:
        # Calculate angular offsets
        angles = calculate_angular_offsets(tag_center_x, tag_center_y,
                                           image_width, image_height,
                                           camera_hfov_deg, camera_vfov_deg)

        # Estimate angular size
        size = estimate_target_size_angular(tag_width_pixels, tag_height_pixels,
                                            image_width, image_height,
                                            camera_hfov_deg, camera_vfov_deg)

        # Send LANDING_TARGET message directly (no sender instance needed)
        result = send_landing_target_msg(
            angle_x=angles["angle_x"],
            angle_y=angles["angle_y"],
            distance=0.0,  # Distance unknown from vision alone
            size_x=size["size_x"],
            size_y=size["size_y"],
            target_num=tag_id,  # Use AprilTag ID as target number
            sysid=sysid
        )

        if result["success"]:
            logger.debug(f"{logging_prefix_str} sent LANDING_TARGET for ID {tag_id} with SysID {sysid} CompID {MAV_COMP_ID_ONBOARD_COMPUTER}: "
                         f"angle_x={angles['angle_x_deg']:.2f}, angle_y={angles['angle_y_deg']:.2f}")

        # Add angle and size information to result
        result.update({
            "tag_id": tag_id,
            "angles": angles,
            "size": size,
            "sysid": sysid
        })

        return result

    except Exception as e:
        logger.error(f"{logging_prefix_str} error sending LANDING_TARGET: {str(e)}")
        return {
            "success": False,
            "message": f"Conversion error: {str(e)}",
            "conversion_error": True
        }


# calculate angular offsets required for LANDING_TARGET message
def calculate_angular_offsets(tag_center_x: float,
                              tag_center_y: float,
                              image_width: int,
                              image_height: int,
                              camera_hfov_deg: float,
                              camera_vfov_deg: float) -> Dict[str, float]:
    """
    Calculate angular offsets from AprilTag detection data

    Args:
        tag_center_x: X coordinate of tag center in pixels
        tag_center_y: Y coordinate of tag center in pixels
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        camera_hfov_deg: Horizontal field of view in degrees
        camera_vfov_deg: Vertical field of view in degrees

    Returns:
        Dictionary with angular offset information:
        - angle_x: Horizontal angular offset in radians
        - angle_y: Vertical angular offset in radians
        - angle_x_deg: Horizontal angular offset in degrees
        - angle_y_deg: Vertical angular offset in degrees
    """

    # logging prefix for all messages from this function
    logging_prefix_str = "calculate_angular_offsets:"

    # Validate inputs to prevent mathematical errors
    if image_width <= 0 or image_height <= 0:
        logger.error(f"{logging_prefix_str} invalid image dimensions: {image_width}x{image_height}")
        return {
            "angle_x": 0.0, "angle_y": 0.0, "angle_x_deg": 0.0, "angle_y_deg": 0.0,
            "normalized_x": 0.0, "normalized_y": 0.0, "pixel_offset_x": 0.0, "pixel_offset_y": 0.0
        }

    if camera_hfov_deg <= 0 or camera_vfov_deg <= 0:
        logger.error(f"{logging_prefix_str} invalid FOV hfov={camera_hfov_deg}, vfov={camera_vfov_deg}")
        return {
            "angle_x": 0.0, "angle_y": 0.0, "angle_x_deg": 0.0, "angle_y_deg": 0.0,
            "normalized_x": 0.0, "normalized_y": 0.0, "pixel_offset_x": 0.0, "pixel_offset_y": 0.0
        }

    # Get image center
    image_center_x = image_width / 2
    image_center_y = image_height / 2

    # Calculate pixel offsets from center
    pixel_offset_x = tag_center_x - image_center_x
    pixel_offset_y = tag_center_y - image_center_y

    # Convert to normalized coordinates (-1 to 1)
    normalized_x = pixel_offset_x / (image_width / 2)
    normalized_y = pixel_offset_y / (image_height / 2)

    # Convert to angular offsets
    # For small angles, angle ≈ tan(angle) ≈ normalized_offset * (fov/2)
    angle_x_rad = normalized_x * math.radians(camera_hfov_deg / 2)
    angle_y_rad = normalized_y * math.radians(camera_vfov_deg / 2)

    return {
        "angle_x": angle_x_rad,
        "angle_y": angle_y_rad,
        "angle_x_deg": math.degrees(angle_x_rad),
        "angle_y_deg": math.degrees(angle_y_rad),
        "normalized_x": normalized_x,
        "normalized_y": normalized_y,
        "pixel_offset_x": pixel_offset_x,
        "pixel_offset_y": pixel_offset_y
    }


# Estimate angular size of the AprilTag
def estimate_target_size_angular(tag_width_pixels: float,
                                 tag_height_pixels: float,
                                 image_width: int,
                                 image_height: int,
                                 camera_hfov_deg: float,
                                 camera_vfov_deg: float) -> Dict[str, float]:
    """
    Estimate angular size of the AprilTag target

    Args:
        tag_width_pixels: Width of the tag in pixels
        tag_height_pixels: Height of the tag in pixels
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        camera_hfov_deg: Horizontal field of view in degrees
        camera_vfov_deg: Vertical field of view in degrees

    Returns:
        Dictionary with angular size information:
        - size_x: Horizontal angular size in radians
        - size_y: Vertical angular size in radians
    """

    # logging prefix for all messages from this function
    logging_prefix_str = "estimate_target_size_angular:"

    # Validate inputs to prevent mathematical errors
    if image_width <= 0 or image_height <= 0:
        logger.error(f"{logging_prefix_str} invalid image dimensions: {image_width}x{image_height}")
        return {"size_x": 0.0, "size_y": 0.0, "size_x_deg": 0.0, "size_y_deg": 0.0}

    if camera_hfov_deg <= 0 or camera_vfov_deg <= 0:
        logger.error(f"{logging_prefix_str} invalid FOV hfov={camera_hfov_deg}, vfov={camera_vfov_deg}")
        return {"size_x": 0.0, "size_y": 0.0, "size_x_deg": 0.0, "size_y_deg": 0.0}

    # Convert pixel size to angular size
    pixels_per_degree_h = image_width / camera_hfov_deg
    pixels_per_degree_v = image_height / camera_vfov_deg

    size_x_deg = tag_width_pixels / pixels_per_degree_h
    size_y_deg = tag_height_pixels / pixels_per_degree_v

    size_x_rad = math.radians(size_x_deg)
    size_y_rad = math.radians(size_y_deg)

    return {
        "size_x": size_x_rad,
        "size_y": size_y_rad,
        "size_x_deg": size_x_deg,
        "size_y_deg": size_y_deg
    }


# Low level function to send LANDING_TARGET MAVLink message
def send_landing_target_msg(angle_x: float,
                            angle_y: float,
                            distance: float,
                            size_x: float,
                            size_y: float,
                            target_num: int,
                            sysid: int) -> Dict[str, Any]:
    """
    Send LANDING_TARGET MAVLink message

    Args:
        angle_x: X-axis angular offset in radians
        angle_y: Y-axis angular offset in radians
        distance: Distance to target in meters (0 if unknown)
        size_x: Size of target along x-axis in radians (0 if unknown)
        size_y: Size of target along y-axis in radians (0 if unknown)
        target_num: Target number (0 for standard landing target)
        sysid: System ID to send message to

    Returns:
        Dictionary with send results
    """

    # logging prefix for all messages from this function
    logging_prefix_str = "send_landing_target_msg:"

    try:
        # Get current time in microseconds since UNIX epoch
        current_time = time.time()
        time_usec = int(current_time * 1000000)

        # Always use MAV_FRAME_LOCAL_FRD
        frame_name = "LOCAL_FRD"

        # Map position type integer to type name (always use VISION_FIDUCIAL)
        position_type_name = "VISION_FIDUCIAL"

        # Format the LANDING_TARGET message using BlueOS-style template
        landing_target_data = LANDING_TARGET_TEMPLATE.format(
            sysid=sysid,
            component_id=MAV_COMP_ID_ONBOARD_COMPUTER,
            time_usec=time_usec,
            target_num=target_num,
            frame_name=frame_name,
            angle_x=angle_x,
            angle_y=angle_y,
            distance=distance,
            size_x=size_x,
            size_y=size_y,
            x=0.0,  # X Position of the landing target in MAV_FRAME (not used for angular)
            y=0.0,  # Y Position of the landing target in MAV_FRAME (not used for angular)
            z=0.0,  # Z Position of the landing target in MAV_FRAME (not used for angular)
            q0=1.0,  # Quaternion of landing target orientation (not used)
            q1=0.0,
            q2=0.0,
            q3=0.0,
            position_type_name=position_type_name
        )

        # Send message via MAV2Rest using BlueOS-style post
        url = f"{MAV2REST_ENDPOINT}/mavlink"
        response = post_to_mav2rest(url, landing_target_data)

        if response is not None:
            logger.debug(f"{logging_prefix_str} LANDING_TARGET sent with SysID {sysid} CompID {MAV_COMP_ID_ONBOARD_COMPUTER} angle_x={angle_x:.2f} rad, angle_y={angle_y:.2f} rad")
            return {
                "success": True,
                "message": f"LANDING_TARGET message sent successfully with SysID {sysid} CompID {MAV_COMP_ID_ONBOARD_COMPUTER}",
                "time_usec": time_usec,
                "angle_x": angle_x,
                "angle_y": angle_y,
                "sysid": sysid,
                "response": response
            }
        else:
            logger.error(f"{logging_prefix_str} failed to send LANDING_TARGET")
            return {
                "success": False,
                "message": "MAV2Rest returned no response",
                "network_error": True
            }

    except Exception as e:
        logger.error(f"{logging_prefix_str} unexpected error {str(e)}")
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "unexpected_error": True
        }


# Low level function to send OPTICAL_FLOW MAVLink message
def send_optical_flow_msg(sysid: int,
                          flow_x: int,
                          flow_y: int,
                          flow_comp_m_x: float,
                          flow_comp_m_y: float,
                          quality: int,
                          ground_distance: float,
                          flow_rate_x: float,
                          flow_rate_y: float) -> Dict[str, Any]:
    """
    Send OPTICAL_FLOW MAVLink message

    Args:
        sysid: System ID to send message (normally 1)
        flow_x: Flow in x-sensor direction in dpix
        flow_y: Flow in y-sensor direction in dpiy
        flow_comp_m_x: Flow in x-axis in ground plane in meters/second
        flow_comp_m_y: Flow in y-axis in ground plane in meters/second
        quality: Optical flow quality (0=bad, 255=maximum quality)
        ground_distance: Ground distance in meters, negative if unknown
        flow_rate_x: Flow rate about X axis in radians/second
        flow_rate_y: Flow rate about Y axis in radians/second

    Returns:
        Dictionary with send results
    """

    # logging prefix for all messages from this function
    logging_prefix_str = "send_optical_flow_msg:"

    try:
        # Get current time in microseconds since UNIX epoch
        time_usec = int(time.time() * 1000000)

        # Format the OPTICAL_FLOW message
        optical_flow_data = OPTICAL_FLOW_TEMPLATE.format(
            sysid=sysid,
            component_id=MAV_COMP_ID_ONBOARD_COMPUTER,
            time_usec=time_usec,
            sensor_id=0,
            flow_x=int(flow_x),  # Flow in x-sensor direction in dpix
            flow_y=int(flow_y),  # Flow in y-sensor direction in dpiy
            flow_comp_m_x=flow_comp_m_x,  # Flow in x-sensor direction in m/s, angular-speed compensated
            flow_comp_m_y=flow_comp_m_y,  # Flow in y-sensor direction in m/s, angular-speed compensated
            quality=quality,  # Optical flow quality / confidence. 0: bad, 255: maximum quality
            ground_distance=ground_distance,  # Ground distance. Positive value: distance known. Negative value: Unknown distance
            flow_rate_x=flow_rate_x,  # Flow rate about X axis in radians/second
            flow_rate_y=flow_rate_y   # Flow rate about Y axis in radians/second
        )

        # Send message via MAV2Rest
        url = f"{MAV2REST_ENDPOINT}/mavlink"
        response = post_to_mav2rest(url, optical_flow_data)

        if response is not None:
            logger.debug(f"{logging_prefix_str} OPTICAL_FLOW sent with SysID {sysid} CompID {MAV_COMP_ID_ONBOARD_COMPUTER} flow_rate_x={flow_rate_x:.4f} rad/s, flow_rate_y={flow_rate_y:.4f} rad/s")
            return {
                "success": True,
                "message": f"OPTICAL_FLOW message sent successfully with SysID {sysid} CompID {MAV_COMP_ID_ONBOARD_COMPUTER}",
                "time_usec": time_usec,
                "flow_rate_x": flow_rate_x,
                "flow_rate_y": flow_rate_y,
                "quality": quality,
                "ground_distance": ground_distance,
                "sysid": sysid,
                "response": response
            }
        else:
            logger.error(f"{logging_prefix_str} failed to send OPTICAL_FLOW")
            return {
                "success": False,
                "message": "MAV2Rest returned no response",
                "network_error": True
            }

    except Exception as e:
        logger.error(f"{logging_prefix_str} unexpected error {str(e)}")
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "unexpected_error": True
        }


# Retrieve the latest GIMBAL_DEVICE_ATTITUDE_STATUS message via MAV2Rest
def get_gimbal_attitude(sysid: int) -> Dict[str, Any]:
    """
    Retrieve the latest GIMBAL_DEVICE_ATTITUDE_STATUS message via MAV2Rest

    Args:
        sysid: System ID to query gimbal attitude from

    Returns:
        Dictionary with gimbal attitude data or error information
    """

    # logging prefix for all messages from this function
    logging_prefix_str = "get_gimbal_attitude:"

    try:
        # Request the latest GIMBAL_DEVICE_ATTITUDE_STATUS message from the autopilot
        compid = 1  # autopilot component ID
        url = f"{MAV2REST_ENDPOINT}/mavlink/vehicles/{sysid}/components/{compid}/messages/GIMBAL_DEVICE_ATTITUDE_STATUS"
        req = urllib.request.Request(url)
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=5) as response:
            if response.getcode() == 200:
                response_data = response.read().decode()

                # Parse the response (MAV2Rest returns JSON)
                data = json.loads(response_data)

                # Extract attitude information from the message
                if "message" in data:
                    msg = data["message"]

                    # Extract quaternion components
                    q = msg.get("q", [1.0, 0.0, 0.0, 0.0])

                    # log attitude as quaternion
                    logger.debug(f"{logging_prefix_str} q0:{q[0]:.4f}, q1:{q[1]:.4f}, q2:{q[2]:.5f}, q3:{q[3]:.4f}")

                    return {
                        "success": True,
                        "message": f"Gimbal attitude retrieved successfully",
                        "quaternion": {
                            "w": q[0],
                            "x": q[1],
                            "y": q[2],
                            "z": q[3]
                        }
                    }
                else:
                    logger.warning(f"{logging_prefix_str} could not parse GIMBAL_DEVICE_ATTITUDE_STATUS message")
                    return {
                        "success": False,
                        "message": "Invalid gimbal attitude message format",
                        "raw_response": response_data
                    }
            else:
                logger.warning(f"{logging_prefix_str} HTTP error {response.getcode()} retrieving gimbal attitude")
                return {
                    "success": False,
                    "message": f"HTTP error {response.getcode()}",
                    "http_error": True
                }

    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.warning(f"{logging_prefix_str} GIMBAL_DEVICE_ATTITUDE_STATUS not found")
            return {
                "success": False,
                "message": "Gimbal not found or not publishing attitude data",
                "error": "Gimbal not available"
            }
        else:
            logger.error(f"{logging_prefix_str} could not retrieve GIMBAL_DEVICE_ATTITUDE_STATUS {e.code} - {e.reason}")
            return {
                "success": False,
                "message": f"HTTP error {e.code}: {e.reason}",
                "http_error": True
            }
    except urllib.error.URLError as e:
        logger.error(f"Network error retrieving gimbal attitude: {e.reason}")
        return {
            "success": False,
            "message": f"Network error: {e.reason}",
            "network_error": True
        }
    except Exception as e:
        logger.error(f"{logging_prefix_str} could not retrieve GIMBAL_DEVICE_ATTITUDE_STATUS: {str(e)}")
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "unexpected_error": True
        }


# request GIMBAL_DEVICE_ATTITUDE_STATUS messages at specified rate
def request_gimbal_attitude_status(sysid: int, interval_hz: float) -> Dict[str, Any]:
    """
    Request GIMBAL_DEVICE_ATTITUDE_STATUS messages at specified rate

    Args:
        sysid: System ID to send message to
        interval_hz: Frequency in Hz (1.0 = 1Hz, 0 = disable)

    Returns:
        Dictionary with send results
    """

    logger.debug(f"request_gimbal_attitude_status: requesting GIMBAL_DEVICE_ATTITUDE_STATUS at {interval_hz}Hz")

    return send_set_message_interval(
        sysid=sysid,
        message_id=285,         # GIMBAL_DEVICE_ATTITUDE_STATUS
        interval_hz=interval_hz
    )


# send SET_MESSAGE_INTERVAL command
def send_set_message_interval(sysid: int,  # target system id
                              message_id: int,  # MAVlink message id
                              interval_hz: float) -> Dict[str, Any]:
    """
    Send COMMAND_LONG with MAV_CMD_SET_MESSAGE_INTERVAL to request specific message at given rate

    Args:
        sysid: System ID to send message to (e.g 1 for autopilot)
        message_id: MAVLink message ID to request (e.g 285 for GIMBAL_DEVICE_ATTITUDE_STATUS)
        interval_hz: Frequency in Hz (-1 = disable, 0 = request default rate, 1.0 = 1Hz)

    Returns:
        Dictionary with send results
    """
    logging_prefix_str = "set_message_interval"

    try:
        # Convert frequency to interval in microseconds
        if interval_hz < 0:
            interval_us = -1  # Disable the message
            logger.info(f"{logging_prefix_str}: disabling message {message_id}")
        elif interval_hz == 0:
            interval_us = 0  # Request default rate (0 = default)
            logger.info(f"{logging_prefix_str}: default rate for {message_id}")
        else:
            interval_us = int(1000000 / interval_hz)  # Convert Hz to microseconds
            logger.info(f"{logging_prefix_str}: requesting message ID {message_id} at {interval_hz}Hz ({interval_us}µs interval)")

        # Format the COMMAND_LONG message with SET_MESSAGE_INTERVAL command
        target_compid = 1  # autopilot component ID
        command_data = COMMAND_LONG_SET_MESSAGE_INTERVAL_TEMPLATE.format(
            sysid=sysid,
            component_id=MAV_COMP_ID_ONBOARD_COMPUTER,
            target_system=sysid,
            target_component=target_compid,  # autopilot component ID
            message_id=message_id,
            interval_us=interval_us,
            param3=0.0,        # Unused
            param4=0.0,        # Unused
            param5=0.0,        # Unused
            param6=0.0,        # Unused
            response_target=1  # requestor is target address of message stream
        )

        # Send message via MAV2Rest
        url = f"{MAV2REST_ENDPOINT}/mavlink"
        response = post_to_mav2rest(url, command_data)

        if response is not None:
            logger.info(f"{logging_prefix_str}: sent SET_MESSAGE_INTERVAL to sysid:{sysid} compid:{target_compid} msg_id:{message_id}, interval={interval_hz}Hz)")
            return {
                "success": True,
                "message": f"SET_MESSAGE_INTERVAL sent to sysid:{sysid} compid:{target_compid} msg_id:{message_id}, interval={interval_hz}Hz)",
                "message_id": message_id,
                "interval_hz": interval_hz,
                "interval_us": interval_us,
                "target_component": target_compid,
                "response": response
            }
        else:
            logger.error(f"{logging_prefix_str}: failed to send SET_MESSAGE_INTERVAL to sysid:{sysid} compid:{target_compid} msg_id:{message_id}, interval={interval_hz}Hz)")
            return {
                "success": False,
                "message": "MAV2Rest returned no response",
                "network_error": True
            }

    except Exception as e:
        logger.error(f"{logging_prefix_str}: failed to send SET_MESSAGE_INTERVAL: {str(e)}")
        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}",
            "unexpected_error": True
        }
