"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx


class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
    
    def get_turn_angle(self, idx: int) -> float:
        wp1 = self.maneuverable_waypoints[(idx + 5) % len(self.maneuverable_waypoints)].location[:2]
        wp2 = self.maneuverable_waypoints[(idx + 10) % len(self.maneuverable_waypoints)].location[:2]
        wp3 = self.maneuverable_waypoints[(idx + 15) % len(self.maneuverable_waypoints)].location[:2]

        v1 = wp2 - wp1
        v2 = wp3 - wp2

        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])

        return abs(normalize_rad(a2 - a1))
    
    def get_max_future_turn(self, start_idx: int, end_offset: int, step: int = 4) -> float:
        max_turn = 0.0
        for offset in range(0, end_offset, step):
            idx = (start_idx + offset) % len(self.maneuverable_waypoints)
            turn = self.get_turn_angle(idx)
            max_turn = max(max_turn, turn)
        return max_turn

    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        self.prev_steer = 0.0

    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        # TODO: Implement your solution here.

        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        if vehicle_velocity_norm < 12:
            base_lookahead = 6
        elif vehicle_velocity_norm < 20:
            base_lookahead = 8
        elif vehicle_velocity_norm < 28:
            base_lookahead = 11
        else:
            base_lookahead = 14

        if vehicle_velocity_norm < 12:
            speed_lookahead = 16
        elif vehicle_velocity_norm < 20:
            speed_lookahead = 24
        elif vehicle_velocity_norm < 28:
            speed_lookahead = 34
        elif vehicle_velocity_norm < 36:
            speed_lookahead = 46
        else:
            speed_lookahead = 60

        future_turn_angle = self.get_max_future_turn(
            self.current_waypoint_idx,
            end_offset=speed_lookahead,
            step=2
        )

        if future_turn_angle < 0.05:
            lookahead = base_lookahead + 1
        elif future_turn_angle < 0.10:
            lookahead = base_lookahead
        elif future_turn_angle < 0.16:
            lookahead = base_lookahead - 1
        elif future_turn_angle < 0.24:
            lookahead = base_lookahead - 1
        else:
            lookahead = base_lookahead - 2

        lookahead = int(np.clip(lookahead, 5, 16))

        waypoint_to_follow = self.maneuverable_waypoints[
            (self.current_waypoint_idx + lookahead) % len(self.maneuverable_waypoints)
        ]

        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        steer_control = (
            -8.0 / np.sqrt(vehicle_velocity_norm) * delta_heading / np.pi
        ) if vehicle_velocity_norm > 1e-2 else -np.sign(delta_heading)
        
        if future_turn_angle < 0.05:
            steer_boost = 1.00
        elif future_turn_angle < 0.12:
            steer_boost = 1.06
        elif future_turn_angle < 0.22:
            steer_boost = 1.14
        else:
            steer_boost = 1.22

        steer_control *= steer_boost

        steer_control = np.clip(steer_control, -1.0, 1.0)

        max_steer_change = 0.105
        steer_control = np.clip(
            steer_control,
            self.prev_steer - max_steer_change,
            self.prev_steer + max_steer_change
        )
        self.prev_steer = steer_control

        if not hasattr(self, "step_count"):
            self.step_count = 0
        self.step_count += 1

        turn_angle = future_turn_angle

        if future_turn_angle < 0.05:
            target_speed = 87.0
        elif future_turn_angle < 0.12:
            target_speed = 70.0
        elif future_turn_angle < 0.22:
            target_speed = 47.0
        else:
            target_speed = 35.0

        if turn_angle < 0.04:
            brake_buffer = 0.0
        elif turn_angle < 0.08:
            brake_buffer = 0.8
        elif turn_angle < 0.14:
            brake_buffer = 1.5
        elif turn_angle < 0.23:
            brake_buffer = 2.7
        else:
            brake_buffer = 5.2

        effective_target_speed = target_speed - brake_buffer
        speed_error = effective_target_speed - vehicle_velocity_norm

        if speed_error > 8:
            throttle = 1.0
            brake = 0.0
        elif speed_error > 3:
            throttle = 0.7
            brake = 0.0
        elif speed_error > 0:
            throttle = 0.3
            brake = 0.0
        elif speed_error > -3:
            throttle = 0.0
            brake = 0.15
        elif speed_error > -7:
            throttle = 0.0
            brake = 0.4
        else:
            throttle = 0.0
            brake = 0.75

        if future_turn_angle > 0.22 and vehicle_velocity_norm > target_speed + 2:
            throttle = 0.0
            brake = max(brake, 0.85)
        elif future_turn_angle > 0.18 and vehicle_velocity_norm > target_speed + 4:
            throttle = 0.0
            brake = max(brake, 0.58)
            

        if self.step_count % 20 == 0:
            print(
                f"speed={vehicle_velocity_norm:.2f} | "
                f"lookahead={lookahead} | "
                f"turn_angle={turn_angle:.3f} | "
                f"target_speed={target_speed:.1f} | "
                f"steer={steer_control:.3f} | "
                f"throttle={throttle:.2f} | "
                f"brake={brake:.2f}"
            )

        control = {
            "throttle": throttle,
            "steer": steer_control,
            "brake": brake,
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control
