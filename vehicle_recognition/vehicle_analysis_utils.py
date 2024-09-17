#
# Copyright 2024 Fraunhofer Italia Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations
from typing import List
from dataclasses import dataclass
from enum import Enum
import numpy as np
from .wheel import Wheel, WheelType
from .utils import (
    wrap_zerotwopi,
    unit_vector,
    angle_between_axes,
    distance_line_point,
    from_vec_point_to_abc_coeffs,
    points_center,
    angle_between,
    angle_between2D,
    get_euler_angles,
    IDENTITY_TF,
    EPSILON,
    X_AXIS,
)

ACCEPTABLE_AXIS_DISTANCE_ERROR: float = 0.02
ACCEPTABLE_AXIS_ANGLE_ERROR: float = 0.02


class VehicleKinematicType(Enum):
    OMNIDIRECTIONAL = "omnidirectional"
    DIFFERENTIAL = "differential"
    CARLIKE = "carlike"

    def __str__(self):
        return str(self.value)


class VehicleKinematic:
    @dataclass
    class Configuration:
        name: str
        frame: str
        tf: np.ndarray

    type: VehicleKinematicType
    configurations: List[Configuration]
    best_configuration: Configuration = None

    def __init__(self, type):
        self.type = type
        self.configurations = []


@dataclass
class WheelsCommonAxis:
    point: np.ndarray
    versor: np.ndarray
    max_distance_error: float
    max_angle_error: float
    extremes: List[float]
    wheels: List[Wheel]


def compute_wheels_common_axis(wheels: List[Wheel]) -> WheelsCommonAxis:
    axes: List[np.ndarray] = []
    contact_points: List[np.ndarray] = []
    for wheel in wheels:
        tf = wheel.origin_tf
        contact_points += [tf[:2, 3]]
        axes += [(tf[:3, :3] @ wheel.joints[-1].axis)[:2]]

    if len(wheels) == 1:
        centers_point = contact_points[0][:2]
        centers_axis = unit_vector(axes[0])
        max_distance_error = 0
        max_angle_error = 0
        extremes = [0, 0]

    elif len(wheels) == 2:
        centers_point = points_center(contact_points)
        centers_axis = unit_vector(contact_points[0] - contact_points[1])
        max_distance_error = 0
        # Angle error between the Centers Axis and the axes of the wheels
        max_angle_error = 0
        for axis in axes:
            angle_error = angle_between_axes(centers_axis, axis)
            if angle_error > max_angle_error:
                max_angle_error = angle_error
        extremes = find_extremes(centers_point, centers_axis, contact_points)

    elif len(wheels) > 2:
        # Distance error between the contact points
        contacts_matrix = contact_points[0].reshape(1, 3)
        for i in range(1, len(wheels)):
            contacts_matrix: np.ndarray = np.concatenate(
                (contacts_matrix, contact_points[i].reshape(1, 3))
            )
        x_vector: np.ndarray = contacts_matrix[:, 0]
        y_vector: np.ndarray = contacts_matrix[:, 1]
        # y = mx + q
        # [m q]' = pinv([x 1]) [y]'
        A_matrix = np.c_[x_vector.reshape(len(wheels), 1), np.ones(len(wheels))]
        coefficients = np.linalg.pinv(A_matrix) @ y_vector
        y_dependent = True
        if abs(coefficients[0]) > 1:
            # x = my + q
            # [m q]' = pinv([y 1]) [x]'
            A_matrix = np.c_[y_vector.reshape(len(wheels), 1), np.ones(len(wheels))]
            coefficients = np.linalg.pinv(A_matrix) @ x_vector
            y_dependent = False

        # Centerr Axis and Centers Point
        # [m q]' = pinv([x 1]) [y]'
        if y_dependent:
            centers_point = np.array([0, coefficients[1]])
            centers_axis = unit_vector(np.array([1, coefficients[0]]))
        else:
            centers_point = np.array([coefficients[1], 0])
            centers_axis = unit_vector(np.array([coefficients[0], 1]))
        abc_coeffs = from_vec_point_to_abc_coeffs(centers_point, centers_axis)

        # Distance error between the Centers Axis the centers of the wheels
        # straight line coefficients ax + by + c = 0
        max_distance_error = 0
        for point in contact_points:
            distance_error = distance_line_point(abc_coeffs, point)
            if distance_error > max_distance_error:
                max_distance_error = distance_error

        # Angle error between the Centers Axis and the axes of the wheels
        max_angle_error = 0
        for axis in axes:
            angle_error = angle_between_axes(centers_axis, axis)
            if angle_error > max_angle_error:
                max_angle_error = angle_error

        extremes = find_extremes(centers_point, centers_axis, contact_points)
    else:
        return None

    return WheelsCommonAxis(
        centers_point,
        centers_axis,
        max_distance_error,
        max_angle_error,
        extremes,
        wheels,
    )


def find_extremes(
    line_point: np.ndarray, line_vector: np.ndarray, points: List[np.ndarray]
) -> List[np.ndarray]:
    k_min = float("inf")
    k_max = float("-inf")
    # idea coming from https://www.youtube.com/watch?v=l-_DbsFjz_s
    for point in points:
        k = k_value_closestpoint_on_line(line_point, line_vector, point)
        if k > k_max:
            k_max = k
        if k < k_min:
            k_min = k
    return [k_min, k_max]


def k_value_closestpoint_on_line(
    line_point: np.ndarray, line_vector: np.ndarray, point: np.ndarray
):
    k = (np.dot(point, line_vector) - np.dot(line_point, line_vector)) / np.dot(
        line_vector, line_vector
    )
    return k


def search_omnidirectional_kinematics(
    wheels: List[Wheel], autonomous: bool = True
) -> VehicleKinematic:
    exist = False
    steering_wheels: List[Wheel] = []
    for wheel in wheels:
        if wheel.type is WheelType.STEERING:
            steering_wheels += [wheel]
    if len(steering_wheels) == len(wheels) and len(wheels) > 0:
        exist = True
    if exist:
        print("The vehicle can have an omnidirectional kinematics")
        kinematic = VehicleKinematic(VehicleKinematicType.OMNIDIRECTIONAL)
        # configurations with control frame the root frame
        kinematic.configurations.append(
            VehicleKinematic.Configuration(
                VehicleKinematicType.OMNIDIRECTIONAL.__str__()
                + "_"
                + wheel.get_root().name,
                wheel.get_root().name,
                IDENTITY_TF,
            )
        )

        # configuration with control frame the wheels center (considers the steering axis)
        origins: List[np.ndarray] = []
        for wheel in wheels:
            tf = IDENTITY_TF
            for tf_ in wheel.get_tf(wheel.links[0]):
                tf = tf @ tf_[1]
                origins += [tf[:2, 3]]
        center = points_center(origins)
        if np.any((abs(center) > EPSILON)):
            tf = np.array(
                [[1, 0, 0, center[0]], [0, 1, 0, center[1]], [0, 0, 1, 0], [0, 0, 0, 1]]
            )
            kinematic.configurations.append(
                VehicleKinematic.Configuration(
                    VehicleKinematicType.OMNIDIRECTIONAL.__str__()
                    + "_"
                    + "wheels_geometric_center",
                    "wheels_geometric_center",
                    tf,
                )
            )
        # the best omnidirectional configuration is when all wheels are centered
        kinematic.best_configuration = kinematic.configurations[-1]

        # user defined configuration
        print("Default omnidirectional kinematics configuration defined:")
        for configuration in kinematic.configurations:
            print(
                f"- {configuration.name} as a control frame, position {configuration.tf[:2,3]}"
            )
        if not autonomous:
            if str(input("Do you want to define another one? [Yes/No]")) == "Yes":
                name_user = str(input("Which is the frame name?"))
                positions = [
                    float(i)
                    for i in input(
                        f"Write x and y component with respect to the {wheel.get_root().name} separated by a space"
                    ).split()
                ]
                try:
                    x = positions[0]
                except IndexError:
                    x = 0
                try:
                    y = positions[1]
                except IndexError:
                    y = 0
                tf_user = np.array(
                    [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, 0], [0, 0, 0, 1]]
                )
                kinematic.best_configuration = VehicleKinematic.Configuration(
                    VehicleKinematicType.OMNIDIRECTIONAL.__str__() + "_" + name_user,
                    name_user,
                    tf_user,
                )
                kinematic.configurations.append(kinematic.best_configuration)

        return kinematic

    return None


def search_differential_kinematics(wheels: List[Wheel]) -> VehicleKinematic:
    exist = False
    fixed_wheels: List[Wheel] = []
    for wheel in wheels:
        if wheel.type is WheelType.FIXED:
            fixed_wheels += [wheel]
    fixed_axes: List[WheelsCommonAxis] = []
    if len(wheels) > 1:
        if len(fixed_wheels) > 1:
            axis = compute_wheels_common_axis(fixed_wheels)
            if (
                axis.max_angle_error <= ACCEPTABLE_AXIS_ANGLE_ERROR
                and axis.max_distance_error <= ACCEPTABLE_AXIS_DISTANCE_ERROR
            ):
                fixed_axes += [axis]
        elif len(fixed_wheels) == 0:
            couples = [
                (wheel1, wheel2)
                for idx, wheel1 in enumerate(wheels)
                for wheel2 in wheels[idx + 1 :]
            ]
            for couple in couples:
                solutions = steering_wheels_aligning_angles(couple[0], couple[1])
                for solution in solutions:
                    solution1 = solution
                    if not couple[0].clockwise_steer:
                        solution1 = -solution
                    if not couple[0].joints[0].set_value(solution1):
                        if not couple[0].joints[0].set_value(solution1 + np.pi):
                            couple[0].joints[0].set_value(solution1 - np.pi)
                    solution2 = solution
                    if not couple[1].clockwise_steer:
                        solution2 = -solution
                    if not couple[1].joints[0].set_value(solution2):
                        if not couple[1].joints[0].set_value(solution2 + np.pi):
                            couple[0].joints[0].set_value(solution2 - np.pi)
                    axis = compute_wheels_common_axis([couple[0], couple[1]])
                    if (
                        axis.max_angle_error <= ACCEPTABLE_AXIS_ANGLE_ERROR
                        and axis.max_distance_error <= ACCEPTABLE_AXIS_DISTANCE_ERROR
                    ):
                        fixed_axes += [axis]
                    else:
                        print(axis.max_angle_error)
                        print(axis.max_distance_error)
    if len(fixed_axes) > 0:
        exist = True
    if exist:
        print("The vehicle can have an differential kinematics")
        kinematic = VehicleKinematic(VehicleKinematicType.DIFFERENTIAL)
        for i, axis in enumerate(fixed_axes):
            name = "fixed_axis_" + str(i)
            wheels_direction = np.array([axis.versor[1], -axis.versor[0]])
            if angle_between(wheels_direction, X_AXIS[:2]) > np.pi / 2:
                wheels_direction = -wheels_direction
            angle = angle_between2D(X_AXIS[:2], wheels_direction)
            tf = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0, axis.point[0]],
                    [np.sin(angle), np.cos(angle), 0, axis.point[1]],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            corrected_direction = tf[:3, 0]
            print(
                f"- {name} has center in x={axis.point[0]} and y={axis.point[1]} and direction [{corrected_direction[0]}, {corrected_direction[1]}]"
            )
            print(
                f"    it has an error of: distance -> {axis.max_distance_error}, angle -> {axis.max_angle_error}"
            )
            print("  Fixed Wheels:")
            for wheel in axis.wheels:
                print("  - " + wheel.joints[-1].name)

            kinematic.configurations.append(
                VehicleKinematic.Configuration(
                    VehicleKinematicType.DIFFERENTIAL.__str__() + "_" + name, name, tf
                )
            )

        # the best configuration is the one with the fixed axis best oriented
        # here the axis of fixed wheels must be the most orthogonal to the x axis of the root_link
        # between the most orthogonal axes it is chosen the closer one
        smaller_z_angle = float("+inf")
        smaller_distance = float("+inf")
        for configuration in kinematic.configurations:
            z_angle = abs(get_euler_angles(configuration.tf)[0])
            distance = abs(np.linalg.norm(configuration.tf[3, :2]))
            if z_angle < smaller_z_angle:
                smaller_distance = distance
                kinematic.best_configuration = configuration
            elif z_angle == smaller_z_angle:
                if distance < smaller_distance:
                    kinematic.best_configuration = configuration

        return kinematic

    return None


def search_carlike_kinematics(wheels: List[Wheel]) -> VehicleKinematic:
    exist = False
    fixed_wheels: List[Wheel] = []
    for wheel in wheels:
        if wheel.type is WheelType.FIXED:
            fixed_wheels += [wheel]
    fixed_axes: List[WheelsCommonAxis] = []
    if len(wheels) > 1:
        if len(fixed_wheels) > 0:
            axis = compute_wheels_common_axis(fixed_wheels)
            if (
                axis.max_angle_error <= ACCEPTABLE_AXIS_ANGLE_ERROR
                and axis.max_distance_error <= ACCEPTABLE_AXIS_DISTANCE_ERROR
            ):
                fixed_axes += [axis]
        elif len(fixed_wheels) == 0:
            steeringpoints: List[np.ndarray] = []
            wheelaxes: List[np.ndarray] = []
            for wheel in wheels:
                wheel.reset()
                tfs1 = wheel.get_tf(wheel.links[-1])
                tf = IDENTITY_TF
                for tf_ in tfs1[:-1]:
                    tf = tf @ tf_[1]
                steeringpoints += [tf[:2, 3]]
                wheelaxes += [((tf @ tfs1[-1][1])[:3, :3] @ wheel.joints[-1].axis)[:2]]
            center = points_center(steeringpoints)
            for i, wheel in enumerate(wheels):
                psi = angle_between2D(X_AXIS[:2], center - steeringpoints[i])
                # get the zero angle of the wheels
                theta0 = angle_between2D(X_AXIS[:2], wheelaxes[i])
                value = wrap_zerotwopi(psi - theta0 - (np.pi / 2))
                if not wheel.clockwise_steer:
                    value = -value
                if not wheel.joints[0].set_value(value):
                    if not wheel.joints[0].set_value(value + np.pi):
                        wheel.joints[0].set_value(value - np.pi)
                fixed_axes += [compute_wheels_common_axis([wheel])]
    if len(fixed_axes) > 0:
        exist = True
    if exist:
        print("The vehicle can have an car-like kinematics")
        kinematic = VehicleKinematic(VehicleKinematicType.CARLIKE)
        for i, axis in enumerate(fixed_axes):
            name = "fixed_axis_" + str(i)
            wheels_direction = np.array([axis.versor[1], -axis.versor[0]])
            if len(fixed_wheels) == 0:
                if (
                    angle_between(wheels_direction, center - steeringpoints[i])
                    > np.pi / 2
                ):
                    wheels_direction = -wheels_direction
            else:
                if angle_between(wheels_direction, X_AXIS[:2]) > np.pi / 2:
                    wheels_direction = -wheels_direction

            angle = angle_between2D(X_AXIS[:2], wheels_direction)
            tf = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0, axis.point[0]],
                    [np.sin(angle), np.cos(angle), 0, axis.point[1]],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            corrected_direction = tf[:3, 0]
            print(
                f"- {name} has center in x={axis.point[0]} and y={axis.point[1]} and direction [{corrected_direction[0]}, {corrected_direction[1]}]"
            )
            print(
                f"    it has an error of: distance -> {axis.max_distance_error}, angle -> {axis.max_angle_error}"
            )
            print("  Fixed Wheel: ", axis.wheels[0].name)
            kinematic.configurations.append(
                VehicleKinematic.Configuration(
                    VehicleKinematicType.CARLIKE.__str__() + "_" + name, name, tf
                )
            )

        # the best configuration is the one with the fixed axis best oriented
        # here the axis of fixed wheels must be the most orthogonal to the x axis of the root_link
        # between the most orthogonal axes it is chosen the closer one
        smaller_z_angle = float("+inf")
        smaller_distance = float("+inf")
        for configuration in kinematic.configurations:
            z_angle = abs(get_euler_angles(configuration.tf)[0])
            distance = abs(np.linalg.norm(configuration.tf[3, :2]))
            if z_angle < smaller_z_angle:
                smaller_distance = distance
                kinematic.best_configuration = configuration
            elif z_angle == smaller_z_angle:
                if distance < smaller_distance:
                    kinematic.best_configuration = configuration

        return kinematic

    return None


def steering_wheels_aligning_angles(wheel1: Wheel, wheel2: Wheel) -> List[float]:
    solutions: List[float] = []
    # set the default value
    wheel1.reset()
    wheel2.reset()

    # get the position of the steering axis and the vector of the wheel axis
    # Wheel 1
    tfs1 = wheel1.get_tf(wheel1.links[-1])
    tf = IDENTITY_TF
    for tf_ in tfs1[:-1]:
        tf = tf @ tf_[1]
    steeringpoint1 = tf[:2, 3]
    wheelaxis1 = ((tf @ tfs1[-1][1])[:3, :3] @ wheel1.joints[-1].axis)[:2]
    # Wheel 2
    tfs2 = wheel2.get_tf(wheel2.links[-1])
    tf = IDENTITY_TF
    for tf_ in tfs2[:-1]:
        tf = tf @ tf_[1]
    steeringpoint2 = tf[:2, 3]
    wheelaxis2 = ((tf @ tfs2[-1][1])[:3, :3] @ wheel2.joints[-1].axis)[:2]
    # Define a new frame that has a origin on the first steering point and the x axis passes through the second
    psi = angle_between2D(X_AXIS[:2], steeringpoint2 - steeringpoint1)
    matrix = np.array(
        [
            [np.cos(psi), -np.sin(psi), steeringpoint1[0]],
            [np.sin(psi), np.cos(psi), steeringpoint1[1]],
            [0, 0, 1],
        ]
    )
    matrix_inv = np.linalg.pinv(matrix)
    # define the steering points in the new reference frame
    P1 = matrix_inv @ np.array([steeringpoint1[0], steeringpoint1[1], 1])
    P2 = matrix_inv @ np.array([steeringpoint2[0], steeringpoint2[1], 1])
    # get the distance between the two points
    x = P2[0] - P1[1]
    if x != 0:
        # get the zero angle of the wheels
        theta01 = angle_between2D(X_AXIS[:2], wheelaxis1)
        theta02 = angle_between2D(X_AXIS[:2], wheelaxis2)
        # get d, if both are zero, the solution is simpler
        if wheel1.params["d"] == 0 and wheel2.params["d"] == 0:
            theta = np.pi / 2
            solutions += [wrap_zerotwopi(psi + theta - theta01 - np.pi / 2)]
        # if we have at least one d different than zero
        else:
            d1 = wheel1.params["d"]
            d2 = wheel2.params["d"]
            # if the difference between the two "d" is greater than the distance between the wheels there aren't solutions
            if abs(d2 - d1) < x:
                theta = np.arccos(-(d2 - d1) / x)
                solutions += [wrap_zerotwopi(psi + theta - theta01 - np.pi / 2)]
                solutions += [wrap_zerotwopi(psi + theta - theta01 + np.pi / 2)]
            else:
                return solutions

    return solutions
