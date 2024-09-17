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
from typing import List, Tuple
from dataclasses import field
from urdf_parser import Urdf, Link, Joint, Chain
import numpy as np
from .wheel import Wheel, WheelType
from .sensor import Sensor, SensorType
from .vehicle_analysis_utils import (
    VehicleKinematic,
    VehicleKinematicType,
    search_omnidirectional_kinematics,
    search_differential_kinematics,
    search_carlike_kinematics,
)

ACCEPTABLE_RADIUS_ERROR: float = 0.02


class Vehicle:
    urdf: Urdf = None
    root: Link = None
    wheels: List[Wheel] = field(default_factory=list)
    sensors: List[Sensor] = field(default_factory=list)
    kinematics: List[VehicleKinematic] = field(default_factory=list)
    wheels_not_considered: List[Wheel] = field(default_factory=list)
    chosen_kinematic: Tuple[VehicleKinematicType, VehicleKinematic.Configuration] = None

    def __init__(self, path_to_urdf: Urdf, ignore_wheels_radius: bool = False):
        print("------------------------------------- ")
        print("Recognizing the vehicle structure ... ")
        self.urdf = Urdf.init_from_file(path_to_urdf)
        self.urdf.merge_fixed_joints()

        self.sensors = Sensor.discovery_from_urdf(self.urdf)
        self.wheels = Wheel.discovery_from_urdf(self.urdf)
        self._check_discovered_wheels(ignore_wheels_radius)
        self.kinematics = []
        print("Vehicle modules recognition ended! ")
        print("------------------------------------- ")

    def _check_discovered_wheels(self, ignore_wheels_radius: bool = False) -> None:
        self.wheels_not_considered = []
        if len(self.wheels) > 0:
            # remove Wheels with UNDEFINED TYPES
            not_considered = 0
            for wheel in reversed(self.wheels):
                if wheel.type is WheelType.UNDEFINED:
                    self.wheels_not_considered.append(wheel)
                    self.wheels.remove(wheel)
                    not_considered += 1
            if not_considered > 0:
                print("Removed ", not_considered, " wheels, because of UNDEFINED type")

        if len(self.wheels) > 0 and not ignore_wheels_radius:
            positions: List[float] = []
            for wheel in self.wheels:
                if wheel.radius is None:
                    print("Wheel radius not specified in the urdf")
                    try:
                        wheel.radius = float(input("Specify the value in meters: "))
                    except ValueError:
                        wheel.radius = 0
                z = wheel.origin_tf[2, 3]
                positions += [z - wheel.radius]

            # the most common heigh is the floor heigh
            floor_height = min(positions)
            for i, wheel in enumerate(reversed(self.wheels)):
                if (
                    positions[i] <= floor_height - ACCEPTABLE_RADIUS_ERROR
                    or positions[i] >= floor_height + ACCEPTABLE_RADIUS_ERROR
                ):
                    print(
                        f"The found wheel {wheel.name} and radius {wheel.radius} does not seem to touch the floor",
                    )
                    self.wheels_not_considered.append(wheel)
                    self.wheels.remove(wheel)

        if len(self.wheels) > 0:
            root_names: List[str] = []
            for wheel in self.wheels:
                root_names += [wheel.get_root().name]
            root_name = max(set(root_names), key=root_names.count)
            for wheel in reversed(self.wheels):
                if not root_name is wheel.get_root().name:
                    self.wheels_not_considered.append(wheel)
                    self.wheels.remove(wheel)
            if len(self.wheels) > 0:
                self.root = self.wheels[0].get_root()
            else:
                print("No vehicle root frame can be found")

        return

    def study_kinematics(self, autonomous: bool = True) -> VehicleKinematic:
        print("------------------------------------- ")
        print("Starting the vehicle kinematics recognition ... ")
        print("------ CAR-LIKE KINEMATICS STUDY ------")
        carlike_kinematic = search_carlike_kinematics(self.wheels)
        if not carlike_kinematic is None:
            self.kinematics += [carlike_kinematic]
            self.chosen_kinematic = (
                carlike_kinematic.type,
                carlike_kinematic.best_configuration,
            )
        else:
            print("No possible vehicle carlike kinematics found")
        print("------ DIFFERENTIAL KINEMATICS STUDY ------")
        differential_kinematic = search_differential_kinematics(self.wheels)
        if not differential_kinematic is None:
            self.kinematics += [differential_kinematic]
            self.chosen_kinematic = (
                differential_kinematic.type,
                differential_kinematic.best_configuration,
            )
        else:
            print("No possible vehicle differential kinematics found")
        print("------ OMNIDIRECTIONAL KINEMATICS STUDY ------")
        omnidirectional_kinematic = search_omnidirectional_kinematics(
            self.wheels, autonomous
        )
        if not omnidirectional_kinematic is None:
            self.kinematics += [omnidirectional_kinematic]
            self.chosen_kinematic = (
                omnidirectional_kinematic.type,
                omnidirectional_kinematic.best_configuration,
            )
        else:
            print("No possible vehicle omnidirectional kinematics found")

        print("Vehicle kinematics recognition ended! ")
        print("------------------------------------- ")
        return

    def choose_kinematic_configuration(
        self, kinematic_type: VehicleKinematicType, name: str = ""
    ) -> bool:
        for kinematic in self.kinematics:
            if kinematic_type == kinematic.type:
                if name != "":
                    for configuration in kinematic.configurations:
                        if configuration.name == name:
                            self.chosen_kinematic = (kinematic_type, configuration)
                            return True
                    self.chosen_kinematic = (
                        kinematic_type,
                        kinematic.best_configuration,
                    )
                else:
                    self.chosen_kinematic = (
                        kinematic_type,
                        kinematic.best_configuration,
                    )
                    return True

        return False

    def estimate_footprint(self, padding: float = 0.0) -> List[List[float]]:
        padding = abs(padding)
        x = []
        y = []
        for wheel in self.wheels:
            import numpy as np

            wheel.reset()
            wheel_tf = wheel.origin_tf
            x += [wheel_tf[0, 3]]
            y += [wheel_tf[1, 3]]
        max_x = max(x)
        min_x = min(x)
        max_y = max(y)
        min_y = min(y)
        footprint = [
            [max_x + padding, max_y + padding],
            [min_x - padding, max_y + padding],
            [min_x - padding, min_y - padding],
            [max_x + padding, min_y - padding],
        ]

        return footprint
