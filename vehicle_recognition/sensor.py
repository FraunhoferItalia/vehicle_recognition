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
from enum import Enum
from urdf_parser import Urdf, Link, Joint, Chain
import numpy as np


class SensorType(Enum):
    IMU = "imu"
    LASER = "laser"
    POINTCLOUD = "pointcloud"
    CAMERA = "camera"
    GYRO = "gyro"
    UNDEFINED = "undefined"

    def __str__(self):
        return str(self.value)


class Sensor(Chain):
    name: str
    type: SensorType

    def __init__(self, chain: Chain):
        self.name = chain.links[-1].name
        self.type = self._define_type()
        self.links = chain.links
        self.joints = chain.joints
        self.parent = chain.parent
        self.subchains = chain.subchains

    def _define_type(self) -> SensorType:
        types_found: List[bool] = [
            SensorType.IMU.__str__() in self.name,
            SensorType.LASER.__str__() in self.name,
            SensorType.POINTCLOUD.__str__() in self.name,
            SensorType.CAMERA.__str__() in self.name,
            SensorType.GYRO.__str__() in self.name,
        ]
        if sum(types_found) == 1:
            if types_found[0] == True:
                return SensorType.IMU
            if types_found[1] == True:
                return SensorType.LASER
            if types_found[2] == True:
                return SensorType.POINTCLOUD
            if types_found[3] == True:
                return SensorType.CAMERA
            if types_found[4] == True:
                return SensorType.GYRO
        else:
            return SensorType.UNDEFINED

    def discovery_from_urdf(urdf: Urdf) -> List[Sensor]:
        sensors: List[Sensor] = []
        for chain in urdf.root_chains:
            assumptions = Sensor._sensor_assumptions(chain)
            if all(assumptions):
                sensors.append(Sensor(chain))
        if len(sensors) > 0:
            print(f"Found {len(sensors)} possible sensors: ")
            for sensor in sensors:
                print(f" - {sensor.name} of {sensor.type.__str__().upper()} type")
        return sensors

    def _sensor_assumptions(chain: Chain) -> List[bool]:
        assumptions: List[bool] = []

        # it has only 1 joint
        assumptions.append(len(chain.joints) == 1)

        # the joint must be fixed
        assumptions.append(chain.joints[0].type == "fixed")

        return assumptions
