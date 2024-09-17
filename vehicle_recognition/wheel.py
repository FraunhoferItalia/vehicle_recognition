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
from typing import List, Dict, Tuple
from enum import Enum
from urdf_parser import Urdf, Link, Joint, Chain
from urdf_parser.utils import recurse_subtree
import numpy as np
import math
from .utils import (
    angle_between,
    angle_between2D,
    angle_between_axes,
    EPSILON,
    X_AXIS,
    Y_AXIS,
    Z_AXIS,
    IDENTITY_TF,
)


class WheelType(Enum):
    FIXED = 1
    STEERING = 2
    UNDEFINED = 3


class Wheel(Chain):
    name: str
    type: WheelType
    params: Dict[str, float] = []
    radius: float
    clockwise_steer: bool = True

    def __init__(self, chain: Chain):
        self.name = chain.links[-1].name
        self.links = chain.links
        self.joints = chain.joints
        self.parent = chain.parent
        self.subchains = chain.subchains
        try:
            self.radius = float(
                recurse_subtree(chain.links[-1].xml, ["radius"]).attrib["value"]
            )
        except AttributeError:
            self.radius = None

        self.type = self._define_type()

        self.params = self.compute_params(IDENTITY_TF)

    def _define_type(self) -> WheelType:
        if len(self.joints) == 1:
            return WheelType.FIXED
        elif len(self.joints) == 2:
            if self.joints[0].type in ("continuous", "revolute"):
                # steering link tf
                steering_link_tf = np.identity(4)
                for tf in self.get_tf(self.links[1]):
                    steering_link_tf = steering_link_tf @ tf[1]
                # steering axis must be parallel to the vertical axis
                steering_joint_axis = steering_link_tf[:3, :3] @ self.joints[0].axis

                if angle_between_axes(Z_AXIS, steering_joint_axis) < EPSILON:
                    if not angle_between(Z_AXIS, steering_joint_axis) < EPSILON:
                        self.clockwise_steer = False
                    if (
                        self.joints[0].limits.upper - self.joints[0].limits.lower
                        >= np.pi
                    ):
                        return WheelType.STEERING
                    else:
                        return WheelType.UNDEFINED

        return WheelType.UNDEFINED

    @property
    def origin_tf(self) -> np.ndarray:
        tf = IDENTITY_TF
        for tf_ in self.get_tf(self.links[-1]):
            tf = tf @ tf_[1]
        return tf

    def reset(self) -> None:
        for j in self.joints:
            j.set_value(0.0)

    def compute_params(self, frame: np.ndarray = IDENTITY_TF) -> Dict[str, float]:
        self.reset()
        params: Dict[str, float] = {}
        if not self.type is WheelType.UNDEFINED:
            frame_inv = np.linalg.inv(frame)
            first_joint_center = (frame_inv @ self.joints[0].base_tf)[:2, 3]
            len = float(np.linalg.norm(first_joint_center))
            if len < float("1e-10"):
                first_joint_center = frame[0, :2]
                len = 0.0
            params["len"] = len
            alpha = float(math.atan2(first_joint_center[1], first_joint_center[0]))
            if alpha < 0:
                alpha = 2 * np.pi + alpha
            params["alpha"] = alpha
            if self.type is WheelType.STEERING:
                params["d"] = float(np.linalg.norm(self.joints[1].base_tf[:2, 3]))
                second_joint_axis = (
                    (frame_inv @ self.joints[0].base_tf @ self.joints[1].base_tf)[
                        :3, :3
                    ]
                    @ self.joints[1].axis
                )[:2]
            elif self.type is WheelType.FIXED:
                second_joint_axis = (
                    (frame_inv @ self.joints[0].base_tf)[:3, :3] @ self.joints[-1].axis
                )[:2]
            beta0 = float(angle_between2D(first_joint_center, second_joint_axis))
            params["clockwise_steer"] = self.clockwise_steer
            if self.clockwise_steer:
                params["beta0"] = beta0
            else:
                params["beta0"] = 2 * np.pi - beta0
        return params

    def discovery_from_urdf(urdf: Urdf) -> List[Wheel]:
        wheels: List[Wheel] = []
        for chain in urdf.root_chains:
            assumptions = Wheel._wheel_assumptions(chain)
            if all(assumptions):
                wheels.append(Wheel(chain))

        if len(wheels) > 0:
            print(f"Found {len(wheels)} possible wheels: ")
            for wheel in wheels:
                print(f" - {wheel.name}")

        return wheels

    def _wheel_assumptions(chain: Chain) -> List[bool]:
        assumptions: List[bool] = []

        # last joint must be continuous
        assumptions.append(chain.joints[-1].type in ("continuous", "revolute"))

        # at least the last joint and link must have the world "wheel" inside the name
        assumptions.append("wheel" in chain.joints[-1].name)
        assumptions.append("wheel" in chain.links[-1].name)

        # last joint axis of rotation must be orthogonal to the vertical axis
        last_joint_tf = np.identity(4)
        for tf in chain.get_tf(chain.links[-1]):
            last_joint_tf = last_joint_tf @ tf[1]
        assumptions.append(
            np.dot(Z_AXIS, (last_joint_tf[:3, :3] @ chain.joints[-1].axis)) <= EPSILON
        )

        return assumptions
