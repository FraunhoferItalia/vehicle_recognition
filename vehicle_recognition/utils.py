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
import numpy as np
from scipy.spatial.transform import Rotation
import math

EPSILON: float = 1e-5
X_AXIS: np.ndarray = np.array([1, 0, 0])
Y_AXIS: np.ndarray = np.array([0, 1, 0])
Z_AXIS: np.ndarray = np.array([0, 0, 1])
IDENTITY_TF = np.identity(4)


def wrap_zerotwopi(angle: float):
    return angle % (2 * np.pi)


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def get_euler_angles(tf: np.ndarray) -> np.ndarray:
    rotation_matrix = Rotation.from_matrix(tf[:3, :3])
    angles = rotation_matrix.as_euler("xyz", degrees=False)
    return angles


# gets the shortest angle between two vectors, always positive
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# gets the angle between two 2D vectors, it depends on the order
# the angle is always positive and goes from 0 to 2*pi
def angle_between2D(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = np.dot(v1, v2)
    det = np.linalg.det(np.concatenate((v1.reshape(2, 1), v2.reshape(2, 1)), axis=1))
    angle = np.arctan2(det, dot)
    if angle < 0:
        angle = 2 * np.pi + angle
    return angle


def angle_between_axes(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    angle: float = angle_between(v1, v2)
    if angle > np.pi / 2 and angle <= (3 * np.pi) / 2:
        return abs(np.pi - angle)
    return angle


def from_vec_point_to_abc_coeffs(
    xy_point: np.ndarray, xy_vec: np.ndarray
) -> List[float]:
    a: float = -xy_vec[1]
    b: float = xy_vec[0]
    c: float = xy_vec[1] * xy_point[0] - xy_vec[0] * xy_point[1]
    return [a, b, c]


def distance_line_point(abc_coeffs: List[float], xy_point: np.ndarray):
    [a, b, c] = abc_coeffs
    numerator = abs(a * xy_point[0] + b * xy_point[1] + c)
    denominator = math.sqrt(a**2 + b**2)

    return numerator / denominator


def lines_intersection(
    abc_coeffs1: List[float], abc_coeffs2: List[float]
) -> np.ndarray:
    [a1, b1, c1] = abc_coeffs1
    [a2, b2, c2] = abc_coeffs2
    denominator = a1 * b2 - a2 * b1
    # not parallel
    if denominator != 0:
        return np.array(
            [(b1 * c2 - b2 * c1) / denominator, (c1 * a2 - c2 * a1) / denominator]
        )

    return None


def points_center(points: List[np.ndarray]) -> np.ndarray:
    return np.mean(points, axis=0)
