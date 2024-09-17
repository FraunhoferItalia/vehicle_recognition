# vehicle_recognition

This Python library allows the recognition of a vehicle from its URDF model. The library analyzes the vehicle's mobility and perception modules and computes the available kinematic configurations. The considered vehicle kinematic types are Car-like, Differential, and Omni-like.

## Software Architecture

```kroki-plantuml
@startuml

class Wheel {
  name
  type
  params
  radius
  --
  __init__(chain)
  compute_params(frame)
  origin_tf()
  --
  discovery_from_urdf(urdf)
  _wheel_assumptions(Chain)
}
Wheel -|> Chain
Chain "1..n"*-- Link
Chain "1..n"*-- Joint

class ControlFrame {
  name 
  tf
}
class VehicleKinematicsType {
  OMNIDIRECTIONAL
  CAR-LIKE
  DIFFERENTIAL
}
class VehicleKinematics {
  type: VehicleKinematicsType
  control_frames: List[VehicleKinematic]
}

VehicleKinematics -d-> ControlFrame
VehicleKinematics -d-> VehicleKinematicsType

object compute_wheels_common_axis

object search_omnidirectional_kinematics
object search_differential_kinematics
object search_carlike_kinematics

search_differential_kinematics -u-> compute_wheels_common_axis
search_carlike_kinematics -u-> compute_wheels_common_axis

class Vehicle {
  urdf
  wheels
  kinematics
  wheels_not_correctly_defined  
  __init__(path_to_urdf)
  study_kinematics()
}

Vehicle "1..n"*-- Wheel
Vehicle "1..n"*-- VehicleKinematics
Vehicle -u-> search_omnidirectional_kinematics
Vehicle -u-> search_differential_kinematics
Vehicle -u-> search_carlike_kinematics

@enduml
```

### Licence

vehicle_recognition is licensed under the terms of the Apache License 2.0. The project has received financial support by the Horizon 2020 EU Project [CONCERT](https://concertproject.eu/).