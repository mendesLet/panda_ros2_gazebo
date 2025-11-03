# Fixed Panda pick-and-place with proper Gazebo spawn and robust state machine.

import enum
import copy
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity

from .scripts.models.panda import Panda, FingersAction
from .helpers.rviz_helper import RVizHelper


def sdf_unit_box(name: str, size: float, mass: float = 0.05) -> str:
    # Primitive SDF model. No dependency on GAZEBO_MODEL_PATH.
    ix = iy = iz = (1.0/12.0) * mass * (2*(size**2))  # rough inertia for a cube
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='{name}'>
    <static>false</static>
    <link name='link'>
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ix}</ixx><iyy>{iy}</iyy><izz>{iz}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name='col'>
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface>
      </collision>
      <visual name='vis'>
        <geometry><box><size>{size} {size} {size}</size></box></geometry>
        <material>
          <ambient>0.7 0.5 0.3 1</ambient>
          <diffuse>0.7 0.5 0.3 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


class State(enum.Enum):
    HOME = 0
    HOVER = 1
    GRAB = 2
    DELIVER = 3


def quat_y(deg: float):
    q = R.from_euler("y", deg, degrees=True).as_quat()
    return q[0], q[1], q[2], q[3]


class PandaPickAndPlace(Node):
    def __init__(self):
        super().__init__("panda")

        # Parameters with safe defaults
        self.declare_parameters(
            "",
            [
                ("control_dt", 0.01),
                ("joint_control_topic", "/panda_arm_controller/joint_commands"),
                ("end_effector_target_topic", "/panda/ee_target"),
                ("end_effector_pose_topic", "/panda/ee_pose"),
                ("base_frame", "panda_link0"),
                ("end_effector_frame", "panda_link8"),
                ("publish_fingers", True),            # set False for 7-DOF controllers
                ("reach_pos_tol", 0.05),
                ("max_wait_ticks", 200),
                ("spawn_reference_frame", "world"),   # use 'world' unless you publish TF to base
                ("cube_size", 0.05),
                ("share_dir", None),
                ("model_file", None),
                ("arm_joint_tag", None),
                ("finger_joint_tag", None),
                ("initial_joint_angles", None),
            ],
        )

        self._dt = float(self.get_parameter("control_dt").value)
        self._pub_fingers = bool(self.get_parameter("publish_fingers").value)
        self._tol = float(self.get_parameter("reach_pos_tol").value)
        self._max_wait = int(self.get_parameter("max_wait_ticks").value)
        self._base_frame = self.get_parameter("base_frame").value
        self._spawn_ref = self.get_parameter("spawn_reference_frame").value
        self._cube_size = float(self.get_parameter("cube_size").value)

        # Publishers/subscribers
        self._pub_cmd = self.create_publisher(Float64MultiArray, self.get_parameter("joint_control_topic").value, 10)
        self._pub_target = self.create_publisher(Odometry, self.get_parameter("end_effector_target_topic").value, 10)
        self._pub_pose = self.create_publisher(Odometry, self.get_parameter("end_effector_pose_topic").value, 10)
        self._sub_js = self.create_subscription(JointState, "/joint_states", self._on_js, 10)

        # Gazebo spawn service
        self._spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")

        # Robot model and initial state
        self._panda = Panda(self)
        self._nj = self._panda.num_joints
        self._js = JointState()
        self._js.position = list(self._panda.reset_model())
        self._js.velocity = [0.0] * self._nj
        self._js.effort = [0.0] * self._nj

        self._joint_targets: List[float] = list(self._js.position)
        self._publish_cmd(self._joint_targets)

        self._ee_current: Odometry = self._panda.solve_fk(self._js, remap=False)
        self._ee_target: Odometry = copy.deepcopy(self._ee_current)
        qx, qy, qz, qw = quat_y(90.0)
        self._ee_target.pose.pose.orientation.x = qx
        self._ee_target.pose.pose.orientation.y = qy
        self._ee_target.pose.pose.orientation.z = qz
        self._ee_target.pose.pose.orientation.w = qw
        self._ee_target_initial = copy.deepcopy(self._ee_target)

        jt0 = self._safe_ik(self._ee_target)
        if jt0 is not None:
            self._joint_targets = jt0

        self._rviz = RVizHelper(self)

        # State machine
        self._state = State.HOME
        self._wait = 0

        # Cube bookkeeping
        self._cube_pose = Pose()
        self._cube_pose.position.x = 0.5
        self._cube_pose.position.y = 0.0
        self._cube_pose.position.z = self._cube_size / 2.0
        self._cube_pose.orientation.w = 1.0

        self._cube_counter = 0

        self.get_logger().info(f"Spawn reference frame: {self._spawn_ref}")

    # ---------- Helpers ----------

    def _publish_cmd(self, joints: List[float]) -> None:
        cmd = joints[:7] if not self._pub_fingers else joints[: self._nj]
        msg = Float64MultiArray()
        msg.data = list(map(float, cmd))
        self._pub_cmd.publish(msg)

    def _safe_ik(self, ee: Odometry) -> Optional[List[float]]:
        jt = self._panda.solve_ik(ee)
        if jt is None:
            self.get_logger().warn("IK failed: None")
            return None
        arr = np.asarray(jt, dtype=float)
        if arr.shape[0] != self._nj:
            self.get_logger().warn(f"IK bad length {arr.shape[0]} != {self._nj}")
            return None
        if not np.all(np.isfinite(arr)):
            self.get_logger().warn("IK non-finite")
            return None
        return list(arr)

    def _reached(self, tol: float) -> bool:
        p = np.array(
            [
                self._ee_current.pose.pose.position.x,
                self._ee_current.pose.pose.position.y,
                self._ee_current.pose.pose.position.z,
            ],
            dtype=float,
        )
        t = np.array(
            [
                self._ee_target.pose.pose.position.x,
                self._ee_target.pose.pose.position.y,
                self._ee_target.pose.pose.position.z,
            ],
            dtype=float,
        )
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(t))):
            return False
        return np.linalg.norm(p - t) < tol

    def _spawn_cube(self):
        # Build SDF at call time. Independent of model database.
        name = f"cube_{self._cube_counter}"
        xml = sdf_unit_box(name, size=self._cube_size)

        req = SpawnEntity.Request()
        req.name = name
        req.xml = xml
        req.robot_namespace = ""  # optional
        req.reference_frame = self._spawn_ref
        req.initial_pose = copy.deepcopy(self._cube_pose)

        if not self._spawn_cli.service_is_ready():
            self._spawn_cli.wait_for_service()

        self._spawn_cli.call_async(req)  # do not block
        self._cube_counter += 1
        self.get_logger().info(f"Spawn requested: {name} at "
                               f"({req.initial_pose.position.x:.3f}, "
                               f"{req.initial_pose.position.y:.3f}, "
                               f"{req.initial_pose.position.z:.3f}) in {self._spawn_ref}")

    # ---------- ROS callbacks ----------

    def _on_js(self, js: JointState):
        if not js.position:
            return
        self._js = js
        self._panda.set_joint_states(self._js)
        self._ee_current = self._panda.solve_fk(self._js)

        self._tick_sm()

        self._pub_target.publish(self._ee_target)
        self._pub_pose.publish(self._ee_current)
        self._rviz.publish(self._ee_current)

        self._publish_cmd(self._joint_targets)

    # ---------- State machine ----------

    def _tick_sm(self):
        HOVER_Z = 0.30
        GRAB_Z = max(0.03, self._cube_size * 0.6)  # safe grasp height
        STACK_X = 0.30
        STACK_Y = 0.50

        # keep gripper policy
        if self._state in (State.GRAB, State.DELIVER):
            self._joint_targets[-2:] = self._panda.move_fingers(self._joint_targets, FingersAction.CLOSE)[-2:]
        else:
            self._joint_targets[-2:] = self._panda.move_fingers(self._joint_targets, FingersAction.OPEN)[-2:]

        # fix orientation
        qx, qy, qz, qw = quat_y(90.0)
        self._ee_target.pose.pose.orientation.x = qx
        self._ee_target.pose.pose.orientation.y = qy
        self._ee_target.pose.pose.orientation.z = qz
        self._ee_target.pose.pose.orientation.w = qw

        if self._state == State.HOME:
            if self._wait < self._max_wait:
                self._wait += 1
                return
            self._wait = 0
            # randomize source pose and spawn
            rp = np.random.uniform(low=[0.35, -0.10], high=[0.65, 0.30])
            self._cube_pose.position.x = float(rp[0])
            self._cube_pose.position.y = float(rp[1])
            self._cube_pose.position.z = self._cube_size / 2.0
            self._spawn_cube()

            # target hover above cube
            self._ee_target.pose.pose.position.x = self._cube_pose.position.x
            self._ee_target.pose.pose.position.y = self._cube_pose.position.y
            self._ee_target.pose.pose.position.z = HOVER_Z
            jt = self._safe_ik(self._ee_target)
            if jt is None:
                return
            self._joint_targets = jt
            self._state = State.HOVER
            self.get_logger().info("STATE → HOVER")
            return

        if self._state == State.HOVER:
            if self._reached(self._tol):
                self._ee_target.pose.pose.position.z = GRAB_Z
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
                    self._state = State.GRAB
                    self._wait = 0
                    self.get_logger().info("STATE → GRAB")
            return

        if self._state == State.GRAB:
            # dwell then lift
            self._wait += 1
            if self._wait == self._max_wait:
                self._ee_target.pose.pose.position.z = HOVER_Z
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
            if self._wait >= 2 * self._max_wait:
                # move over stack area, keep high
                self._ee_target.pose.pose.position.x = STACK_X
                self._ee_target.pose.pose.position.y = STACK_Y
                self._ee_target.pose.pose.position.z = HOVER_Z
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
                    self._state = State.DELIVER
                    self._wait = 0
                    self.get_logger().info("STATE → DELIVER")
            return

        if self._state == State.DELIVER:
            # descend to stack height then return home
            stack_h = max(self._cube_size * (self._cube_counter - 0.5), self._cube_size * 0.5)
            approach_z = stack_h + 0.5 * self._cube_size
            if self._wait == 0 and self._reached(self._tol):
                self._ee_target.pose.pose.position.z = approach_z
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
            self._wait += 1
            if self._wait >= 3 * self._max_wait:
                self._ee_target = copy.deepcopy(self._ee_target_initial)
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
                self._state = State.HOME
                self._wait = 0
                self.get_logger().info("STATE → HOME")
            return


def main(args=None):
    rclpy.init(args=args)
    node = PandaPickAndPlace()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
