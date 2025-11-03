# Copyright (C) 2021 Bosch LLC
# Simplified, robust Panda pick-insert node with guards and clear state machine.
# Key fixes:
# - Strict IK validity checks before publishing
# - Controller joint-dimension contract enforcement (7 vs 9)
# - Relaxed 'reached' predicate (position only) with NaN checks
# - Deterministic state transitions with explicit logging
# - Non-blocking Gazebo spawn and consistent reference frames

import enum
import copy
import math
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.task import Future

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnEntity, SetEntityState

# Local model/helpers
from .scripts.models.panda import Panda, FingersAction
from .helpers.rviz_helper import RVizHelper

MODEL_DATABASE_TEMPLATE = """\
<sdf version="1.4">
  <world name="default">
    <include>
      <uri>model://{}</uri>
      <static>{}</static>
    </include>
  </world>
</sdf>"""

class State(enum.Enum):
    HOME = 0
    HOVER = 1
    GRAB = 2
    DELIVER = 3

def _quat_y_deg(deg: float):
    q = R.from_euler(seq="y", angles=deg, degrees=True).as_quat()
    return q[0], q[1], q[2], q[3]

class PandaPickAndInsert(Node):
    def __init__(self):
        super().__init__('panda')

        # -------------------- Parameters --------------------
        self.declare_parameters(
            namespace='',
            parameters=[
                ('control_dt', 0.01),
                ('joint_controller_name', None),
                ('joint_control_topic', '/panda_arm_controller/joint_commands'),
                ('end_effector_target_topic', '/panda/ee_target'),
                ('end_effector_pose_topic', '/panda/ee_pose'),
                ('model_file', None),
                ('base_frame', 'panda_link0'),
                ('end_effector_frame', 'panda_link8'),
                ('arm_joint_tag', None),
                ('finger_joint_tag', None),
                ('initial_joint_angles', None),
                ('share_dir', None),

                # New safety/behavior params
                ('publish_fingers', True),          # set False if controller is 7-DOF only
                ('reference_frame_for_spawns', ''), # '' → use base_frame; or set 'world'
                ('max_wait_ticks', 200),
                ('reach_pos_tol', 0.05),            # [m]
                ('ik_retry_on_fail', True),
            ]
        )

        self._control_dt = float(self.get_parameter('control_dt').value)
        self._pub_fingers = bool(self.get_parameter('publish_fingers').value)
        self._max_wait = int(self.get_parameter('max_wait_ticks').value)
        self._reach_tol = float(self.get_parameter('reach_pos_tol').value)

        self._base_frame = self.get_parameter('base_frame').value
        self._spawn_ref_frame = self.get_parameter('reference_frame_for_spawns').value or self._base_frame

        # -------------------- Interfaces --------------------
        self._pub_cmd = self.create_publisher(
            Float64MultiArray, self.get_parameter('joint_control_topic').value, 10)
        self._pub_ee_target = self.create_publisher(
            Odometry, self.get_parameter('end_effector_target_topic').value, 10)
        self._pub_ee_pose = self.create_publisher(
            Odometry, self.get_parameter('end_effector_pose_topic').value, 10)

        self._sub_js = self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10)

        # Gazebo services
        self._spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self._set_state_cli = self.create_client(SetEntityState, '/set_entity_state')

        # -------------------- Robot model --------------------
        self._panda = Panda(self)
        self._num_joints = self._panda.num_joints  # likely 9 including fingers

        # Joint state cache
        self._js: JointState = JointState()
        self._js.position = list(self._panda.reset_model())
        self._js.velocity = [0.0] * self._num_joints
        self._js.effort = [0.0] * self._num_joints

        # Initial command publish to wake controllers
        self._joint_targets: List[float] = list(self._js.position)
        self._publish_cmd(self._joint_targets)

        # -------------------- EE targets --------------------
        self._ee_current: Odometry = self._panda.solve_fk(self._js, remap=False)
        self._ee_target: Odometry = copy.deepcopy(self._ee_current)
        qx, qy, qz, qw = _quat_y_deg(90.0)
        self._ee_target.pose.pose.orientation.x = qx
        self._ee_target.pose.pose.orientation.y = qy
        self._ee_target.pose.pose.orientation.z = qz
        self._ee_target.pose.pose.orientation.w = qw
        self._ee_target_initial = copy.deepcopy(self._ee_target)

        # Seed IK once
        jt0 = self._safe_ik(self._ee_target)
        if jt0 is not None:
            self._joint_targets = jt0

        # RViz helper
        self._rviz = RVizHelper(self)

        # -------------------- State machine --------------------
        self._state = State.HOME
        self._wait = 0

        # Sparkplug spawn bookkeeping
        self._sparkplug_pose = Pose()
        self._sparkplug_pose.position.x = 0.40
        self._sparkplug_pose.position.y = 0.00
        self._sparkplug_pose.position.z = 0.01
        self._sparkplug_pose.orientation.w = 1.0

        self._sparkplug_counter = 0
        self._spawn_req = SpawnEntity.Request()
        self._spawn_req.reference_frame = self._spawn_ref_frame

        self.get_logger().info(f"Spawn reference frame: {self._spawn_ref_frame}")

    # -------------------- Utilities --------------------
    def _publish_cmd(self, joints: List[float]) -> None:
        """Publish joint command matching the controller dimension."""
        if not self._pub_fingers:
            cmd = joints[:7]
        else:
            cmd = joints[:self._num_joints]
        msg = Float64MultiArray()
        msg.data = list(map(float, cmd))
        self._pub_cmd.publish(msg)

    def _safe_ik(self, ee_odom: Odometry) -> Optional[List[float]]:
        """Call IK and validate result."""
        jt = self._panda.solve_ik(ee_odom)
        if jt is None:
            self.get_logger().warn("IK failed: None")
            return None
        arr = np.asarray(jt, dtype=float)
        if arr.shape[0] != self._num_joints:
            self.get_logger().warn(f"IK bad length {arr.shape[0]} != {self._num_joints}")
            return None
        if not np.all(np.isfinite(arr)):
            self.get_logger().warn("IK contains non-finite values")
            return None
        return list(arr)

    def _end_effector_reached(self, tol: float) -> bool:
        """Position-only reach check with NaN guards."""
        p = np.array([
            self._ee_current.pose.pose.position.x,
            self._ee_current.pose.pose.position.y,
            self._ee_current.pose.pose.position.z], dtype=float)
        t = np.array([
            self._ee_target.pose.pose.position.x,
            self._ee_target.pose.pose.position.y,
            self._ee_target.pose.pose.position.z], dtype=float)
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(t))):
            return False
        return np.linalg.norm(p - t) < tol

    def _spawn_once(self):
        """Spawn socket on first call, then spawn a sparkplug at sampled pose."""
        # 1) socket fixture on very first run
        if self._sparkplug_counter == 0:
            req = copy.deepcopy(self._spawn_req)
            req.xml = MODEL_DATABASE_TEMPLATE.format("sparkplug_socket", "1")
            req.name = "sparkplug_socket"
            req.initial_pose = Pose()
            req.initial_pose.position.x = 0.25
            req.initial_pose.position.z = 0.20
            if not self._spawn_cli.service_is_ready():
                self._spawn_cli.wait_for_service()
            _ = self._spawn_cli.call_async(req)

        # 2) sample a new sparkplug pose near the robot
        rp = np.random.uniform(low=[0.50, -0.20], high=[0.60, 0.20])
        self._sparkplug_pose.position.x = float(rp[0])
        self._sparkplug_pose.position.y = float(rp[1])
        self._sparkplug_pose.position.z = 0.01
        qx, qy, qz, qw = _quat_y_deg(90.0)
        self._sparkplug_pose.orientation.x = qx
        self._sparkplug_pose.orientation.y = qy
        self._sparkplug_pose.orientation.z = qz
        self._sparkplug_pose.orientation.w = qw

        # 3) spawn sparkplug
        req = copy.deepcopy(self._spawn_req)
        req.xml = MODEL_DATABASE_TEMPLATE.format("sparkplug", "0")
        req.name = f"sparkplug{self._sparkplug_counter}"
        req.initial_pose = copy.deepcopy(self._sparkplug_pose)
        if not self._spawn_cli.service_is_ready():
            self._spawn_cli.wait_for_service()
        _ = self._spawn_cli.call_async(req)

        self._sparkplug_counter += 1

    # -------------------- Core callbacks --------------------
    def _on_joint_states(self, js: JointState):
        # Update cache
        if not js.position:
            return
        self._js = js
        # FK in controller frame
        self._panda.set_joint_states(self._js)
        self._ee_current = self._panda.solve_fk(self._js)

        # State machine tick
        self._tick_state_machine()

        # Publish target visualization
        self._pub_ee_target.publish(self._ee_target)
        self._pub_ee_pose.publish(self._ee_current)
        self._rviz.publish(self._ee_current)

        # Drive controller
        self._publish_cmd(self._joint_targets)

    # -------------------- State machine --------------------
    def _tick_state_machine(self):
        HOVER_Z = 0.20
        GOAL_Z = 0.27  # tray surface + plug height approx
        GRAB_Z = 0.05

        # keep a consistent gripper policy
        if self._state in (State.DELIVER, State.GRAB):
            self._joint_targets[-2:] = self._panda.move_fingers(self._joint_targets, FingersAction.CLOSE)[-2:]
        else:
            self._joint_targets[-2:] = self._panda.move_fingers(self._joint_targets, FingersAction.OPEN)[-2:]

        # Orientation kept at +Y 90 deg
        qx, qy, qz, qw = _quat_y_deg(90.0)
        self._ee_target.pose.pose.orientation.x = qx
        self._ee_target.pose.pose.orientation.y = qy
        self._ee_target.pose.pose.orientation.z = qz
        self._ee_target.pose.pose.orientation.w = qw

        if self._state == State.HOME:
            if self._wait < self._max_wait:
                self._wait += 1
                return
            self._wait = 0
            # Spawn new parts and set hover over sparkplug
            self._spawn_once()
            self._ee_target.pose.pose.position = copy.deepcopy(self._sparkplug_pose.position)
            self._ee_target.pose.pose.position.x += 0.012
            self._ee_target.pose.pose.position.z = HOVER_Z
            jt = self._safe_ik(self._ee_target)
            if jt is None:
                if self.get_parameter('ik_retry_on_fail').value:
                    return  # keep trying next tick
                else:
                    return
            self._joint_targets = jt
            self._state = State.HOVER
            self.get_logger().info("STATE → HOVER")
            return

        if self._state == State.HOVER:
            # descend to GRAB_Z once reached hover
            if self._end_effector_reached(self._reach_tol):
                self._ee_target.pose.pose.position.z = GRAB_Z
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
                    self._state = State.GRAB
                    self._wait = 0
                    self.get_logger().info("STATE → GRAB")
            return

        if self._state == State.GRAB:
            # after short dwell, lift to hover*2
            self._wait += 1
            if self._wait == self._max_wait:
                self._ee_target.pose.pose.position.z = 2.0 * HOVER_Z
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
            if self._wait >= 2 * self._max_wait:
                # set delivery XY based on index; keep Z high
                idx = self._sparkplug_counter - 1
                dx = -52.5e-3 + 35e-3 * (idx % 4)
                dy = -52.5e-3 + 35e-3 * ((idx // 4) % 4)
                self._ee_target.pose.pose.position.x = 0.25 + dx
                self._ee_target.pose.pose.position.y = 0.00 + dy
                self._ee_target.pose.pose.position.z = 0.20 + 2.0 * HOVER_Z
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
                    self._state = State.DELIVER
                    self._wait = 0
                    self.get_logger().info("STATE → DELIVER")
            return

        if self._state == State.DELIVER:
            # approach down to GOAL_Z, release, and go back home
            if self._wait == 0 and self._end_effector_reached(self._reach_tol):
                self._ee_target.pose.pose.position.x += 0.01  # small approach
                self._ee_target.pose.pose.position.z = GOAL_Z
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
            # dwell and retract
            self._wait += 1
            if self._wait == self._max_wait:
                self._ee_target.pose.pose.position.z = GOAL_Z + 0.02
                jt = self._safe_ik(self._ee_target)
                if jt is not None:
                    self._joint_targets = jt
            if self._wait >= 3 * self._max_wait:
                # back to home target
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
    node = PandaPickAndInsert()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
