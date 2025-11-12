#!/usr/bin/env python3
# panda_task_library_action.py
import copy
import enum
import threading
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor

from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

# Adjust this import to your package name if needed.
# e.g., from manip_task_msgs.action import ManipTask
from panda_task_msgs.action import ManipTask

from .scripts.models.panda import Panda, FingersAction
from .helpers.rviz_helper import RVizHelper


def quat_rpy_deg(rpy_deg: Tuple[float, float, float]):
    r, p, y = np.deg2rad(rpy_deg)
    q = R.from_euler("xyz", [r, p, y]).as_quat()
    return q[0], q[1], q[2], q[3]


class Phase(enum.Enum):
    IDLE = 0
    GO = 1
    PICK_PREOPEN = 2
    PICK_DESCEND = 3
    PICK_DWELL = 4
    PICK_LIFT = 5
    PLACE_DESCEND = 6
    PLACE_OPEN = 7
    PLACE_LIFT = 8


class PandaTaskLibrary(Node):
    def __init__(self):
        super().__init__("panda_task_library")

        self.declare_parameters(
            "",
            [
                ("control_dt", 0.01),
                ("normal_open_width_m", 0.08),
                ("joint_control_topic", "/panda_arm_controller/joint_commands"),
                ("end_effector_target_topic", "/panda/ee_target"),
                ("end_effector_pose_topic", "/panda/ee_pose"),
                ("base_frame", "panda_link0"),
                ("end_effector_frame", "panda_link8"),
                ("publish_fingers", True),
                ("reach_pos_tol", 0.02),
                ("home_rpy_deg", [0.0, 90.0, 0.0]),
                ("home_xyz", [0.45, 0.0, 0.40]),
                ("approach_ticks", 600),
                ("grasp_dwell_ticks", 200),
                ("preclose_offset", 0.005),
                ("max_joint_vel", 0.15),
                ("preopen_ticks", 120),
                ("max_finger_joint_vel", 0.6),
                ("pregrasp_pause_ticks", 30),          # short settle pause at bottom
                ("close_on_actual_z", True),           # use measured EE z to decide when to close
                ("grasp_close_speed_m_per_s", 0.01),   # per-finger speed (m/s) for gentle close
                ("grasp_clearance_m", 0.004),          # squeeze below object size

                # Required by Panda model:
                ("share_dir", None),
                ("model_file", None),
                ("arm_joint_tag", None),
                ("finger_joint_tag", None),
                ("initial_joint_angles", None),

                # Action name
                ("action_name", "manip_task"),
            ],
        )

        # Params
        self._dt = float(self.get_parameter("control_dt").value)
        self._pub_fingers = bool(self.get_parameter("publish_fingers").value)
        self._tol = float(self.get_parameter("reach_pos_tol").value)
        self._approach_ticks = int(self.get_parameter("approach_ticks").value)
        self._grasp_dwell_ticks = int(self.get_parameter("grasp_dwell_ticks").value)
        self._preclose_offset = float(self.get_parameter("preclose_offset").value)
        self._max_jstep = float(self.get_parameter("max_joint_vel").value) * self._dt
        self._home_xyz = [float(x) for x in self.get_parameter("home_xyz").value]
        self._home_rpy = [float(x) for x in self.get_parameter("home_rpy_deg").value]
        self._action_name = self.get_parameter("action_name").value
        self._preopen_ticks = int(self.get_parameter("preopen_ticks").value)
        self._max_finger_jstep = float(self.get_parameter("max_finger_joint_vel").value) * self._dt
        self._normal_open_width_m = float(self.get_parameter("normal_open_width_m").value)
        self._pregrasp_pause = int(self.get_parameter("pregrasp_pause_ticks").value)
        self._close_on_actual_z = bool(self.get_parameter("close_on_actual_z").value)
        self._grasp_close_step = float(self.get_parameter("grasp_close_speed_m_per_s").value) * self._dt
        self._grasp_clearance = float(self.get_parameter("grasp_clearance_m").value)


        # IO
        self._pub_cmd = self.create_publisher(
            Float64MultiArray, self.get_parameter("joint_control_topic").value, 10
        )
        self._pub_target = self.create_publisher(
            Odometry, self.get_parameter("end_effector_target_topic").value, 10
        )
        self._pub_pose = self.create_publisher(
            Odometry, self.get_parameter("end_effector_pose_topic").value, 10
        )
        self._pub_status = self.create_publisher(String, "/manip_task/status", 10)
        self._sub_js = self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self._sub_cmd = self.create_subscription(String, "/manip_task/cmd", self._on_cmd, 10)

        # Model
        self._panda = Panda(self)
        self._nj = self._panda.num_joints

        self._js = JointState()
        self._js.position = list(self._panda.reset_model())
        self._js.velocity = [0.0] * self._nj
        self._js.effort = [0.0] * self._nj

        self._ee_current: Odometry = self._panda.solve_fk(self._js, remap=False)
        self._ee_target: Odometry = copy.deepcopy(self._ee_current)

        # Home target
        qx, qy, qz, qw = quat_rpy_deg(tuple(self._home_rpy))
        self._ee_target.pose.pose.position.x = self._home_xyz[0]
        self._ee_target.pose.pose.position.y = self._home_xyz[1]
        self._ee_target.pose.pose.position.z = self._home_xyz[2]
        self._ee_target.pose.pose.orientation.x = qx
        self._ee_target.pose.pose.orientation.y = qy
        self._ee_target.pose.pose.orientation.z = qz
        self._ee_target.pose.pose.orientation.w = qw
        self._home_target = copy.deepcopy(self._ee_target)

        # Initial command
        jt0 = self._safe_ik(self._ee_target)
        self._joint_targets: List[float] = list(self._js.position) if jt0 is None else jt0
        self._cmd_prev = np.array(
            self._joint_targets[: (7 if not self._pub_fingers else self._nj)], dtype=float
        )
        self._publish_cmd(self._joint_targets)

        self._rviz = RVizHelper(self)

        # Task state
        self._phase = Phase.IDLE
        self._queue: Deque[Tuple[str, List[float]]] = deque()
        self._tick = 0
        self._jt_start = self._joint_targets.copy()
        self._jt_goal = self._joint_targets.copy()
        self._hover_z = self._home_xyz[2]
        self._pick_z = 0.03
        self._pick_xy = (self._home_xyz[0], self._home_xyz[1])
        self._place_xy = (self._home_xyz[0], self._home_xyz[1])
        self._rpy = self._home_rpy

        # Action server state
        self._active_goal_lock = threading.Lock()
        self._active_goal_handle = None
        self._action_done_evt = threading.Event()
        self._last_status = ""

        self._as = ActionServer(
            self,
            ManipTask,
            self._action_name,
            execute_callback=self._execute_cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
        )

        self._status("ready")

    # ---------- Utils ----------

    def _status(self, msg: str):
        self._last_status = msg
        self._pub_status.publish(String(data=msg))
        with self._active_goal_lock:
            gh = self._active_goal_handle
        if gh is not None:
            fb = ManipTask.Feedback()
            fb.status = msg
            try:
                gh.publish_feedback(fb)
            except Exception:
                pass
            if msg.lower().startswith("done") or msg.startswith("ERR"):
                self._action_done_evt.set()

    def _publish_cmd(self, joints: List[float]) -> None:
        n = (7 if not self._pub_fingers else self._nj)
        desired = np.array(joints[:n], dtype=float)

        delta = desired - self._cmd_prev
        # arm joints
        if n >= 7:
            delta[:7] = np.clip(delta[:7], -self._max_jstep, self._max_jstep)
        # fingers (last two joints)
        if self._pub_fingers and n >= 9:
            delta[-2:] = np.clip(delta[-2:], -self._max_finger_jstep, self._max_finger_jstep)

        limited = self._cmd_prev + delta
        self._cmd_prev = limited.copy()
        self._pub_cmd.publish(Float64MultiArray(data=list(map(float, limited))))

    def _safe_ik(self, odom: Odometry) -> Optional[List[float]]:
        jt = self._panda.solve_ik(odom)
        if jt is None:
            return None
        arr = np.asarray(jt, float)
        if arr.shape[0] != self._nj or not np.all(np.isfinite(arr)):
            return None
        return list(arr)

    def _reached(self, tol: float) -> bool:
        p = np.array(
            [
                self._ee_current.pose.pose.position.x,
                self._ee_current.pose.pose.position.y,
                self._ee_current.pose.pose.position.z,
            ],
            float,
        )
        t = np.array(
            [
                self._ee_target.pose.pose.position.x,
                self._ee_target.pose.pose.position.y,
                self._ee_target.pose.pose.position.z,
            ],
            float,
        )
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(t))):
            return False
        return np.linalg.norm(p - t) < tol

    @staticmethod
    def _lerp_joints(a: List[float], b: List[float], w: float) -> List[float]:
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        return list((1.0 - w) * a + w * b)

    def _goal_from_xyzrpy(self, x, y, z, rpy_deg):
        od = copy.deepcopy(self._ee_target)
        od.pose.pose.position.x = x
        od.pose.pose.position.y = y
        od.pose.pose.position.z = z
        qx, qy, qz, qw = quat_rpy_deg(tuple(rpy_deg))
        od.pose.pose.orientation.x = qx
        od.pose.pose.orientation.y = qy
        od.pose.pose.orientation.z = qz
        od.pose.pose.orientation.w = qw
        return od

    def _set_gripper_open_width(self, total_width_m: float):
        """
        Open to an explicit total width (meters). For Franka, per-finger joint range is ~[0, 0.04] m.
        """
        per_finger = max(0.0, min(total_width_m * 0.5, 0.04))
        self._joint_targets[-2:] = [per_finger, per_finger]

    def _set_gripper(self, act: FingersAction):
        self._joint_targets[-2:] = self._panda.move_fingers(self._joint_targets, act)[-2:]

    # ---------- Command parsing ----------

    def _on_cmd(self, msg: String):
        txt = msg.data.strip()
        if not txt:
            return
        cmd = txt.split()[0].upper()
        try:
            if cmd == "HOME":
                self._queue.append(("HOME", []))
            elif cmd == "OPEN":
                self._queue.append(("OPEN", []))
            elif cmd == "CLOSE":
                self._queue.append(("CLOSE", []))
            elif cmd == "GO":
                parts = txt.split()
                if len(parts) not in (4, 7):
                    self._status("ERR go usage: GO x y z [roll pitch yaw_deg]")
                    return
                x, y, z = map(float, parts[1:4])
                rpy = list(map(float, parts[4:7])) if len(parts) == 7 else self._home_rpy
                self._queue.append(("GO", [x, y, z] + rpy))
            elif cmd == "PICK":
                parts = txt.split()
                if len(parts) not in (4, 5):
                    self._status("ERR pick usage: PICK x y z [size]")
                    return
                x, y, z = map(float, parts[1:4])
                size = float(parts[4]) if len(parts) == 5 else 0.05
                self._queue.append(("PICK", [x, y, z, size]))
            elif cmd == "PLACE":
                parts = txt.split()
                if len(parts) not in (4, 5):
                    self._status("ERR place usage: PLACE x y z [size]")
                    return
                x, y, z = map(float, parts[1:4])
                size = float(parts[4]) if len(parts) == 5 else 0.05
                self._queue.append(("PLACE", [x, y, z, size]))
            else:
                self._status(f"ERR unknown: {cmd}")
                return
            self._status(f"queued {cmd}")
        except Exception as e:
            self._status(f"ERR parse: {e}")

    # ---------- JS callback ----------

    def _on_js(self, js: JointState):
        if not js.position:
            return
        self._js = js
        self._panda.set_joint_states(self._js)
        self._ee_current = self._panda.solve_fk(self._js)

        self._step_fsm()

        self._pub_target.publish(self._ee_target)
        self._pub_pose.publish(self._ee_current)
        self._rviz.publish(self._ee_current)

        self._publish_cmd(self._joint_targets)

    # ---------- FSM ----------

    def _step_fsm(self):
        # pull next task
        if self._phase == Phase.IDLE and self._queue:
            name, args = self._queue.popleft()
            if name == "HOME":
                self._start_go(self._home_xyz[0], self._home_xyz[1], self._home_xyz[2], self._home_rpy)
                self._status("exec HOME")
            elif name == "OPEN":
                self._set_gripper(FingersAction.OPEN)
                self._status("done OPEN")
            elif name == "CLOSE":
                self._set_gripper(FingersAction.CLOSE)
                self._status("done CLOSE")
            elif name == "GO":
                x, y, z, r, p, ydeg = args
                self._start_go(x, y, z, [r, p, ydeg])
                self._status(f"exec GO {x:.3f} {y:.3f} {z:.3f}")
            elif name == "PICK":
                x, y, z, size = args
                self._start_pick(x, y, z, size)
                self._status(f"exec PICK {x:.3f} {y:.3f} {z:.3f}")
            elif name == "PLACE":
                x, y, z, size = args
                self._start_place(x, y, z, size)
                self._status(f"exec PLACE {x:.3f} {y:.3f} {z:.3f}")
            return

        # phases
        if self._phase == Phase.GO:
            if self._reached(self._tol):
                self._phase = Phase.IDLE
                self._status("done GO")
            return

        if self._phase == Phase.PICK_DESCEND:
            w = min(1.0, self._tick / max(1, self._approach_ticks))
            self._joint_targets = self._lerp_joints(self._jt_start, self._jt_goal, w)
            self._tick += 1

            reached_by_tick = (self._tick >= self._approach_ticks)
            if self._close_on_actual_z:
                z_now = float(self._ee_current.pose.pose.position.z)
                at_bottom = (z_now <= self._pick_z + self._preclose_offset)
            else:
                at_bottom = reached_by_tick

            if at_bottom:
                self._phase = Phase.PICK_PREOPEN  # reuse this slot for a short settle pause
                self._tick = 0
            return

        if self._phase == Phase.PICK_PREOPEN:
            # settle at the bottom so the object stops moving
            self._tick += 1
            if self._tick >= max(self._pregrasp_pause, 1):
                self._phase = Phase.PICK_DWELL
                self._tick = 0
            return

        if self._phase == Phase.PICK_DWELL:
            # target = object size minus a small clearance
            target_width = max(0.0, self._obj_size - self._grasp_clearance)
            self._ramp_gripper_to_width(target_width, self._grasp_close_step)

            self._tick += 1
            if self._tick >= self._grasp_dwell_ticks:
                up = self._goal_from_xyzrpy(self._pick_xy[0], self._pick_xy[1], self._hover_z, self._rpy)
                jt = self._safe_ik(up)
                if jt is not None:
                    self._joint_targets = jt
                    self._ee_target = up
                self._phase = Phase.PICK_LIFT
                self._tick = 0
            return

        if self._phase == Phase.PICK_LIFT:
            if self._reached(self._tol):
                self._phase = Phase.IDLE
                self._status("done PICK")
            return

        if self._phase == Phase.PLACE_DESCEND:
            if self._reached(self._tol):
                self._phase = Phase.PLACE_OPEN
                self._tick = 0
            return

        if self._phase == Phase.PLACE_OPEN:
            self._set_gripper(FingersAction.OPEN)
            self._tick += 1
            if self._tick >= max(50, int(0.5 / self._dt)):
                up = self._goal_from_xyzrpy(self._place_xy[0], self._place_xy[1], self._hover_z, self._rpy)
                jt = self._safe_ik(up)
                if jt is not None:
                    self._joint_targets = jt
                    self._ee_target = up
                self._phase = Phase.PLACE_LIFT
                self._tick = 0
            return

        if self._phase == Phase.PLACE_LIFT:
            if self._reached(self._tol):
                self._phase = Phase.IDLE
                self._status("done PLACE")
            return

    # ---------- Task starters ----------

    def _ramp_gripper_to_width(self, total_width_m: float, max_step_per_tick_m: float):
        per_finger_target = max(0.0, min(0.5 * total_width_m, 0.04))  # ~Franka limit
        cur = float(self._joint_targets[-2])  # assume symmetry
        step = np.clip(per_finger_target - cur, -max_step_per_tick_m, max_step_per_tick_m)
        new = cur + step
        self._joint_targets[-2:] = [new, new]

    def _start_go(self, x: float, y: float, z: float, rpy_deg: List[float]):
        goal = self._goal_from_xyzrpy(x, y, z, rpy_deg)
        jt = self._safe_ik(goal)
        if jt is None:
            self._status("ERR IK GO")
            self._phase = Phase.IDLE
            return
        self._joint_targets = jt
        self._ee_target = goal
        self._rpy = rpy_deg
        self._phase = Phase.GO
        self._tick = 0

    def _start_pick(self, x: float, y: float, z: float, size: float):
        self._rpy = self._home_rpy
        self._hover_z = max(z + size * 0.6, z + 0.15)
        self._pick_z = z + max(0.5 * size, 0.03)
        self._pick_xy = (x, y)

        hover = self._goal_from_xyzrpy(x, y, self._hover_z, self._rpy)
        jt_hover = self._safe_ik(hover)
        if jt_hover is None:
            self._status("ERR IK HOVER")
            self._phase = Phase.IDLE
            return
        self._joint_targets = jt_hover
        self._ee_target = hover

        down = self._goal_from_xyzrpy(x, y, self._pick_z, self._rpy)
        jt_down = self._safe_ik(down)
        if jt_down is None:
            self._status("ERR IK PICK")
            self._phase = Phase.IDLE
            return
        self._jt_start = jt_hover.copy()
        self._jt_goal = jt_down.copy()

        self._set_gripper(FingersAction.OPEN)
        if self._pub_fingers:
            self._set_gripper_open_width(self._normal_open_width_m)
        else:
            self._set_gripper(FingersAction.OPEN)
        self._phase = Phase.PICK_PREOPEN
        self._tick = 0
        self._obj_size = size


    def _start_place(self, x: float, y: float, z: float, size: float):
        self._rpy = self._home_rpy
        self._hover_z = max(z + size * 0.6, z + 0.15)
        self._place_xy = (x, y)
        place_z = z + max(0.5 * size, 0.03)

        above = self._goal_from_xyzrpy(x, y, self._hover_z, self._rpy)
        jt_above = self._safe_ik(above)
        if jt_above is None:
            self._status("ERR IK ABOVE")
            self._phase = Phase.IDLE
            return
        self._joint_targets = jt_above
        self._ee_target = above

        down = self._goal_from_xyzrpy(x, y, place_z, self._rpy)
        jt_down = self._safe_ik(down)
        if jt_down is None:
            self._status("ERR IK PLACE")
            self._phase = Phase.IDLE
            return
        self._joint_targets = jt_down
        self._ee_target = down
        self._phase = Phase.PLACE_DESCEND
        self._tick = 0

    # ---------- Action callbacks ----------

    def _goal_cb(self, goal_request: ManipTask.Goal) -> int:
        with self._active_goal_lock:
            if self._active_goal_handle is not None:
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> int:
        with self._active_goal_lock:
            if self._active_goal_handle != goal_handle:
                return CancelResponse.REJECT
        self._queue.clear()
        self._phase = Phase.IDLE
        self._status("ERR canceled")
        self._action_done_evt.set()
        return CancelResponse.ACCEPT

    def _execute_cb(self, goal_handle):
        cmd = goal_handle.request.command.strip()
        with self._active_goal_lock:
            self._active_goal_handle = goal_handle
        self._action_done_evt.clear()

        # route through the existing command parser
        self._on_cmd(String(data=cmd))

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                res = ManipTask.Result()
                res.result = f"Aborted: canceled {cmd}"
                with self._active_goal_lock:
                    self._active_goal_handle = None
                return res
            if self._action_done_evt.wait(timeout=0.01):
                break

        res = ManipTask.Result()
        if self._last_status.lower().startswith("done"):
            goal_handle.succeed()
            res.result = f"Done: {cmd}"
        else:
            goal_handle.abort()
            why = self._last_status if self._last_status else "unknown error"
            res.result = f"Aborted: {why}"
        with self._active_goal_lock:
            self._active_goal_handle = None
        return res


def main(args=None):
    rclpy.init(args=args)
    node = PandaTaskLibrary()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
