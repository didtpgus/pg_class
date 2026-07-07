#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""신호등 기반 출발/정지 제어 노드.

trajectory_tools/idm_node 를 참고했으나, 특정 정지선 좌표 로직은 제거하고
아래 두 가지만 판단한다:

  1) local_path 의 곡률을 보고 "직진에 가까운" 구간에서만 신호등 제어를 적용.
     좌/우회전 등 커브 구간에서는 신호등 제어를 하지 않는다(주행 그대로 통과).
  2) 직진 구간에서 빨간불(red)이면 정지, 직진/직진좌회전(straight/left_straight)이면
     정지 해제. (yellow 는 상태 유지)

정지 시:
  - ctrl_cmd_topic(기본 ctrl_cmd_0) 에 full-brake CtrlCmd 를 발행하고
  - /traffic_light_stop (Bool) 을 True 로 발행한다.
    pure_pursuit 가 이 플래그를 구독해 주행 명령을 멈추므로 두 명령이 충돌하지 않는다.
"""

import math

import rospy
from std_msgs.msg import Bool
from nav_msgs.msg import Path
from morai_msgs.msg import CtrlCmd

from yolo.msg import YoloDetectionArray


def _parse_class_list(param_value, default):
    if isinstance(param_value, str):
        items = [s.strip() for s in param_value.split(",") if s.strip()]
        return items if items else list(default)
    if isinstance(param_value, (list, tuple)):
        return [str(s).strip() for s in param_value]
    return list(default)


class TrafficLightControlNode:
    def __init__(self):
        rospy.init_node("traffic_light_control", anonymous=False)

        # --- 토픽 ---
        self.detection_topic = rospy.get_param("~detection_topic", "/yolo_detections")
        self.local_path_topic = rospy.get_param("~local_path_topic", "/local_path")
        self.ctrl_cmd_topic = rospy.get_param("~ctrl_cmd_topic", "ctrl_cmd_0")
        self.stop_flag_topic = rospy.get_param("~stop_flag_topic", "/traffic_light_stop")

        # --- 신호등 클래스 ---
        self.red_classes = _parse_class_list(rospy.get_param("~red_classes", ["red"]), ["red"])
        self.go_classes = _parse_class_list(
            rospy.get_param("~go_classes", ["straight", "left_straight"]),
            ["straight", "left_straight"],
        )
        self.confidence_threshold = float(rospy.get_param("~confidence_threshold", 0.5))
        # bounding box 최소 크기(픽셀). 멀리 있는(=작게 잡힌) 신호등은 무시한다.
        # width/height 는 각각의 하한, area(=width*height) 는 넓이 하한.
        # 값이 0 이하이면 해당 조건은 비활성.
        self.min_box_width = float(rospy.get_param("~min_box_width", 0.0))
        self.min_box_height = float(rospy.get_param("~min_box_height", 0.0))
        self.min_box_area = float(rospy.get_param("~min_box_area", 600.0))

        # --- 곡률(직진 판정) ---
        # local_path 시작점부터 lookahead_dist[m] 구간에서 곡률을 계산하고,
        # 그 구간의 "최대 곡률"이 max_curvature_thresh[1/m] 미만이면 직진에 가깝다고 본다.
        # (곡률 κ = 1/R 이므로 예: 0.05 → 반경 20m 보다 완만하면 직진)
        self.curvature_lookahead_dist = float(rospy.get_param("~curvature_lookahead_dist", 8.0))
        self.max_curvature_thresh = float(rospy.get_param("~max_curvature_thresh", 0.07))
        # 판정을 위한 최소 경로 길이. 이보다 짧으면 직전 판정을 유지(플리커 방지).
        self.min_eval_dist = float(rospy.get_param("~min_eval_dist", 2.0))

        self.control_rate = float(rospy.get_param("~control_rate", 50.0))

        # 빨간불 래치 타임아웃(초). 마지막 red 감지 후 이 시간 동안 red 가 다시
        # 안 보이면 래치를 자동 해제한다. 곡선 구간에서 신호등이 시야를 벗어난 뒤
        # 다시 직진이 될 때 신호등도 없는데 멈추는 문제를 막는다. 0 이하이면 비활성.
        self.red_latch_timeout = float(rospy.get_param("~red_latch_timeout", 2.0))

        # --- 상태 ---
        self.stop_active = False        # 빨간불 래치 (green 또는 타임아웃까지 유지)
        self.last_red_time = None       # 마지막으로 red 를 감지한 시각
        self.is_straight = False        # local_path 가 직진에 가까운가
        self.latest_path = None

        # --- 통신 ---
        self.ctrl_cmd_pub = rospy.Publisher(self.ctrl_cmd_topic, CtrlCmd, queue_size=1)
        self.stop_flag_pub = rospy.Publisher(self.stop_flag_topic, Bool, queue_size=1)

        self.stop_cmd = CtrlCmd()
        self.stop_cmd.longlCmdType = 1   # pure_pursuit 와 동일 (throttle/brake 제어)
        self.stop_cmd.accel = 0.0
        self.stop_cmd.brake = 1.0
        self.stop_cmd.steering = 0.0

        rospy.Subscriber(self.detection_topic, YoloDetectionArray,
                         self.detection_callback, queue_size=1)
        rospy.Subscriber(self.local_path_topic, Path,
                         self.path_callback, queue_size=1)

        rospy.loginfo("🚦 Traffic Light Control Node")
        rospy.loginfo("  - detection : %s", self.detection_topic)
        rospy.loginfo("  - local_path: %s", self.local_path_topic)
        rospy.loginfo("  - ctrl_cmd  : %s (stop override)", self.ctrl_cmd_topic)
        rospy.loginfo("  - stop_flag : %s", self.stop_flag_topic)
        rospy.loginfo("  - red=%s  go=%s  conf>=%.2f",
                      self.red_classes, self.go_classes, self.confidence_threshold)
        rospy.loginfo("  - bbox 최소크기: w>=%.0f h>=%.0f area>=%.0f (px)",
                      self.min_box_width, self.min_box_height, self.min_box_area)
        rospy.loginfo("  - 직진판정: lookahead=%.1fm, max_curv<%.3f (1/m)",
                      self.curvature_lookahead_dist, self.max_curvature_thresh)
        rospy.loginfo("  - red 래치 타임아웃: %.1fs", self.red_latch_timeout)

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    def path_callback(self, msg):
        self.latest_path = msg

    def _box_big_enough(self, det):
        """bounding box 가 최소 크기 조건을 모두 만족하는지 확인한다."""
        if self.min_box_width > 0.0 and det.width < self.min_box_width:
            return False
        if self.min_box_height > 0.0 and det.height < self.min_box_height:
            return False
        if self.min_box_area > 0.0 and (det.width * det.height) < self.min_box_area:
            return False
        return True

    def detection_callback(self, msg):
        """가장 신뢰도 높은 신호등 클래스로 정지 래치를 갱신한다."""
        try:
            best_conf = 0.0
            best_class = None
            for det in msg.detections:
                if det.confidence < self.confidence_threshold:
                    continue
                if det.class_name not in self.red_classes and det.class_name not in self.go_classes:
                    continue
                if not self._box_big_enough(det):
                    rospy.logdebug_throttle(
                        1.0, "작은 bbox 무시: %s (w=%d h=%d area=%d)",
                        det.class_name, det.width, det.height, det.width * det.height)
                    continue
                if det.confidence > best_conf:
                    best_conf = det.confidence
                    best_class = det.class_name

            if best_class is None:
                return  # 관련 신호 미검출 → 상태 유지(래치)

            if best_class in self.red_classes:
                if not self.stop_active:
                    rospy.loginfo("🔴 %s (conf %.2f) → 정지 래치", best_class, best_conf)
                self.stop_active = True
                self.last_red_time = rospy.Time.now()
            elif best_class in self.go_classes:
                if self.stop_active:
                    rospy.loginfo("🟢 %s (conf %.2f) → 정지 해제", best_class, best_conf)
                self.stop_active = False
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(5.0, "detection callback error: %s", exc)

    # ------------------------------------------------------------------ #
    # Curvature / straightness
    # ------------------------------------------------------------------ #
    def _update_straightness(self):
        """local_path 시작 구간의 "최대 곡률" 로 직진 여부를 판정한다.

        lookahead_dist[m] 구간 안의 연속 3점마다 곡률(κ=1/R)을 구하고,
        그 중 최댓값이 max_curvature_thresh 미만이면 직진에 가깝다고 본다.
        경로가 없거나 너무 짧으면 직전 판정을 유지한다.
        """
        path = self.latest_path
        if path is None or len(path.poses) < 3:
            return

        # lookahead_dist 까지의 점들을 모은다.
        pts = []
        arc = 0.0
        prev = None
        for p in path.poses:
            x, y = p.pose.position.x, p.pose.position.y
            if prev is not None:
                seg = math.hypot(x - prev[0], y - prev[1])
                if seg < 1e-4:
                    continue
                arc += seg
            pts.append((x, y))
            prev = (x, y)
            if arc >= self.curvature_lookahead_dist:
                break

        if arc < self.min_eval_dist or len(pts) < 3:
            return  # 판정 근거 부족 → 유지

        # 연속 3점의 외접원 곡률(Menger curvature)의 최댓값을 본다.
        max_curv = 0.0
        for (x1, y1), (x2, y2), (x3, y3) in zip(pts, pts[1:], pts[2:]):
            a = math.hypot(x2 - x1, y2 - y1)
            b = math.hypot(x3 - x2, y3 - y2)
            c = math.hypot(x3 - x1, y3 - y1)
            if a < 1e-3 or b < 1e-3 or c < 1e-3:
                continue
            area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
            curv = 2.0 * area2 / (a * b * c)
            if curv > max_curv:
                max_curv = curv

        self.is_straight = max_curv < self.max_curvature_thresh

    # ------------------------------------------------------------------ #
    # Red latch timeout
    # ------------------------------------------------------------------ #
    def _expire_red_latch(self):
        """마지막 red 감지 후 timeout 이 지나면 정지 래치를 해제한다.

        곡선 구간에서 신호등이 시야를 벗어난 뒤 다시 직진이 될 때,
        지나간 빨간불 래치 때문에 신호등도 없는 곳에서 멈추는 것을 막는다.
        """
        if not self.stop_active or self.red_latch_timeout <= 0.0:
            return
        if self.last_red_time is None:
            return
        if (rospy.Time.now() - self.last_red_time).to_sec() > self.red_latch_timeout:
            rospy.loginfo("⏱️ red 미감지 %.1fs 경과 → 정지 래치 자동 해제",
                          self.red_latch_timeout)
            self.stop_active = False

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            self._update_straightness()
            self._expire_red_latch()

            # 커브 구간(직진 아님)에서는 신호등 제어를 하지 않는다.
            apply_control = self.is_straight
            stop_now = apply_control and self.stop_active

            self.stop_flag_pub.publish(Bool(data=stop_now))

            if stop_now:
                # 정지 명령 오버라이드 (pure_pursuit 는 플래그를 보고 주행을 멈춘다)
                self.ctrl_cmd_pub.publish(self.stop_cmd)
                rospy.loginfo_throttle(
                    1.0, "🔴 신호등 정지 (직진구간) → ctrl_cmd brake=1.0")
            else:
                reason = "커브구간" if (self.stop_active and not apply_control) else "주행가능"
                rospy.loginfo_throttle(
                    5.0, "🟢 제한 없음 (%s, stop_latch=%s)", reason, self.stop_active)

            rate.sleep()


def main():
    try:
        TrafficLightControlNode().run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Traffic Light Control Node 종료")


if __name__ == "__main__":
    main()
