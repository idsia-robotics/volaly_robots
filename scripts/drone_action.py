#!/usr/bin/env python

import math
import numpy as np
import copy

import rospy
import rostopic
import actionlib
import threading

import std_srvs.srv
from std_msgs.msg import Empty
from volaly_msgs.msg import EmptyAction
from volaly_msgs.msg import GoToAction, GoToGoal, GoToFeedback, GoToResult
from volaly_msgs.msg import FollowShapeAction, FollowShapeFeedback, FollowShapeResult
from volaly_msgs.msg import FollowMeAction, FollowMeFeedback, FollowMeResult
from volaly_msgs.msg import WaypointsAction, WaypointsFeedback, WaypointsResult
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Joy
from drone_arena_msgs.msg import State as FlightState
from drone_arena_msgs.msg import TargetSource
from drone_arena_msgs.srv import SetXY, SetXYRequest

import tf2_ros
import tf_conversions as tfc
import tf2_geometry_msgs
import PyKDL as kdl

flight_state_names = {v: k for k,v in FlightState.__dict__.items() if k[0].isupper()}

class ActionHub:
    def __init__(self):
        self.servers = []

    @classmethod
    def instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    @classmethod
    def add_server(cls, ns, action_type, execute_cb, autostart):
        srv = actionlib.SimpleActionServer(ns, action_type, execute_cb, autostart)
        cls.instance().servers.append(srv)
        return srv

    @classmethod
    def preempt_all(cls):
        loop_rate = rospy.Rate(20) # 20 Hz
        for s in cls.instance().servers:
            if s.is_active():
                s.preempt_request = True
                while s.is_active():
                    loop_rate.sleep()

    @classmethod
    def preempt_all_but(cls, exclude_servers):
        # Check if iterable
        excl_ss = exclude_servers
        try:
            iter(excl_ss)
        except TypeError:
            # Make iterable
            excl_ss = [exclude_servers]

        loop_rate = rospy.Rate(20) # 20 Hz
        for s in cls.instance().servers:
            if (s not in excl_ss and s.is_active()):
                s.preempt_request = True
                while s.is_active():
                    loop_rate.sleep()

    @staticmethod
    def preempt(servers):
        loop_rate = rospy.Rate(20) # 20 Hz

        # Check if iterable
        ss = servers
        try:
            iter(ss)
        except TypeError:
            # Make iterable
            ss = [servers]

        for s in ss:
            if s.is_active():
                s.preempt_request = True
                while s.is_active():
                    loop_rate.sleep()

class DroneActionsServer:
    def __init__(self):
        action_ns = rospy.get_param('~action_ns', '')

        # Monitor manual control and cancel any actions if user presses any button
        joy_safety_topic = rospy.get_param('~joy_safety_topic', '/drone/joy')

        self.distance_threshold = rospy.get_param('~distance_threshold', 0.10)
        self.yaw_threshold = rospy.get_param('~yaw_threshold', math.pi / 180.0 * 10.0)

        self.robot_desired_pose_topic = rospy.get_param('~robot_desired_pose_topic', 'desired_pose')
        robot_odom_topic = rospy.get_param('~robot_odom_topic', '/drone/odom')

        robot_state_topic = rospy.get_param('~robot_state_topic', '/drone/flight_state')

        takeoff_service = rospy.get_param('~takeoff_service', '/drone/safe_takeoff')
        land_service = rospy.get_param('~land_service', '/drone/safe_land')
        feedback_service = rospy.get_param('~feedback_service', '/drone/give_feedback')

        target_source_topic = rospy.get_param('~target_source_topic', '/drone/target_source')
        set_pose_topic = rospy.get_param('~set_pose_topic', '/drone/set_pose')
        set_xy_service = rospy.get_param('~set_xy_service', '/drone/set_xy')

        self.robot_desired_pose = PoseStamped()
        self.robot_current_pose = PoseStamped()

        self.tf_buff = tf2_ros.Buffer()
        self.tf_ls = tf2_ros.TransformListener(self.tf_buff)

        # Action client to self, see below
        self.goto_client = actionlib.SimpleActionClient(action_ns + '/goto_action', GoToAction)

        # Action servers
        self.goto_server = ActionHub.add_server(action_ns + '/goto_action', GoToAction, self.execute_goto, False)
        self.goto_server.start()
        # self.shape_server = ActionHub.add_server(action_ns + '/followshape_action', FollowShapeAction, self.execute_shape, False)
        # self.shape_server.start()
        self.precise_shape_server = ActionHub.add_server(action_ns + '/precise_shape_action', FollowShapeAction, self.execute_precise_shape, False)
        self.precise_shape_server.start()

        self.takeoff_server = ActionHub.add_server(action_ns + '/takeoff_action', EmptyAction, self.execute_takeoff, False)
        self.takeoff_server.start()
        self.land_server = ActionHub.add_server(action_ns + '/land_action', EmptyAction, self.execute_land, False)
        self.land_server.start()

        self.feedback_server = ActionHub.add_server(action_ns + '/feedback_action', EmptyAction, self.execute_feedback, False)
        self.feedback_server.start()

        self.followme_server = ActionHub.add_server(action_ns + '/followme_action', FollowMeAction, self.execute_followme, False)
        self.followme_server.start()

        self.waypoints_server = ActionHub.add_server(action_ns + '/waypoints_action', WaypointsAction, self.execute_waypoints, False)
        self.waypoints_server.start()

        self.reset_odom_server = ActionHub.add_server(action_ns + '/reset_odom_action', EmptyAction, self.execute_reset_odom, False)
        self.reset_odom_server.start()

        self.pub_desired_pose = rospy.Publisher(self.robot_desired_pose_topic, PoseStamped, queue_size = 10)
        self.sub_odom = rospy.Subscriber(robot_odom_topic, Odometry, self.robot_odom_cb)

        self.takeoff_svc = rospy.ServiceProxy(takeoff_service, std_srvs.srv.Trigger)
        self.land_svc = rospy.ServiceProxy(land_service, std_srvs.srv.Trigger)
        self.feedback_svc = rospy.ServiceProxy(feedback_service, std_srvs.srv.Trigger)

        self.pub_target_source = rospy.Publisher(target_source_topic, TargetSource, queue_size = 1)
        self.pub_set_pose = rospy.Publisher(set_pose_topic, Pose, queue_size = 1)
        self.set_xy_service = rospy.ServiceProxy(set_xy_service, SetXY)

        self.last_known_flight_state = FlightState.Landed #None
        self.sub_flight_state = rospy.Subscriber(robot_state_topic, FlightState, self.flight_state_cb)

        self.sub_joy_safety = rospy.Subscriber(joy_safety_topic, Joy, self.joy_safety_cb)

        # self.ros_thread = threading.Thread(target=self.run)
        # self.ros_thread.start()

        rospy.wait_for_service(takeoff_service)
        rospy.wait_for_service(land_service)
        rospy.wait_for_service(feedback_service)

    def set_target_source(self, topic = None):
        if topic:
            # We always use the pose mode
            self.pub_target_source.publish(TargetSource(mode = TargetSource.Pos, topic = topic))
        else:
            # Reset / Detach
            self.pub_target_source.publish(TargetSource(mode = TargetSource.Teleop))

    def joy_safety_cb(self, msg):
        if msg.buttons[7]:
            ActionHub.preempt_all()

    def flight_state_cb(self, msg):
        last_state_name = 'unknown' if not self.last_known_flight_state else flight_state_names[self.last_known_flight_state]
        new_state_name = flight_state_names[msg.state]

        rospy.loginfo('Flying state changed from [{}] to [{}]'.format(last_state_name, new_state_name))

        self.last_known_flight_state = msg.state

        # Cancel all actions if the drone landed for some reason
        # but ignore if it was us to ask it to land
        if not self.land_server.is_active:
            if self.last_known_flight_state == FlightState.Landed:
                ActionHub.preempt_all()

    def robot_odom_cb(self, msg):
        # E.g. /drone/odom in 'drone/odom' frame
        header = msg.header
        pose = msg.pose.pose

        pose_stamped = PoseStamped()
        pose_stamped.header = header
        pose_stamped.pose = pose

        self.robot_current_pose = pose_stamped

    def kdl_to_transform(self, k):
        t = TransformStamped()
        t.transform.translation.x = k.p.x()
        t.transform.translation.y = k.p.y()
        t.transform.translation.z = k.p.z()
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w) = k.M.GetQuaternion()

        return t

    def distance(self, pose_a, pose_b):
        try:
            # E.g. transform pose_a from 'odom' to 'World' frame
            trans = self.tf_buff.lookup_transform(pose_a.header.frame_id, pose_b.header.frame_id, rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException), e:
            rospy.logerr_throttle(10.0, e)
            return None, None, None

        pose_trans = tf2_geometry_msgs.do_transform_pose(pose_b, trans)
        p1 = tfc.fromMsg(pose_a.pose)
        p2 = tfc.fromMsg(pose_trans.pose)

        full_dist = (p1.p - p2.p).Norm()

        # Ignore altitude
        p1.p[2] = 0.0
        p2.p[2] = 0.0

        hor_dist = (p1.p - p2.p).Norm()
        _,_,yaw1 = p1.M.GetRPY()
        _,_,yaw2 = p2.M.GetRPY()
        yaw = math.fabs(yaw1 - yaw2)

        return hor_dist, full_dist, yaw

    def execute_goto(self, goal):
        loop_rate = rospy.Rate(50) # 50 Hz

        # Cancel other actions, BUT do not preempt
        # PreciseShape because GoTo might be called by it!

        ActionHub.preempt_all_but([self.goto_server, self.precise_shape_server])

        self.set_target_source(self.robot_desired_pose_topic)

        self.robot_desired_pose = goal.pose
        self.pub_desired_pose.publish(self.robot_desired_pose)

        while not rospy.is_shutdown():
            if self.goto_server.is_preempt_requested():
                self.goto_server.set_preempted()
                rospy.logwarn('GoTo action has been preempted')
                break

            d, fd, yaw = self.distance(self.robot_desired_pose, self.robot_current_pose)

            if d == None:
                self.goto_server.set_aborted()
                break

            f = GoToFeedback()
            f.distance = fd
            f.yaw = yaw

            self.goto_server.publish_feedback(f)

            if d < self.distance_threshold and yaw < self.yaw_threshold:
                self.goto_server.set_succeeded()
                break

            loop_rate.sleep()

        self.set_target_source()

    def execute_precise_shape(self, goal):
        loop_rate = rospy.Rate(50) # 50 Hz

        ActionHub.preempt_all_but(self.precise_shape_server)

        poses = self.path_from_shape(goal)

        while not rospy.is_shutdown():
            for idx, p in enumerate(poses):
                if self.precise_shape_server.is_preempt_requested():

                    self.goto_client.cancel_goal()
                    self.precise_shape_server.set_preempted()

                    rospy.logwarn('Precise FollowShape action has been preempted')
                    break

                self.goto_client.send_goal_and_wait(GoToGoal(p))

                f = FollowShapeFeedback()
                self.precise_shape_server.publish_feedback(f)

                loop_rate.sleep()

    def execute_takeoff(self, goal):
        loop_rate = rospy.Rate(10) # 10 Hz

        ## Allow takeoff ONLY if the drone is 'landed' not 'landing'.
        # Attempt to takeoff while the drone is still airborne but 'landing'
        # may result in the crash into the ceiling (as happened once)
        if self.last_known_flight_state != FlightState.Landed:
            self.takeoff_server.set_aborted()
            return

        # Cancel other actions
        ActionHub.preempt_all_but(self.takeoff_server)

        while not rospy.is_shutdown():
            if self.takeoff_server.is_preempt_requested():
                self.takeoff_server.set_preempted()

                rospy.logwarn('Takeoff action has been preempted')
                break

            if self.last_known_flight_state == FlightState.Hovering or self.last_known_flight_state == FlightState.Flying:
                self.takeoff_server.set_succeeded()
                break

            if self.last_known_flight_state != FlightState.TakingOff:
                try:
                    self.takeoff_svc()
                except rospy.ServiceException, e:
                    rospy.logerr('TakeOff service call failed: {}'.format(e))
                    self.takeoff_server.set_aborted()
                    break

            loop_rate.sleep()

    def execute_land(self, goal):
        loop_rate = rospy.Rate(10) # 10 Hz

        if not (self.last_known_flight_state == FlightState.Hovering or
            self.last_known_flight_state == FlightState.Flying or
            self.last_known_flight_state == FlightState.TakingOff):

            self.land_server.set_aborted()

            return

        # Cancel other actions
        ActionHub.preempt_all_but(self.land_server)

        while not rospy.is_shutdown():
            if self.land_server.is_preempt_requested():
                self.land_server.set_preempted()
                rospy.logwarn('Land action has been preempted')
                break

            if self.last_known_flight_state == FlightState.Landed:
                self.land_server.set_succeeded()
                break

            if self.last_known_flight_state != FlightState.Landing or self.last_known_flight_state != FlightState.Landed:
                try:
                    self.land_svc()
                except rospy.ServiceException, e:
                    rospy.logerr('Land service call failed: {}'.format(e))
                    self.land_server.set_aborted()
                    break

            loop_rate.sleep()

    def execute_feedback(self, goal):
        loop_rate = rospy.Rate(10) # 10 Hz

        if self.last_known_flight_state != FlightState.Flying and self.last_known_flight_state != FlightState.Hovering:
            self.feedback_server.set_aborted()
            return

        # Cancel other actions
        ActionHub.preempt_all_but(self.feedback_server)

        while not rospy.is_shutdown():
            if self.feedback_server.is_preempt_requested():
                self.feedback_server.preempt_request = False
                rospy.logwarn('Feedback action cannot been preempted')
                # break

            if self.last_known_flight_state != FlightState.TakingOff:
                try:
                    self.feedback_svc()
                    self.feedback_server.set_succeeded()
                    break
                except rospy.ServiceException, e:
                    rospy.logerr('Feedback service call failed: {}'.format(e))
                    self.feedback_server.set_aborted()
                    break

            loop_rate.sleep()

    def execute_followme(self, goal):
        loop_rate = rospy.Rate(50) # 10 Hz

        if not (self.last_known_flight_state == FlightState.Hovering or
            self.last_known_flight_state == FlightState.Flying):
            rospy.logerr('Drone is neither flying nor hovering. Current state: {}'.format(flight_state_names[self.last_known_flight_state]))
            self.followme_server.set_aborted()
            return
        else:
            try:
                topic_class, real_topic, eval_func = rostopic.get_topic_class(rospy.resolve_name(goal.topic))

                def pose_cb(msg):
                    # Extract the actual message
                    if eval_func:
                        real_msg = copy.deepcopy(eval_func(msg))
                    else:
                        real_msg = copy.deepcopy(msg)

                    hor_d, full_d, yaw = self.distance(real_msg, self.robot_current_pose)
                    dist = full_d

                    if goal.ignore_z or goal.override_z > 0.0:
                        try:
                            # Find transform from given frame to 'World' frame
                            trans = self.tf_buff.lookup_transform(self.robot_current_pose.header.frame_id, real_msg.header.frame_id, rospy.Time())
                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException), e:
                            rospy.logerr_throttle(10.0, e)
                            return

                        # Convert pose to 'World' frame
                        pose_trans = tf2_geometry_msgs.do_transform_pose(real_msg, trans)
                        # Override its altitude
                        if goal.override_z > 0.0:
                            pose_trans.pose.position.z = goal.override_z
                        else:
                            pose_trans.pose.position.z = self.robot_current_pose.pose.position.z

                        ## Convert the pose back to its original frame
                        # Get inversed transform as PyKDL.Frame
                        k = tf2_geometry_msgs.transform_to_kdl(trans).Inverse()
                        trans_inv = self.kdl_to_transform(k)

                        old_header = real_msg.header
                        real_msg = tf2_geometry_msgs.do_transform_pose(pose_trans, trans_inv)
                        real_msg.header = old_header
                        dist = hor_d

                    if goal.lookat:
                        try:
                            # Find transform from given frame to 'World' frame
                            trans = self.tf_buff.lookup_transform(self.robot_current_pose.header.frame_id, real_msg.header.frame_id, rospy.Time())
                        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException), e:
                            rospy.logerr_throttle(10.0, e)
                            return

                        # Convert pose to 'World' frame
                        pose_trans = tf2_geometry_msgs.do_transform_pose(real_msg, trans)
                        # Target pose
                        tp = kdl.Vector(pose_trans.pose.position.x,
                                        pose_trans.pose.position.y,
                                        0.0)
                        # Current pose
                        cp = kdl.Vector(self.robot_current_pose.pose.position.x,
                                        self.robot_current_pose.pose.position.y,
                                        0.0)
                        # Direction vector
                        dir_v = tp - cp

                        dir_yaw = math.atan2(dir_v.y(), dir_v.x())
                        quat = kdl.Rotation.RPY(0.0, 0.0, dir_yaw).GetQuaternion()

                        pose_trans.pose.orientation.x = quat[0]
                        pose_trans.pose.orientation.y = quat[1]
                        pose_trans.pose.orientation.z = quat[2]
                        pose_trans.pose.orientation.w = quat[3]

                        ## Convert the pose back to its original frame
                        # Get inversed transform as PyKDL.Frame
                        k = tf2_geometry_msgs.transform_to_kdl(trans).Inverse()
                        trans_inv = self.kdl_to_transform(k)

                        old_header = real_msg.header
                        real_msg = tf2_geometry_msgs.do_transform_pose(pose_trans, trans_inv)
                        real_msg.header = old_header

                        _, _, yaw = self.distance(real_msg, self.robot_current_pose)

                    if goal.margin > 0.0:
                        if dist <= goal.margin:
                            return

                    f = FollowMeFeedback()
                    f.distance = dist
                    f.yaw = yaw
                    self.followme_server.publish_feedback(f)

                    self.last_desired_pose = real_msg
                    self.pub_desired_pose.publish(real_msg)


                # Set the source
                self.set_target_source(self.robot_desired_pose_topic)

                # Subscribe to the topic
                sub_pose = rospy.Subscriber(real_topic, topic_class, pose_cb)

            except rostopic.ROSTopicException as e:
                rospy.logerr('Cannot determine/load the topic: %s', goal.topic)
                self.followme_server.set_aborted()
                return

        # Cancel other actions
        ActionHub.preempt_all_but(self.followme_server)

        while not rospy.is_shutdown():
            if self.followme_server.is_preempt_requested():
                rospy.logwarn('FollowMe action preempt request accepted. Going to final desired location')

                self.followme_server.set_preempted()
                sub_pose.unregister()

                self.goto_client.send_goal_and_wait(GoToGoal(self.last_desired_pose))

                rospy.logwarn('FollowMe action has been preempted')
                break

            loop_rate.sleep()

        self.set_target_source()

    def execute_waypoints(self, goal):
        loop_rate = rospy.Rate(50) # 50 Hz

        if not goal.waypoints:
            rospy.logerr('Waypoints list cannot be empty')
            self.waypoints_server.set_aborted()
            return

        # if not (self.last_known_flight_state == FlightState.Hovering or
        #     self.last_known_flight_state == FlightState.Flying):
        #     self.waypoints_server.set_aborted()
        #     return

        # Cancel other actions
        ActionHub.preempt_all_but(self.waypoints_server)

        # Enumerate
        waypoints = enumerate(goal.waypoints)
        # Fetch next waypoint
        index, next_pose = waypoints.next()

        self.set_target_source(self.robot_desired_pose_topic)

        while not rospy.is_shutdown():
            if self.waypoints_server.is_preempt_requested():
                self.waypoints_server.set_preempted()
                rospy.logwarn('Waypoints action has been preempted')
                break

            # Set the next target
            self.pub_desired_pose.publish(next_pose)

            # Calculate the distance to the current target
            hor_dist, full_dist, yaw = self.distance(next_pose, self.robot_current_pose)

            if hor_dist == None:
                self.waypoints_server.set_aborted()
                break

            f = WaypointsFeedback()
            f.current_index = index
            f.distance = full_dist
            f.yaw = yaw
            self.waypoints_server.publish_feedback(f)

            if hor_dist < goal.distance_threshold and yaw < goal.yaw_threshold:
                # Fetch the next target
                try:
                    index, next_pose = waypoints.next()
                except StopIteration:
                    if goal.loop:
                        # Reload waypoints
                        waypoints = enumerate(goal.waypoints)
                        index, next_pose = waypoints.next()
                    else:
                        # Otherwise succeed
                        self.waypoints_server.set_succeeded()
                        break

            loop_rate.sleep()

        self.set_target_source()

    def execute_reset_odom(self, goal):
        if self.reset_odom_server.is_preempt_requested():
            self.reset_odom_server.set_preempted()
            rospy.logwarn('Reset odometry action has been preempted')
            return

        if self.last_known_flight_state != FlightState.Hovering and self.last_known_flight_state != FlightState.Landed:
            self.logerr('Attempt to reset odometry while flying')
            self.reset_odom_server.set_aborted()
            return

        # Keep the altitude and the attitude
        # new_pose = copy.deepcopy(self.robot_current_pose.pose)
        # # only reset x and y
        # new_pose.position.x = 0
        # new_pose.position.y = 0

        # self.pub_set_pose.publish(new_pose)

        try:
            self.set_xy_service(SetXYRequest(x=0, y=0))
        except rospy.ServiceException as e:
            rospy.logerr('Failed to call XY service: {}'.format(e))
            self.reset_odom_server.set_aborted()

        self.reset_odom_server.set_succeeded()

    def run(self):
        loop_rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            try:
                loop_rate.sleep()
            except rospy.ROSException, e:
                if e.message == 'ROS time moved backwards':
                    rospy.logwarn("Saw a negative time change, resetting.")

if __name__ == '__main__':
    rospy.init_node('drone_actions')
    server = DroneActionsServer()

    rospy.spin()
