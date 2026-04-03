#!/usr/bin/env python3
import math
import time
import heapq
import threading
from typing import List ,Optional ,Tuple ,Dict ,Set

import numpy as np
import rclpy
from rclpy .node import Node
from rclpy .callback_groups import MutuallyExclusiveCallbackGroup
from rclpy .executors import MultiThreadedExecutor
from rclpy .time import Time
from rclpy .qos import (
QoSProfile ,
QoSHistoryPolicy ,
QoSReliabilityPolicy ,
QoSDurabilityPolicy ,
)

from geometry_msgs .msg import Twist ,PoseStamped ,PointStamped
from nav_msgs .msg import OccupancyGrid ,Path
from sensor_msgs .msg import LaserScan ,Image

import tf2_ros

try :
    import cv2
except Exception :
    cv2 =None


class Task3 (Node ):


    def __init__ (self ):
        super ().__init__ ("task3_node")


        self .declare_parameter ("image_topic","/camera/image_raw")
        self .declare_parameter ("camera_hfov_deg",60.0 )
        self .declare_parameter ("loop_waypoints",True )

        image_topic =self .get_parameter ("image_topic").get_parameter_value ().string_value
        self .camera_hfov_deg =float (self .get_parameter ("camera_hfov_deg").value )
        self .loop_waypoints =bool (self .get_parameter ("loop_waypoints").value )


        self .declare_parameter ("vision_throttle_sec",0.15 )
        self .declare_parameter ("vision_resize_width",320 )

        self .vision_throttle_sec =float (self .get_parameter ("vision_throttle_sec").value )
        self .vision_resize_width =int (self .get_parameter ("vision_resize_width").value )
        self ._last_vision_time =0.0


        self ._det_lock =threading .Lock ()


        self .cmd_pub =self .create_publisher (Twist ,"cmd_vel",10 )
        self .path_pub =self .create_publisher (Path ,"global_plan",10 )

        self .red_pub =self .create_publisher (PointStamped ,"/balls/red",10 )
        self .green_pub =self .create_publisher (PointStamped ,"/balls/green",10 )
        self .blue_pub =self .create_publisher (PointStamped ,"/balls/blue",10 )


        map_qos =QoSProfile (
        history =QoSHistoryPolicy .KEEP_LAST ,
        depth =1 ,
        reliability =QoSReliabilityPolicy .RELIABLE ,
        durability =QoSDurabilityPolicy .TRANSIENT_LOCAL ,
        )


        self .map_sub =self .create_subscription (OccupancyGrid ,"map",self .map_callback ,map_qos )
        self .scan_sub =self .create_subscription (LaserScan ,"scan",self .scan_callback ,10 )
        self .image_topic =image_topic

        self .img_sub =None


        self .tf_buffer =tf2_ros .Buffer ()
        self .tf_listener =tf2_ros .TransformListener (self .tf_buffer ,self )


        self .control_group =MutuallyExclusiveCallbackGroup ()
        self .vision_group =MutuallyExclusiveCallbackGroup ()


        self .image_qos =QoSProfile (
        history =QoSHistoryPolicy .KEEP_LAST ,
        depth =1 ,
        reliability =QoSReliabilityPolicy .RELIABLE ,
        durability =QoSDurabilityPolicy .VOLATILE ,
        )


        self .map_msg :Optional [OccupancyGrid ]=None
        self .map_data :Optional [List [int ]]=None
        self .map_width :int =0
        self .map_height :int =0
        self .map_resolution :float =0.0
        self .map_origin_x :float =0.0
        self .map_origin_y :float =0.0


        self .latest_scan :Optional [LaserScan ]=None


        self .waypoints =[
        (8.44, 15.8),
        (-0.52, 3.00),
        (-4.0, 1.5),
        (-4.21, -1.21),
        (-4.24, -4.01),
        (5.45, 3.09),
        (3.27, 3.00),
        (3.25, 1.00),
        (8.77, 1.76),
        (8.17, -2.28),
        (8.11, -5.38),
        ]
        self .wp_index =0
        self .current_wp :Optional [Tuple [float ,float ]]=None


        self .state ="NAV_PLAN"
        self .after_nav_state ="SPIN_SCAN"

        self .current_goal :Optional [Tuple [float ,float ]]=None
        self .current_path :List [Tuple [float ,float ]]=[]
        self .current_path_index :int =0


        self .spin_omega =0.6
        self .spin_accum =0.0
        self .spin_prev_yaw =0.0
        self .spin_target =2.0 *math .pi


        self .last_detection :Optional [Tuple [str ,float ,Tuple [float ,float ,float ]]]=None


        self .det_hold_count =0
        self .det_need_count =2
        self .det_last_color :Optional [str ]=None
        self .det_last_bearing :float =0.0


        self .target_color :Optional [str ]=None
        self .target_point :Optional [Tuple [float ,float ]]=None

        self .ball_samples :Dict [str ,List [Tuple [float ,float ]]]={
        "red":[],"green":[],"blue":[]
        }
        self .found :Dict [str ,Tuple [float ,float ]]={}
        self .min_samples_to_confirm =10


        self .return_wp :Optional [Tuple [float ,float ]]=None


        self .inflation_cells =5
        self .a_star_wall_weight =0.6

        self .max_linear_speed =0.7
        self .k_v =1.0
        self .waypoint_tolerance =0.1

        self .k_w_align =2.5
        self .max_angular_speed_align =3.3
        self .k_w_travel =2.5
        self .max_angular_speed_travel =6.5

        self .min_front_obstacle_distance =0.3
        self .max_front_obstacle_distance =0.60
        self .front_distance_gain =0.5

        self .emergency_backing =False
        self .emergency_back_steps =0
        self .max_emergency_back_steps =7
        self .current_speed =0.0
        self .aligning_to_path =False

        self .dynamic_obstacles :Set [Tuple [int ,int ]]=set ()


        self .timer =self .create_timer (0.05 ,self .timer_cb ,callback_group =self .control_group )

        self .get_logger ().info (
        f"Task3 started. image_topic={image_topic}, hfov_deg={self.camera_hfov_deg}, loop_waypoints={self.loop_waypoints}"
        )


    def set_vision_active (self ,active :bool ):

        if active and self .img_sub is None :
            self .img_sub =self .create_subscription (
            Image ,
            self .image_topic ,
            self .image_cb ,
            self .image_qos ,
            callback_group =self .vision_group ,
            )
            self .get_logger ().info ('Vision enabled (subscribed to /camera/image_raw).')
        elif (not active )and self .img_sub is not None :
            self .destroy_subscription (self .img_sub )
            self .img_sub =None
            self .get_logger ().info ('Vision disabled (unsubscribed from /camera/image_raw).')


    def map_callback (self ,msg :OccupancyGrid ):
        self .map_msg =msg
        self .map_data =list (msg .data )
        self .map_width =msg .info .width
        self .map_height =msg .info .height
        self .map_resolution =msg .info .resolution
        self .map_origin_x =msg .info .origin .position .x
        self .map_origin_y =msg .info .origin .position .y

        self .remove_single_pixel_obstacle_noise (min_cluster_size =2 )

        self .get_logger ().info (
        f"Received map: {self.map_width}x{self.map_height}, res={self.map_resolution:.3f}"
        )

    def scan_callback (self ,msg :LaserScan ):
        self .latest_scan =msg

    def image_cb (self ,msg :Image ):

        if self .state !="SPIN_SCAN":
            return
        if cv2 is None :
            return

        pose =self .get_robot_pose ()
        if pose is None :
            return

        bgr =self .rosimg_to_bgr (msg )
        if bgr is None :
            return


        now_t =time .time ()
        if (now_t -self ._last_vision_time )<self .vision_throttle_sec :
            return
        self ._last_vision_time =now_t


        if bgr .shape [1 ]>self .vision_resize_width :
            scale =float (self .vision_resize_width )/float (bgr .shape [1 ])
            new_h =max (1 ,int (bgr .shape [0 ]*scale ))
            bgr =cv2 .resize (bgr ,(self .vision_resize_width ,new_h ),interpolation =cv2 .INTER_AREA )

        det =self .detect_ball_color_and_bearing (bgr ,self .camera_hfov_deg )
        if det is None :
            self .det_hold_count =0
            self .det_last_color =None
            return

        color ,bearing =det


        if color in self .found :
            self .det_hold_count =0
            self .det_last_color =None
            return


        if self .det_last_color ==color and abs (bearing -self .det_last_bearing )<0.20 :
            self .det_hold_count +=1
            self .get_logger ().info (
            f"Vision candidate: color={color} bearing={math.degrees(bearing):.1f}deg hold={self.det_hold_count}/{self.det_need_count}",
            throttle_duration_sec =0.5 ,
            )
        else :
            self .det_hold_count =1
            self .det_last_color =color
            self .get_logger ().info (
            f"Vision first hit: color={color} bearing={math.degrees(bearing):.1f}deg",
            throttle_duration_sec =0.5 ,
            )

        self .det_last_bearing =bearing

        if self .det_hold_count >=self .det_need_count :
            with self ._det_lock :
                self .last_detection =(color ,bearing ,pose )


    def timer_cb (self ):
        if self .map_msg is None or self .map_data is None :
            self .get_logger ().info ("Waiting for map...",throttle_duration_sec =5.0 )
            return

        if self .latest_scan is None :
            self .get_logger ().info ("Waiting for LaserScan...",throttle_duration_sec =5.0 )
            self .stop_robot ()
            return

        pose =self .get_robot_pose ()
        if pose is None :
            self .get_logger ().info ("Waiting for TF (map -> base_footprint)...",throttle_duration_sec =5.0 )
            return


        if all (c in self .found for c in ["red","green","blue"]):
            self .stop_robot ()
            self .get_logger ().info ("All 3 colors found. DONE.")
            return

        x ,y ,yaw =pose


        if self .current_goal is None and self .state in ["NAV_PLAN","NAV_FOLLOW"]:
            self .current_goal =self .waypoints [self .wp_index ]
            self .current_wp =self .current_goal
            self .after_nav_state ="SPIN_SCAN"
            self .state ="NAV_PLAN"


        if self .state =="NAV_PLAN":
            if self .current_goal is None :
                return
            gx ,gy =self .current_goal
            path =self .plan_a_star ((x ,y ),(gx ,gy ))

            if path is None or len (path )<2 :
                self .get_logger ().warn ("A* failed for goal. Skipping to next waypoint.")
                self .stop_robot ()


                if self .after_nav_state =="MEASURE_BALL":
                    self .go_return_to_waypoint ()
                    return

                self .advance_waypoint ()
                self .current_goal =self .waypoints [self .wp_index ]
                self .current_wp =self .current_goal
                self .after_nav_state ="SPIN_SCAN"
                self .state ="NAV_PLAN"
                return

            self .current_path =path
            self .current_path_index =0
            self .publish_current_path ()
            self .aligning_to_path =True
            self .state ="NAV_FOLLOW"
            return

        if self .state =="NAV_FOLLOW":
            reached =self .follow_path_step (x ,y ,yaw )
            if reached :
                self .stop_robot ()

                if self .after_nav_state =="SPIN_SCAN":

                    self .spin_accum =0.0
                    self .spin_prev_yaw =yaw
                    self .last_detection =None
                    self .det_hold_count =0
                    self .det_last_color =None
                    self .state ="SPIN_SCAN"
                    self .set_vision_active (True )
                    return

                if self .after_nav_state =="MEASURE_BALL":

                    self .state ="MEASURE_BALL"
                    return

                if self .after_nav_state =="RESUME_AFTER_RETURN":

                    self .advance_waypoint ()
                    self .current_goal =self .waypoints [self .wp_index ]
                    self .current_wp =self .current_goal
                    self .after_nav_state ="SPIN_SCAN"
                    self .state ="NAV_PLAN"
                    return

                self .state ="NAV_PLAN"
            return

        if self .state =="SPIN_SCAN":

            det =None
            with self ._det_lock :
                det =self .last_detection

                self .last_detection =None

            if det is not None :
                color ,bearing ,pose0 =det
                if color in self .found :
                    self .last_detection =None
                    return

                x0 ,y0 ,yaw0 =pose0

                r0 =self .get_range_at_bearing (bearing ,window_deg =4.0 )
                if r0 is None :
                    self .last_detection =None
                    return


                r_left =self .get_range_at_bearing (bearing +math .radians (10.0 ),window_deg =4.0 )
                r_right =self .get_range_at_bearing (bearing -math .radians (10.0 ),window_deg =4.0 )


                if r0 >3.5 or r0 <0.12 :
                    self .last_detection =None
                    return

                if r_left is None or r_right is None :
                    self .last_detection =None
                    return

                protrusion =min (r_left ,r_right )-r0
                if protrusion <0.18 :

                    self .last_detection =None
                    return

                theta0 =yaw0 +bearing
                tx =x0 +r0 *math .cos (theta0 )
                ty =y0 +r0 *math .sin (theta0 )

                self .stop_robot ()

                self .set_vision_active (False )

                self .target_color =color
                self .target_point =(tx ,ty )


                self .current_goal =self .target_point
                self .after_nav_state ="MEASURE_BALL"
                self .current_path =[]
                self .aligning_to_path =False
                self .emergency_backing =False
                self .emergency_back_steps =0
                self .state ="NAV_PLAN"

                self .get_logger ().info (f"Detected {color} -> A* approach to ({tx:.2f}, {ty:.2f}), r0={r0:.2f}")
                return


            tw =Twist ()
            tw .linear .x =0.0
            tw .angular .z =self .spin_omega
            self .cmd_pub .publish (tw )

            dyaw =self .normalize_angle (yaw -self .spin_prev_yaw )
            self .spin_accum +=dyaw
            self .spin_prev_yaw =yaw

            if abs (self .spin_accum )>=self .spin_target :

                self .stop_robot ()
                self .set_vision_active (False )
                self .last_detection =None
                self .det_hold_count =0
                self .det_last_color =None

                self .advance_waypoint ()
                self .current_goal =self .waypoints [self .wp_index ]
                self .current_wp =self .current_goal
                self .after_nav_state ="SPIN_SCAN"
                self .state ="NAV_PLAN"
            return

        if self .state =="MEASURE_BALL":
            c =self .target_color
            tp =self .target_point
            if c is None or tp is None :
                self .go_return_to_waypoint ()
                return


            desired =math .atan2 (tp [1 ]-y ,tp [0 ]-x )
            bearing =self .normalize_angle (desired -yaw )

            if abs (bearing )>0.12 :
                tw =Twist ()
                tw .linear .x =0.0
                tw .angular .z =float (np .clip (1.5 *bearing ,-0.6 ,0.6 ))
                self .cmd_pub .publish (tw )
                return
            else :
                self .stop_robot ()

            r =self .get_range_at_bearing (bearing ,window_deg =4.0 )
            if r is None :
                return

            obj_x =x +r *math .cos (yaw +bearing )
            obj_y =y +r *math .sin (yaw +bearing )

            self .ball_samples [c ].append ((obj_x ,obj_y ))
            if len (self .ball_samples [c ])>25 :
                self .ball_samples [c ].pop (0 )

            if len (self .ball_samples [c ])>=self .min_samples_to_confirm :
                xs =np .array ([p [0 ]for p in self .ball_samples [c ]])
                ys =np .array ([p [1 ]for p in self .ball_samples [c ]])
                xm =float (np .median (xs ))
                ym =float (np .median (ys ))

                self .found [c ]=(xm ,ym )
                self .publish_ball (c ,xm ,ym )
                self .get_logger ().info (f"CONFIRMED {c}: ({xm:.2f}, {ym:.2f})")


                self .go_return_to_waypoint ()
            return


    def advance_waypoint (self ):
        if not self .waypoints :
            return
        self .wp_index +=1
        if self .wp_index >=len (self .waypoints ):
            self .wp_index =0 if self .loop_waypoints else (len (self .waypoints )-1 )

    def go_return_to_waypoint (self ):
        if self .current_wp is None :
            self .advance_waypoint ()
            self .current_goal =self .waypoints [self .wp_index ]
            self .current_wp =self .current_goal
            self .after_nav_state ="SPIN_SCAN"
            self .state ="NAV_PLAN"
            return

        self .return_wp =self .current_wp
        self .current_goal =self .return_wp
        self .after_nav_state ="RESUME_AFTER_RETURN"
        self .current_path =[]
        self .aligning_to_path =False
        self .emergency_backing =False
        self .emergency_back_steps =0
        self .state ="NAV_PLAN"

        self .get_logger ().info (f"Returning to waypoint: ({self.return_wp[0]:.2f}, {self.return_wp[1]:.2f})")


        self .target_point =None
        self .target_color =None


    @staticmethod
    def detect_ball_color_and_bearing (bgr :np .ndarray ,hfov_deg :float )->Optional [Tuple [str ,float ]]:
        if bgr is None or bgr .size ==0 :
            return None
        h ,w =bgr .shape [:2 ]
        cx =w *0.5

        blur =cv2 .GaussianBlur (bgr ,(5 ,5 ),0 )
        hsv =cv2 .cvtColor (blur ,cv2 .COLOR_BGR2HSV )


        red1 =cv2 .inRange (hsv ,(0 ,120 ,80 ),(10 ,255 ,255 ))
        red2 =cv2 .inRange (hsv ,(170 ,120 ,80 ),(180 ,255 ,255 ))
        mask_red =cv2 .bitwise_or (red1 ,red2 )


        mask_green =cv2 .inRange (hsv ,(40 ,90 ,80 ),(85 ,255 ,255 ))
        mask_blue =cv2 .inRange (hsv ,(95 ,120 ,80 ),(135 ,255 ,255 ))


        b ,g ,r =cv2 .split (blur )
        dom_red =cv2 .inRange (r -cv2 .max (g ,b ),40 ,255 )
        dom_green =cv2 .inRange (g -cv2 .max (r ,b ),40 ,255 )
        dom_blue =cv2 .inRange (b -cv2 .max (r ,g ),40 ,255 )

        kernel =np .ones ((5 ,5 ),np .uint8 )
        mask_red =cv2 .morphologyEx (mask_red ,cv2 .MORPH_OPEN ,kernel )
        mask_red =cv2 .morphologyEx (mask_red ,cv2 .MORPH_CLOSE ,kernel )
        mask_green =cv2 .morphologyEx (mask_green ,cv2 .MORPH_OPEN ,kernel )
        mask_green =cv2 .morphologyEx (mask_green ,cv2 .MORPH_CLOSE ,kernel )
        mask_blue =cv2 .morphologyEx (mask_blue ,cv2 .MORPH_OPEN ,kernel )
        mask_blue =cv2 .morphologyEx (mask_blue ,cv2 .MORPH_CLOSE ,kernel )

        img_area =float (w *h )
        max_area_ratio_by_color ={"red":0.05 ,"green":0.08 ,"blue":0.08 }
        edge_margin_px =6
        max_radius_px =0.30 *float (min (w ,h ))
        min_radius_px =5
        circle_fill_min_by_color ={"red":0.32 ,"green":0.30 ,"blue":0.30 }
        circle_ratio_min_by_color ={"red":0.30 ,"green":0.25 ,"blue":0.25 }
        circle_ratio_max =1.40


        best =None
        best_score =0.0

        for color ,mask in [("red",mask_red ),("green",mask_green ),("blue",mask_blue )]:
            contours ,_ =cv2 .findContours (mask ,cv2 .RETR_EXTERNAL ,cv2 .CHAIN_APPROX_SIMPLE )
            if not contours :
                continue

            for cnt in contours :
                area =float (cv2 .contourArea (cnt ))
                if area <250.0 :
                    continue

                max_area =img_area *max_area_ratio_by_color .get (color ,0.08 )
                if area >max_area :
                    continue

                perim =float (cv2 .arcLength (cnt ,True ))+1e-6
                circularity =4.0 *math .pi *area /(perim *perim )
                if circularity <0.55 :
                    continue

                x ,y ,bw ,bh =cv2 .boundingRect (cnt )

                if (x <=edge_margin_px or y <=edge_margin_px or
                (x +bw )>=(w -edge_margin_px )or (y +bh )>=(h -edge_margin_px )):

                    continue
                fill =area /float (bw *bh +1e-6 )
                if fill <0.40 :
                    continue

                ar =bw /(bh +1e-6 )
                if ar <0.55 or ar >1.80 :
                    continue


                hull =cv2 .convexHull (cnt )
                hull_area =float (cv2 .contourArea (hull ))+1e-6
                solidity =area /hull_area
                if solidity <0.70 :
                    continue

                (mcx ,mcy ),radius =cv2 .minEnclosingCircle (cnt )

                if radius <min_radius_px or radius >max_radius_px :
                    continue


                circle_area =math .pi *float (radius )*float (radius )
                circle_ratio =float (area )/(circle_area +1e-6 )
                if circle_ratio <circle_ratio_min_by_color .get (color ,0.25 )or circle_ratio >circle_ratio_max :
                    continue


                circle_mask =np .zeros_like (mask )
                cv2 .circle (circle_mask ,(int (mcx ),int (mcy )),int (radius ),255 ,-1 )
                inside =cv2 .bitwise_and (mask ,circle_mask )
                circle_fill =float (cv2 .countNonZero (inside ))/(float (cv2 .countNonZero (circle_mask ))+1e-6 )
                if circle_fill <circle_fill_min_by_color .get (color ,0.30 ):
                    continue

                score =area *circularity
                if score >best_score :
                    best_score =score

                    hfov =math .radians (float (hfov_deg ))
                    norm =(mcx -cx )/max (cx ,1e-6 )
                    bearing =float (norm *(hfov *0.5 ))
                    best =(color ,bearing )

        return best

    def rosimg_to_bgr (self ,msg :Image )->Optional [np .ndarray ]:
        if msg .height ==0 or msg .width ==0 :
            return None
        if cv2 is None :
            return None

        enc =msg .encoding .lower ()
        arr =np .frombuffer (msg .data ,dtype =np .uint8 )

        try :
            if "bgr8"in enc :
                return arr .reshape ((msg .height ,msg .width ,3 ))
            if "rgb8"in enc :
                img =arr .reshape ((msg .height ,msg .width ,3 ))
                return cv2 .cvtColor (img ,cv2 .COLOR_RGB2BGR )
            if "bgra8"in enc :
                img =arr .reshape ((msg .height ,msg .width ,4 ))
                return cv2 .cvtColor (img ,cv2 .COLOR_BGRA2BGR )
            if "rgba8"in enc :
                img =arr .reshape ((msg .height ,msg .width ,4 ))
                return cv2 .cvtColor (img ,cv2 .COLOR_RGBA2BGR )
        except Exception :
            return None
        return None


    def publish_ball (self ,color :str ,x :float ,y :float ):
        msg =PointStamped ()
        msg .header .frame_id ="map"
        msg .header .stamp =self .get_clock ().now ().to_msg ()
        msg .point .x =float (x )
        msg .point .y =float (y )
        msg .point .z =0.0

        if color =="red":
            self .red_pub .publish (msg )
        elif color =="green":
            self .green_pub .publish (msg )
        elif color =="blue":
            self .blue_pub .publish (msg )


    def get_robot_pose (self )->Optional [Tuple [float ,float ,float ]]:
        try :
            trans =self .tf_buffer .lookup_transform ("map","base_footprint",Time ())
        except Exception :
            return None

        x =trans .transform .translation .x
        y =trans .transform .translation .y
        qx =trans .transform .rotation .x
        qy =trans .transform .rotation .y
        qz =trans .transform .rotation .z
        qw =trans .transform .rotation .w

        siny_cosp =2.0 *(qw *qz +qx *qy )
        cosy_cosp =1.0 -2.0 *(qy *qy +qz *qz )
        yaw =math .atan2 (siny_cosp ,cosy_cosp )
        return (x ,y ,yaw )

    def get_range_at_bearing (self ,bearing_rad :float ,window_deg :float =4.0 )->Optional [float ]:
        s =self .latest_scan
        if s is None :
            return None

        window =math .radians (window_deg )
        a0 =bearing_rad -window
        a1 =bearing_rad +window

        i0 =int ((a0 -s .angle_min )/s .angle_increment )
        i1 =int ((a1 -s .angle_min )/s .angle_increment )
        i0 ,i1 =max (0 ,min (i0 ,i1 )),min (len (s .ranges )-1 ,max (i0 ,i1 ))

        vals =[]
        for r in s .ranges [i0 :i1 +1 ]:
            if math .isfinite (r )and r >0.05 :
                vals .append (r )
        if not vals :
            return None
        return float (np .median (vals ))


    def stop_robot (self ):
        self .cmd_pub .publish (Twist ())
        self .current_speed =0.0

    def world_to_map (self ,x :float ,y :float )->Optional [Tuple [int ,int ]]:
        mx =int ((x -self .map_origin_x )/self .map_resolution )
        my =int ((y -self .map_origin_y )/self .map_resolution )
        if mx <0 or my <0 or mx >=self .map_width or my >=self .map_height :
            return None
        return mx ,my

    def map_to_world (self ,mx :float ,my :float )->Tuple [float ,float ]:
        x =self .map_origin_x +(mx +0.5 )*self .map_resolution
        y =self .map_origin_y +(my +0.5 )*self .map_resolution
        return x ,y

    def remove_single_pixel_obstacle_noise (self ,min_cluster_size :int =2 ):
        if self .map_data is None :
            return

        w ,h =self .map_width ,self .map_height
        visited =[[False for _ in range (w )]for _ in range (h )]
        removed =0

        for my in range (h ):
            for mx in range (w ):
                if visited [my ][mx ]:
                    continue
                idx =my *w +mx
                if self .map_data [idx ]<=50 :
                    continue

                queue =[(mx ,my )]
                visited [my ][mx ]=True
                cluster :List [Tuple [int ,int ]]=[]

                while queue :
                    cx ,cy =queue .pop (0 )
                    cluster .append ((cx ,cy ))
                    for dx ,dy in ((1 ,0 ),(-1 ,0 ),(0 ,1 ),(0 ,-1 )):
                        nx ,ny =cx +dx ,cy +dy
                        if nx <0 or ny <0 or nx >=w or ny >=h :
                            continue
                        if visited [ny ][nx ]:
                            continue
                        nidx =ny *w +nx
                        if self .map_data [nidx ]<=50 :
                            continue
                        visited [ny ][nx ]=True
                        queue .append ((nx ,ny ))

                if len (cluster )<min_cluster_size :
                    for cx ,cy in cluster :
                        cidx =cy *w +cx
                        if self .map_data [cidx ]>50 :
                            self .map_data [cidx ]=0
                            removed +=1

        if removed >0 :
            self .get_logger ().info (
            f"Removed {removed} tiny obstacle cells (cluster size < {min_cluster_size})."
            )

    def mark_dynamic_obstacles_from_scan (self ,robot_x :float ,robot_y :float ,robot_yaw :float ):
        if self .latest_scan is None or self .map_data is None :
            return

        scan =self .latest_scan
        fov =math .radians (70.0 )
        added_cells =0

        for i ,r in enumerate (scan .ranges ):
            if not math .isfinite (r )or r <=0.05 :
                continue
            ang =scan .angle_min +i *scan .angle_increment
            if abs (ang )>fov :
                continue

            beam_yaw =robot_yaw +ang
            ex =robot_x +r *math .cos (beam_yaw )
            ey =robot_y +r *math .sin (beam_yaw )

            cell =self .world_to_map (ex ,ey )
            if cell is None :
                continue
            mx ,my =cell
            idx =my *self .map_width +mx
            v =self .map_data [idx ]

            if v !=0 :
                continue

            if (mx ,my )not in self .dynamic_obstacles :
                self .dynamic_obstacles .add ((mx ,my ))
                added_cells +=1

        if added_cells >0 :
            self .get_logger ().info (f"Marked {added_cells} dynamic obstacle cells from LaserScan.")

    def is_in_collision (self ,x :float ,y :float ,allow_unknown :bool =False )->bool :
        cell =self .world_to_map (x ,y )
        if cell is None :
            return True
        mx ,my =cell

        for dy in range (-self .inflation_cells ,self .inflation_cells +1 ):
            for dx in range (-self .inflation_cells ,self .inflation_cells +1 ):
                nx =mx +dx
                ny =my +dy
                if nx <0 or ny <0 or nx >=self .map_width or ny >=self .map_height :
                    return True

                idx =ny *self .map_width +nx
                v =self .map_data [idx ]
                if v >50 :
                    return True
                if v ==-1 and not allow_unknown :
                    return True
                if (nx ,ny )in self .dynamic_obstacles :
                    return True
        return False

    def compute_wall_penalty (self ,x :float ,y :float ,max_search_cells :int =8 ,safe_dist :float =0.35 )->float :
        if self .map_data is None :
            return 0.0
        cell =self .world_to_map (x ,y )
        if cell is None :
            return 0.0
        mx ,my =cell

        best_cell_dist =None
        for dy in range (-max_search_cells ,max_search_cells +1 ):
            ny =my +dy
            if ny <0 or ny >=self .map_height :
                continue
            for dx in range (-max_search_cells ,max_search_cells +1 ):
                nx =mx +dx
                if nx <0 or nx >=self .map_width :
                    continue
                idx =ny *self .map_width +nx
                if self .map_data [idx ]>50 :
                    d_cells =math .hypot (dx ,dy )
                    if best_cell_dist is None or d_cells <best_cell_dist :
                        best_cell_dist =d_cells

        if best_cell_dist is None :
            return 0.0

        d_m =best_cell_dist *self .map_resolution
        if d_m >=safe_dist :
            return 0.0
        return (safe_dist -d_m )/safe_dist

    def plan_a_star (self ,start :Tuple [float ,float ],goal :Tuple [float ,float ])->Optional [List [Tuple [float ,float ]]]:
        if self .map_data is None :
            return None

        sx ,sy =start
        gx ,gy =goal

        start_cell =self .world_to_map (sx ,sy )
        goal_cell =self .world_to_map (gx ,gy )
        if start_cell is None or goal_cell is None :
            return None

        sx_idx ,sy_idx =start_cell
        gx_idx ,gy_idx =goal_cell

        def cell_traversable (mx :int ,my :int )->bool :
            wx ,wy =self .map_to_world (mx ,my )
            return not self .is_in_collision (wx ,wy ,allow_unknown =True )


        if self .is_in_collision (sx ,sy ,allow_unknown =True ):
            best_start =None
            best_dist =float ("inf")
            search_r =max (self .inflation_cells *2 ,8 )
            for dy in range (-search_r ,search_r +1 ):
                for dx in range (-search_r ,search_r +1 ):
                    nx =sx_idx +dx
                    ny =sy_idx +dy
                    if nx <0 or ny <0 or nx >=self .map_width or ny >=self .map_height :
                        continue
                    if not cell_traversable (nx ,ny ):
                        continue
                    d =math .hypot (nx -sx_idx ,ny -sy_idx )
                    if d <best_dist :
                        best_dist =d
                        best_start =(nx ,ny )
            if best_start is None :
                return None
            sx_idx ,sy_idx =best_start


        if not cell_traversable (gx_idx ,gy_idx ):
            best_goal =None
            best_dist =float ("inf")
            search_r =max (self .inflation_cells *2 ,8 )
            for dy in range (-search_r ,search_r +1 ):
                for dx in range (-search_r ,search_r +1 ):
                    nx =gx_idx +dx
                    ny =gy_idx +dy
                    if nx <0 or ny <0 or nx >=self .map_width or ny >=self .map_height :
                        continue
                    if not cell_traversable (nx ,ny ):
                        continue
                    d =math .hypot (nx -gx_idx ,ny -gy_idx )
                    if d <best_dist :
                        best_dist =d
                        best_goal =(nx ,ny )
            if best_goal is None :
                return None
            gx_idx ,gy_idx =best_goal

        start_key =(sx_idx ,sy_idx )
        goal_key =(gx_idx ,gy_idx )

        moves =[
        (1 ,0 ,1.0 ),(-1 ,0 ,1.0 ),
        (0 ,1 ,1.0 ),(0 ,-1 ,1.0 ),
        (1 ,1 ,math .sqrt (2.0 )),(1 ,-1 ,math .sqrt (2.0 )),
        (-1 ,1 ,math .sqrt (2.0 )),(-1 ,-1 ,math .sqrt (2.0 )),
        ]

        open_heap :List [Tuple [float ,float ,int ,int ]]=[]
        heapq .heappush (open_heap ,(0.0 ,0.0 ,sx_idx ,sy_idx ))

        g_cost :Dict [Tuple [int ,int ],float ]={start_key :0.0 }
        came_from :Dict [Tuple [int ,int ],Tuple [int ,int ]]={}
        closed :Set [Tuple [int ,int ]]=set ()

        while open_heap :
            f ,cur_g ,mx ,my =heapq .heappop (open_heap )
            cur_key =(mx ,my )

            if cur_key in closed :
                continue
            closed .add (cur_key )

            if cur_key ==goal_key :

                cells =[]
                cur =goal_key
                while cur !=start_key :
                    cells .append (cur )
                    if cur not in came_from :
                        return None
                    cur =came_from [cur ]
                cells .append (start_key )
                cells .reverse ()

                path =[self .map_to_world (cx ,cy )for (cx ,cy )in cells ]
                if len (path )>2 :
                    thinned =[path [0 ]]
                    step_keep =3
                    for i in range (step_keep ,len (path )-1 ,step_keep ):
                        thinned .append (path [i ])
                    thinned .append (path [-1 ])
                    path =thinned
                return path

            for dx ,dy ,dist_cells in moves :
                nx =mx +dx
                ny =my +dy
                if nx <0 or ny <0 or nx >=self .map_width or ny >=self .map_height :
                    continue

                n_key =(nx ,ny )
                if n_key in closed :
                    continue

                if not cell_traversable (nx ,ny ):
                    continue

                step_dist =dist_cells *self .map_resolution
                wx ,wy =self .map_to_world (nx ,ny )
                wall_pen =self .compute_wall_penalty (wx ,wy )
                step_cost =step_dist +self .a_star_wall_weight *wall_pen

                new_g =cur_g +step_cost
                if new_g >=g_cost .get (n_key ,float ("inf")):
                    continue

                g_cost [n_key ]=new_g
                h =self .map_resolution *math .hypot (nx -gx_idx ,ny -gy_idx )
                new_f =new_g +h

                came_from [n_key ]=cur_key
                heapq .heappush (open_heap ,(new_f ,new_g ,nx ,ny ))

        return None

    def publish_current_path (self ):
        if not self .current_path :
            return

        path_msg =Path ()
        path_msg .header .stamp =self .get_clock ().now ().to_msg ()
        path_msg .header .frame_id ="map"

        for px ,py in self .current_path :
            ps =PoseStamped ()
            ps .header =path_msg .header
            ps .pose .position .x =float (px )
            ps .pose .position .y =float (py )
            ps .pose .position .z =0.0
            ps .pose .orientation .w =1.0
            path_msg .poses .append (ps )

        self .path_pub .publish (path_msg )

    def follow_path_step (self ,x :float ,y :float ,yaw :float )->bool :

        if self .emergency_backing :
            tw =Twist ()
            tw .linear .x =-0.1
            tw .angular .z =0.0
            self .cmd_pub .publish (tw )

            self .emergency_back_steps +=1
            if self .emergency_back_steps >=self .max_emergency_back_steps :
                self .emergency_backing =False
                self .stop_robot ()
                self .state ="NAV_PLAN"
            return False


        dyn_dist =self .compute_dynamic_front_threshold ()


        if self .latest_scan is not None :
            scan =self .latest_scan
            angle_min =scan .angle_min
            angle_inc =scan .angle_increment
            ranges =scan .ranges

            front_min =float ("inf")
            for i ,r in enumerate (ranges ):
                if not math .isfinite (r ):
                    continue
                ang =angle_min +i *angle_inc
                a =math .atan2 (math .sin (ang ),math .cos (ang ))
                if -math .radians (25.0 )<=a <=math .radians (25.0 ):
                    front_min =min (front_min ,r )

            if front_min <dyn_dist :
                self .get_logger ().warn (
                f"Obstacle too close (front_min={front_min:.2f} < {dyn_dist:.2f}). Marking + backing + replanning."
                )
                self .mark_dynamic_obstacles_from_scan (x ,y ,yaw )
                self .emergency_backing =True
                self .emergency_back_steps =0

                tw =Twist ()
                tw .linear .x =-0.1
                tw .angular .z =0.0
                self .cmd_pub .publish (tw )
                return False

        if not self .current_path :
            self .stop_robot ()
            return True


        goal_x ,goal_y =self .current_path [-1 ]
        if math .hypot (goal_x -x ,goal_y -y )<self .waypoint_tolerance :
            self .stop_robot ()
            return True


        closest_idx =0
        closest_dist =float ("inf")
        for i ,(px ,py )in enumerate (self .current_path ):
            d =math .hypot (px -x ,py -y )
            if d <closest_dist :
                closest_dist =d
                closest_idx =i


        lookahead_dist =0.4
        target_idx =len (self .current_path )-1
        for i in range (closest_idx ,len (self .current_path )):
            px ,py =self .current_path [i ]
            d =math .hypot (px -x ,py -y )
            if d >=lookahead_dist :
                target_idx =i
                break

        target_x ,target_y =self .current_path [target_idx ]
        dist_to_goal =math .hypot (goal_x -x ,goal_y -y )

        target_yaw =math .atan2 (target_y -y ,target_x -x )
        yaw_error =self .normalize_angle (target_yaw -yaw )

        tw =Twist ()


        if self .aligning_to_path :
            angle_thresh_start =math .radians (5.0 )
            if abs (yaw_error )>angle_thresh_start :
                tw .linear .x =0.0
                tw .angular .z =max (
                -self .max_angular_speed_align ,
                min (self .max_angular_speed_align ,self .k_w_align *yaw_error ),
                )
            else :
                self .aligning_to_path =False
                tw .linear .x =0.0
                tw .angular .z =0.0
            self .cmd_pub .publish (tw )
            return False


        if dist_to_goal <0.7 :
            v =min (self .max_linear_speed ,self .k_v *dist_to_goal )
        else :
            v =self .max_linear_speed

        angle_mid =math .radians (20.0 )
        angle_slow =math .radians (45.0 )
        angle_fast =math .radians (10.0 )

        if abs (yaw_error )>angle_slow :
            v *=0.45
        elif abs (yaw_error )>angle_mid :
            v *=0.7
        elif abs (yaw_error )>angle_fast :
            v *=0.8


        w =max (
        -self .max_angular_speed_travel ,
        min (self .max_angular_speed_travel ,self .k_w_travel *yaw_error ),
        )

        v =max (0.0 ,min (self .max_linear_speed ,v ))
        w =max (-self .max_angular_speed_travel ,min (self .max_angular_speed_travel ,w ))

        tw .linear .x =v
        tw .angular .z =w
        self .current_speed =v
        self .cmd_pub .publish (tw )
        return False

    def compute_dynamic_front_threshold (self )->float :
        v_safe =max (0.0 ,min (self .max_linear_speed ,self .current_speed ))
        dyn_dist =self .min_front_obstacle_distance +self .front_distance_gain *v_safe
        dyn_dist =max (self .min_front_obstacle_distance ,min (self .max_front_obstacle_distance ,dyn_dist ))
        return dyn_dist

    @staticmethod
    def normalize_angle (angle :float )->float :
        while angle >math .pi :
            angle -=2.0 *math .pi
        while angle <-math .pi :
            angle +=2.0 *math .pi
        return angle


def main (args =None ):
    rclpy .init (args =args )
    node =Task3 ()
    try :
        executor =MultiThreadedExecutor (num_threads =4 )
        executor .add_node (node )
        executor .spin ()
    except KeyboardInterrupt :
        pass
    finally :
        node .stop_robot ()
        node .destroy_node ()
        rclpy .shutdown ()


if __name__ =="__main__":
    main ()
