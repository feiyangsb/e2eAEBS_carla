import sys
import logging
import os
import queue
import random
from scripts.engines.viewer import pygameViewer
from scripts.engines.dynamic_weather import DynamicPrecipitation
from scripts.engines.distance_calculation import DistanceCalculation
from scripts.engines.pid import PID
import math
import numpy as np
import cv2

import carla
from carla import Transform, Location, Rotation

class SetupWorld():
    def __init__(self, host='127.0.0.1', port=2000, town=1, gui=False):
        self.episode = 0
        self.gui = gui
        # try to import carla python API
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town0{}'.format(town))
        self.episode_count = 0
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.actor = []
        self.display = None
    
    def reset(self, initial_distance, initial_speed):
        if self.gui and self.display:
            self.display.stop()

        # Clean all vehicles before reset the environment
        for i, _ in enumerate(self.actor):
            if self.actor[i] is not None:
                self.actor[i].destroy()
                self.actor[i] = None
        self.actor = []

        self.spawn_vehicles()
        self.dist_calc = DistanceCalculation(self.ego_vehicle, self.leading_vehicle)
        self.pid_controller = PID(P=3.0, I=0.0003, D=0.0)

        if self.gui:
            self.display = pygameViewer()
        
        self.step_count = 0
        self.weather = DynamicPrecipitation(initial_precipitation=round(np.random.uniform(0.0,20.0), 2))
        self.world.set_weather(self.weather.get_weather_parameters())
        
        S0 = self.reset_episode(initial_distance, initial_speed)
        self.episode_count += 1
        return S0
    
    def reset_episode(self, initial_distance, initial_speed):
        velocity = 0.0
        while True:
            weather_parameter = self.weather.get_weather_parameters()
            self.world.set_weather(weather_parameter)
            action = self.pid_controller.update(initial_speed, velocity)
            control = carla.VehicleControl(
                        throttle = action + 0.4,
                        brake = 0.0,
                        steer = 0.0
            )
            self.ego_vehicle.apply_control(control)
            self.world.tick()
            image = self.image_queue.get()
            if not self.collision_queue.empty():
                _ = self.collision_queue.get()
                print("collision occurred during reset...")
            distance = self.dist_calc.getTrueDistance()
            velocity = self.ego_vehicle.get_velocity().y
            if self.gui:
                self.display.updateViewer(image)
            
            if (distance<=initial_distance):
                break

        return [image, velocity]


    def spawn_vehicles(self):
        self.bp_lib = self.world.get_blueprint_library()
        ego_bp = self.bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_point = Transform(Location(x=392.1, y=10.0, z=0.02), Rotation(yaw=89.6))
        self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
        self.setup_sensors(self.ego_vehicle)
        self.actor.append(self.ego_vehicle)
        self.ego_vehicle.get_world()
        leading_bp = self.bp_lib.filter('vehicle.audi.a2')[0]
        #leading_bp = self.bp_lib.filter('vehicle.audi.etron')[0]
        #leading_bp = self.bp_lib.filter('vehicle.audi.tt')[0]
        #leading_bp = random.choice(self.bp_lib.filter('vehicle.audi.*'))
        spawn_point_leading = Transform(Location(x=392.1, y=320.0, z=0.0), Rotation(yaw=90))
        self.leading_vehicle = self.world.try_spawn_actor(leading_bp, spawn_point_leading)
        self.leading_vehicle.get_world()
        self.actor.append(self.leading_vehicle)
        """
        other_bp = self.bp_lib.filter('vehicle.audi.tt')[0]
        spawn_point_other = Transform(Location(x=396.1, y=270.0, z=0.1), Rotation(yaw=-90))
        self.other_vehicle = self.world.try_spawn_actor(other_bp, spawn_point_other)
        self.actor.append(self.other_vehicle)
        """
        """
        walker1_bp = random.choice(self.bp_lib.filter('walker.*'))
        self.walker = self.world.try_spawn_actor(walker1_bp, Transform(Location(x=388.1, y=280.0, z=3.0), Rotation(yaw=90)))
        self.actor.append(self.walker)
        """
    
    def setup_sensors(self, ego_vehicle):
        camera_bp = self.bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '100')
        self.camera = self.world.try_spawn_actor(
                        camera_bp,
                        Transform(Location(x=0.8,y=0.0,z=1.7), Rotation(yaw=0.0)), 
                        attach_to=ego_vehicle)
        self.actor.append(self.camera)
        collision_bp = self.bp_lib.find('sensor.other.collision')
        self.collision = self.world.try_spawn_actor(
                        collision_bp,
                        Transform(),
                        attach_to=self.ego_vehicle)
        self.actor.append(self.collision)
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)
        self.collision_queue = queue.Queue()
        self.collision.listen(self.collision_queue.put)
    
    def step(self, action):
        self.step_count += 1
        weather_parameter = self.weather.get_weather_parameters(step=self.step_count)
        self.world.set_weather(weather_parameter)
        control=carla.VehicleControl(
            throttle = 0.0,
            brake = action
        )
        self.ego_vehicle.apply_control(control)
        self.world.tick()
        image = self.image_queue.get()
        if not self.collision_queue.empty():
            collision = self.collision_queue.get().normal_impulse
            isCollision =True
        else:
            isCollision = False

        groundtruth_distance = self.dist_calc.getTrueDistance()
        velocity = self.ego_vehicle.get_velocity().y

        if self.gui:
            self.display.updateViewer(image)

        isStop = velocity <= 0.0
        done = isStop or isCollision

        self.image = image
            
        if done:
            if (isCollision):
                #reward = -math.sqrt(collision.x**2+collision.y**2+collision.z**2)/100.0
                reward = -200.0 - math.sqrt(collision.x**2+collision.y**2+collision.z**2)/100.0
                print("Collision: {}".format(reward))
            elif (isStop):
                #too_far_reward = -(distance>5.0)*(distance-5)#-((distance-5)/250.0*400 + 20) * (distance>5.0)
                too_far_reward = -((groundtruth_distance-3.0)/120.0*400+30) * (groundtruth_distance>3.0) 
                #too_close_reward = 0.0#-(20.0)*(distance<1.0)
                too_close_reward = -(40.0)*(groundtruth_distance<1.0)
                reward = too_far_reward + too_close_reward
                print("Stop: {}, Distance: {}".format(reward, groundtruth_distance))
            self.episode+=1
        else:
            reward = 0

        return [[image, velocity], reward, done] 
