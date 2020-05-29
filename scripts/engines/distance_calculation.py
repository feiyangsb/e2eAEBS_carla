import carla


class DistanceCalculation():
    def __init__(self, ego_vehicle, leading_vehicle):
        self.ego_vehicle = ego_vehicle
        self.leading_vehicle = leading_vehicle


    def getTrueDistance(self):
        distance = self.leading_vehicle.get_location().y - self.ego_vehicle.get_location().y \
                - self.ego_vehicle.bounding_box.extent.x - self.leading_vehicle.bounding_box.extent.x
        return distance 
    
        