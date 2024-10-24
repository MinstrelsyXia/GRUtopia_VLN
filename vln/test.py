import math

def compute_focal_length(hfov_degrees, sensor_width):
    # Convert HFOV from degrees to radians
    hfov_radians = math.radians(hfov_degrees)
    
    # Calculate the focal length using the HFOV formula rearranged
    focal_length = sensor_width / (2 * math.tan(hfov_radians / 2))
    
    return focal_length

hfov_degrees = 90
sensor_width = 20.955
print(compute_focal_length(hfov_degrees, sensor_width)) # 27.0

class Test:
    def __init__(self):
        self.a = 1
    
    def get_a(self):
        return self.a 

test = Test()
a = test.get_a()
a += 1
print(a)
print(test.a)
