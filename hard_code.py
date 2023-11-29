import global_storage as gs

class HardCode():
    def __init__(self):
        self.duration_speed = 0
        self.duration_steer = 0
        self.speed = 0
        self.steer = 0
        
    def speed_hardcode(self, speed, duration):
        self.duration_speed = duration
        self.speed = speed * 50

    def steer_hardcode(self, steer, duration):
        self.duration_steer = duration
        self.speed = steer * 60
    
    def turn_left_hardcode(self):
        self.speed_hardcode(0.13, 50)
        self.steer_hardcode(-1, 50)
    
    def turn_right_hardcode(self):
        self.speed_hardcode(0.13, 50)
        self.steer_hardcode(1, 50)
        
    def style_left_hardcode(self):
        self.speed_hardcode(1, 200)
        self.steer_hardcode(-1, 200)
    
    def style_right_hardcode(self):
        self.speed_hardcode(1, 200)
        self.steer_hardcode(1, 200)
    
    def run(self):
        if self.duration_speed > 0:
            gs.speed = self.speed
        if self.duration_steer > 0:
            gs.steer = self.steer
        self.duration_speed -= 1
        self.duration_steer -= 1
        if self.duration_speed < 0:
            self.duration_speed = 0
        if self.duration_steer < 0:
            self.duration_steer = 0