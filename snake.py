class Snake():
    x = 0
    y = 0
    height = 0
    width = 0
    speed = 0

    def __init__(self):
        self.x = 210
        self.y = 270
        self.height = 10
        self.width = 10
        self.speed = 20
    
    def reset(self):
        self.x = 210
        self.y = 270
        self.speed = 20