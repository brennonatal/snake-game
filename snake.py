class SnakeCell():
    x = 0
    y = 0
    size = 0

    def __init__(self, x=300, y=200):
        self.x = x
        self.y = y
        self.size = 10
    
    def reset(self):
        self.x = 300
        self.y = 200