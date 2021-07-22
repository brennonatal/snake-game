import math

# Check if snake hit the food
def ate_food(food_x=None, food_y=None, food_ratio=None, snake_x=None, snake_y=None, snake_size=10):
    """Given the coordinates and dimensions of the food and the snake's head, 
    verify that the food has been eaten"""

    # Auxiliar variables to calculate distance
    test_x = food_x
    test_y = food_y

    # Check which is the closest axis
    if (food_x < snake_x):
        # test left edge
        test_x = snake_x    
    elif (food_x > snake_x + snake_size):
        # right edge
        test_x = snake_x + snake_size 
    if (food_y < snake_y):
        # top edge
        test_y = snake_y    
    elif (food_y > snake_y + snake_size):
        # bottom edge
        test_y = snake_y + snake_size 

    # Get distance from closest edges
    dist_x = food_x - test_x
    dist_y = food_y - test_y
    distance = math.sqrt((dist_x*dist_x) + (dist_y*dist_y))

    # If the distance is less than the food_ratio, the snake ate the food
    if (distance <= food_ratio):
        return True

    return False


def snake_collision(snake_head=None, snake_body=None):
    """Given the snake's head and body, check if it hit itself"""

    for cell in snake_body:
        if (snake_head.x < cell.x + cell.size and 
        snake_head.x + snake_head.size > cell.x and 
        snake_head.y < cell.y + cell.size and 
        snake_head.y + snake_head.size > cell.y):
            return True

