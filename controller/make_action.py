import time
import keyboard
import numpy as np

def make_action(self, action, pass_duration=0.25, dribble_duration=0.5, shoot_duration=0.225):
    """
    Execute the specified game action.
    
    Args:
        action (int): The action to take in the game environment (0-16)
        pass_duration (float): Duration for pass actions
        dribble_duration (float): Duration for dribble actions
        shoot_duration (float): Duration for shoot actions
    """
    
    # Direction mappings for cleaner code
    DIRECTIONS = {
        'north': ['w'],
        'northeast': ['w', 'd'],
        'east': ['d'],
        'southeast': ['s', 'd'],
        'south': ['s'],
        'southwest': ['s', 'a'],
        'west': ['a'],
        'northwest': ['w', 'a']
    }
    
    DIRECTION_ORDER = ['north', 'northeast', 'east', 'southeast', 
                      'south', 'southwest', 'west', 'northwest']
    
    self.stop_movement_action(0)
    
    if 0 <= action <= 7:  # Passing actions
        self._execute_pass(action, pass_duration, DIRECTIONS, DIRECTION_ORDER)
    elif 8 <= action <= 15:  # Dribbling actions
        self._execute_dribble(action - 8, dribble_duration, DIRECTIONS, DIRECTION_ORDER)
    elif action == 16:  # Shooting action
        self._execute_shoot(shoot_duration)
    else:
        print(f"Unknown action: {action}")

def _execute_pass(self, direction_index, duration, directions, direction_order):
    """Execute a pass action in the specified direction."""
    direction_name = direction_order[direction_index]
    keys = directions[direction_name] + ['p']  # Add pass key
    
    self._press_keys(keys)
    time.sleep(duration)
    self._release_keys(keys)

def _execute_dribble(self, direction_index, duration, directions, direction_order):
    """Execute a dribble action in the specified direction."""
    direction_name = direction_order[direction_index]
    keys = directions[direction_name]
    
    self._press_keys(keys)
    time.sleep(duration)
    self._release_keys(keys)

def _execute_shoot(self, duration):
    """Execute a shooting action based on agent position."""
    FAR_POST = 49
    NEAR_POST = 41
    
    print(f"SHOOTING FROM {self._get_position_description()}")
    
    if self.agent_y > FAR_POST:
        shoot_direction = self._get_random_shoot_direction(['east', 'southeast'])
    elif self.agent_y < NEAR_POST:
        shoot_direction = self._get_random_shoot_direction(['east', 'northeast'])
    else:  # Middle position
        shoot_direction = self._get_random_shoot_direction(['northeast', 'southeast'])
    
    self._perform_aimed_shot(shoot_direction, duration)

def _get_position_description(self):
    """Get a description of the current shooting position."""
    FAR_POST = 49
    NEAR_POST = 41
    
    if self.agent_y > FAR_POST:
        return "THE FAR POST"
    elif self.agent_y < NEAR_POST:
        return "THE NEAR POST"
    else:
        return "THE MIDDLE"

def _get_random_shoot_direction(self, directions):
    """Randomly select a shooting direction from the given options."""
    return np.random.choice(directions)

def _perform_aimed_shot(self, direction, duration):
    """Perform a shot in the specified direction."""
    direction_keys = {
        'east': ['d'],
        'northeast': ['d', 'w'],
        'southeast': ['d', 's']
    }
    
    movement_keys = direction_keys.get(direction, ['d'])
    
    # Aim
    self._press_keys(movement_keys)
    time.sleep(0.2)
    
    # Shoot
    keyboard.press('k')
    time.sleep(duration)
    keyboard.release('k')
    
    # Stop aiming
    time.sleep(0.2)
    self._release_keys(movement_keys)

def _press_keys(self, keys):
    """Press multiple keys simultaneously."""
    for key in keys:
        keyboard.press(key)

def _release_keys(self, keys):
    """Release multiple keys in reverse order."""
    for key in reversed(keys):
        keyboard.release(key)