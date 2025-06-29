from modules.visual_memory import VisualMemory
from modules.localization import Localization
import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from scipy.spatial.transform import Rotation as R

class KeyBindings:
    FORWARD_KEY = "w"
    LEFT_KEY = "a"
    RIGHT_KEY = "d"
    FINISH_KEY = "f"
    MOVE_KEY = "m"
    RESET_VM_KEY = "k"
    NEXT_VM_KEY = "0"
    LOCALIZATION_KEY = "l"
    OBEY_KEY = "o"
    IBVS_KEY = "i"

class NavigationPolicy:
    def __init__(self, config=None, visual_memory=None, vm_len=0):
        self.config = config
        #self.registration_result = registration_result
        self.fit_threshold = config['navigation']['fit_threshold']
        self.forward_threshold = config['navigation']['forward_threshold']
        self.lateral_threshold = config['navigation']['lateral_threshold']
        self.yaw_threshold = config['navigation']['yaw_threshold']
        #self.vm_path = config['paths']['VM_PATH']
        self.visual_memory = visual_memory or VisualMemory(config)
        #self.localization = Localization(config)
        self.transformation=np.eye(4)
        self.fitness=0
        self.setup_fuzzy_control()
        self.vm_len = vm_len
        self.vel = [0,0,0,0,0,0]
        #self.vs_action = None
        self.suggested_action = None
        self.ibvs_action = None

    def set_vm_len(self, vm_len):
        self.vm_len = vm_len

    def set_suggested_action(self, action):
        self.suggested_action = action
    
    def set_ibvs_action(self, action):
        self.ibvs_action = action
    
    def set_velocity(self, vel):
        self.vel = vel
    
    #def set_vs_action(self, vs_action):
    #    self.vs_action = vs_action

    def setup_fuzzy_control(self):
        # Define the universe of discourse for inputs and outputs using config parameters
        position_error = ctrl.Antecedent(np.arange(-self.config['fuzzy_navigation']['max_position_error'], 
                                                self.config['fuzzy_navigation']['max_position_error'], 
                                                self.config['fuzzy_navigation']['error_resolution']), 'position_error')
        orientation_error = ctrl.Antecedent(np.arange(-self.config['fuzzy_navigation']['max_orientation_error'], 
                                                    self.config['fuzzy_navigation']['max_orientation_error'], 
                                                    self.config['fuzzy_navigation']['error_resolution']), 'orientation_error')
        action = ctrl.Consequent(np.arange(0, 4, 1), 'action')  # 0: Stop, 1: Move Forward, 2: Turn Left, 3: Turn Right

        # Membership functions setup using config thresholds from 'fuzzy_navigation'
        position_error['Near'] = fuzz.trimf(position_error.universe, [self.config['fuzzy_navigation']['position_error']['Near']['start'], 
                                                                    self.config['fuzzy_navigation']['position_error']['Near']['peak'], 
                                                                    self.config['fuzzy_navigation']['position_error']['Near']['end']])
        position_error['Far'] = fuzz.trimf(position_error.universe, [self.config['fuzzy_navigation']['position_error']['Far']['start'], 
                                                                    self.config['fuzzy_navigation']['position_error']['Far']['peak'], 
                                                                    self.config['fuzzy_navigation']['position_error']['Far']['end']])
        orientation_error['Small'] = fuzz.trimf(orientation_error.universe, [self.config['fuzzy_navigation']['orientation_error']['Small']['start'], 
                                                                            self.config['fuzzy_navigation']['orientation_error']['Small']['peak'], 
                                                                            self.config['fuzzy_navigation']['orientation_error']['Small']['end']])
        orientation_error['Moderate'] = fuzz.trimf(orientation_error.universe, [self.config['fuzzy_navigation']['orientation_error']['Moderate']['start'], 
                                                                                self.config['fuzzy_navigation']['orientation_error']['Moderate']['peak'], 
                                                                                self.config['fuzzy_navigation']['orientation_error']['Moderate']['end']])
        orientation_error['Large'] = fuzz.trimf(orientation_error.universe, [self.config['fuzzy_navigation']['orientation_error']['Large']['start'], 
                                                                            self.config['fuzzy_navigation']['orientation_error']['Large']['peak'], 
                                                                            self.config['fuzzy_navigation']['orientation_error']['Large']['end']])
        
        action['Stop'] = fuzz.trimf(action.universe, [0, 0, 1])
        action['Move Forward'] = fuzz.trimf(action.universe, [1, 1, 2])
        action['Turn Left'] = fuzz.trimf(action.universe, [2, 2, 3])
        action['Turn Right'] = fuzz.trimf(action.universe, [3, 3, 3])

        # Fuzzy rules
        rule1 = ctrl.Rule(position_error['Near'] & orientation_error['Small'], action['Stop'])
        rule2 = ctrl.Rule(position_error['Far'] & orientation_error['Small'], action['Move Forward'])
        rule3 = ctrl.Rule(orientation_error['Large'], action['Turn Right'])
        rule4 = ctrl.Rule(orientation_error['Large'], action['Turn Left'])

        # Control system
        self.action_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.action_decision = ctrl.ControlSystemSimulation(self.action_control)

    def print_success(self, message):
        green = "\033[92m"
        reset = "\033[0m"
        print(f"{green}Success: {message}{reset}")

    def print_warning(self, message):
        red = "\033[91m"
        reset = "\033[0m"
        print(f"{red}Warning: {message}{reset}")

    def determine_bot_action(self, transformation, fitness):
        """
        Determine the action a bot should take based on the transformation matrix.
        Args:
        T (np.array): A 4x4 transformation matrix containing rotation and translation.
        Returns:
        str: The action the bot should take: 'Move Forward', 'Turn Right', 'Turn Left', or 'Stop'.
        """
        self.transformation = transformation
        self.fitness = fitness

        if self.fitness < self.fit_threshold:
            self.print_warning("Warning: Regitration failed. Fitness score: {}".format(self.fitness))
        else:
            self.print_success("Registration successful. Fitness score: {}".format(self.fitness))

        # Extract the translation vector and Euler angles
        print('Processing action')
        T = np.copy(self.transformation)
        translation = T[0:3, 3]
        rotation_matrix = T[0:3, 0:3]
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        
        print(f'Translation: {translation}')
        print(f'Angles (xyz): {euler_angles}')

        # Check translation for forward/backward movement
        if translation[0] < -self.forward_threshold:
            action_forward = 'Move Forward'
        else:
            action_forward = 'Stop'  # If the bot is close enough to the target

        # Check lateral translation and yaw angle for turning
        if translation[1] < -self.lateral_threshold or euler_angles[2] < -self.yaw_threshold:
            action_turn = 'Turn Right'
        elif translation[1] > self.lateral_threshold or euler_angles[2] > self.yaw_threshold:
            action_turn = 'Turn Left'
        else:
            action_turn = None  # No turn is needed if within thresholds

        # Combine actions: prioritize turning over moving forward
        if action_turn:
            return action_turn
        else:
            return action_forward
        
    def handle_keystroke(self, keystroke, vm_image_index):
        action_executed = None  # Indicates whether an action was executed ('suggested', 'vs', or None)

        if keystroke == ord(KeyBindings.OBEY_KEY):
            if self.suggested_action == 'forward':
                action = HabitatSimActions.move_forward
            elif self.suggested_action == 'left':
                action = HabitatSimActions.turn_left
            elif self.suggested_action == 'right':
                action = HabitatSimActions.turn_right
            elif self.suggested_action == 'update':
                if vm_image_index == self.vm_len - 1:
                    print("Finishing the episode.")
                    return vm_image_index, "finish", 'suggested'  # Signal to finish the episode
                else:
                    vm_image_index += 1
                return vm_image_index, None, 'suggested'  # No action to execute
            else:
                print("Invalid suggested action: {}".format(self.suggested_action))
                action = HabitatSimActions.stop
            action_executed = 'suggested'

        elif keystroke == ord(KeyBindings.IBVS_KEY):
            if self.ibvs_action == 'forward':
                action = HabitatSimActions.move_forward
            elif self.ibvs_action == 'left':
                action = HabitatSimActions.turn_left
            elif self.ibvs_action == 'right':
                action = HabitatSimActions.turn_right
            elif self.ibvs_action == 'update':
                if vm_image_index == self.vm_len - 1:
                    print("Finishing the episode.")
                    return vm_image_index, "finish", 'vs'  # Signal to finish the episode
                else:
                    vm_image_index += 1
                return vm_image_index, None, 'vs'  # No action to execute
            else:
                print("Invalid ibvs_action: {}".format(self.ibvs_action))
                action = HabitatSimActions.stop
            action_executed = 'vs'

        elif keystroke == ord(KeyBindings.FORWARD_KEY):
            action = HabitatSimActions.move_forward
            action_executed = 'manual'

        elif keystroke == ord(KeyBindings.LEFT_KEY):
            action = HabitatSimActions.turn_left
            action_executed = 'manual'

        elif keystroke == ord(KeyBindings.RIGHT_KEY):
            action = HabitatSimActions.turn_right
            action_executed = 'manual'

        elif keystroke == ord(KeyBindings.FINISH_KEY):
            print("Finishing the episode.")
            return vm_image_index, "finish", None  # Signal to finish the episode

        elif keystroke == ord(KeyBindings.RESET_VM_KEY):
            vm_image_index = 0
            return vm_image_index, None, None  # No action to execute

        elif keystroke == ord(KeyBindings.NEXT_VM_KEY):
            vm_image_index += 1
            return vm_image_index, None, None  # No action to execute

        elif keystroke == ord(KeyBindings.LOCALIZATION_KEY):
            print("Localization not reimplemented yet.")
            return vm_image_index, None, None  # No action to execute after localization

        else:
            return vm_image_index, None, None  # No action for unrecognized keystrokes

        # For actions that involve moving the agent
        return vm_image_index, action, action_executed

    
    def handle_keystroke_old(self, keystroke, vm_image_index, transformation, fitness):#, current_color_image=None):
        self.transformation = transformation
        self.fitness = fitness

        if keystroke == ord(KeyBindings.OBEY_KEY):
            if self.action == 'forward':
                action = HabitatSimActions.move_forward
            elif self.action == 'left':
                action = HabitatSimActions.turn_left
            elif self.action == 'right':
                action = HabitatSimActions.turn_right
            elif self.action =='update':
                if vm_image_index == self.vm_len-1:
                    print("Finishing the episode.")
                    return vm_image_index, "finish"  # Signal to finish the episode
                else:
                    vm_image_index += 1
                return vm_image_index, None  # No action to execute
            else:
                print("Invalid action: {}".format(self.action))
                action = HabitatSimActions.stop
        
        elif keystroke == ord(KeyBindings.IBVS_KEY):
            # Something should happen here
            pass

        elif keystroke == ord(KeyBindings.FORWARD_KEY):
            action = HabitatSimActions.move_forward

        elif keystroke == ord(KeyBindings.LEFT_KEY):
            action = HabitatSimActions.turn_left

        elif keystroke == ord(KeyBindings.RIGHT_KEY):
            action = HabitatSimActions.turn_right

        elif keystroke == ord(KeyBindings.FINISH_KEY):
            print("Finishing the episode.")
            return vm_image_index, "finish"  # Signal to finish the episode

        elif keystroke == ord(KeyBindings.RESET_VM_KEY):
            vm_image_index = 0
            #self.visual_memory.display_visual_memory(vm_image_index)
            return vm_image_index, None  # No action to execute

        elif keystroke == ord(KeyBindings.NEXT_VM_KEY):
            vm_image_index = (vm_image_index + 1)# % len(os.listdir(self.vm_path + "color/"))
            #self.visual_memory.display_visual_memory(vm_image_index)
            return vm_image_index, None  # No action to execute
        
        elif keystroke == ord(KeyBindings.LOCALIZATION_KEY):
            # Call the localization method with the current color image
            print("Localization not reimplemented yet.")
            #best_match_name, best_distance_name, max_matches, min_avg_distance = self.localization.localization_in_visual_memory()
            #print(f"Best match image: {best_match_name} | Number of Matches: {max_matches}")
            #print(f"Best distance image: {best_distance_name} | Average Distance: {min_avg_distance:.2f}")

            return vm_image_index, None  # No action to execute after localization

        else:
            return vm_image_index, None  # No action for unrecognized keystrokes

        # For actions that involve moving the agent
        return vm_image_index, action
    '''
    def fuzzy_bot_action(self, registration_result):
        """
        Determine the action a bot should take based on the transformation matrix.
        Args:
        T (np.array): A 4x4 transformation matrix containing rotation and translation.
        Returns:
        str: The action the bot should take: 'Move Forward', 'Turn Right', 'Turn Left', or 'Stop'.
        """
        self.registration_result = registration_result

        if self.registration_result.fitness < self.fit_threshold:
            self.print_warning("Warning: Regitration failed. Fitness score: {}".format(self.registration_result.fitness))
        else:
            self.print_success("Registration successful. Fitness score: {}".format(self.registration_result.fitness))

        # Extract the translation vector and Euler angles
        print('Processing action')
        T = np.copy(self.registration_result.transformation)
        translation = T[0:3, 3]
        rotation_matrix = T[0:3, 0:3]
        euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        
        print(f'Translation: {translation}')
        print(f'Angles (xyz): {euler_angles}')

        # Convert translation and rotation to position and orientation errors
        position_error_value = np.linalg.norm(translation)  # Simple distance for position error
        orientation_error_value = np.abs(euler_angles[2])  # Absolute yaw angle for orientation error

        # Fuzzy inference
        self.action_decision.input['position_error'] = position_error_value
        self.action_decision.input['orientation_error'] = orientation_error_value
        self.action_decision.compute()

        # Mapping the crisp action output to action commands
        action_value = self.action_decision.output['action']
        if action_value <= 0.5:
            return 'Stop'
        elif action_value <= 1.5:
            return 'Move Forward'
        elif action_value <= 2.5:
            return 'Turn Left'
        else:
            return 'Turn Right'
        '''
