from modules.navigation_policy import NavigationPolicy, KeyBindings
import numpy as np


def main():
    config = {
        'navigation': {
            'fit_threshold': 0.5,
            'forward_threshold': 0.1,
            'lateral_threshold': 0.1,
            'yaw_threshold': 5.0,
        },
        'fuzzy_navigation': {
            'max_position_error': 1.0,
            'max_orientation_error': 30.0,
            'error_resolution': 0.1,
            'position_error': {
                'Near': {'start': 0.0, 'peak': 0.25, 'end': 0.5},
                'Far': {'start': 0.5, 'peak': 1.0, 'end': 1.0},
            },
            'orientation_error': {
                'Small': {'start': 0.0, 'peak': 5.0, 'end': 10.0},
                'Moderate': {'start': 10.0, 'peak': 15.0, 'end': 20.0},
                'Large': {'start': 20.0, 'peak': 25.0, 'end': 30.0},
            },
        },
    }

    policy = NavigationPolicy(config=config, vm_len=3)
    policy.set_suggested_action('forward')

    # Simulate pressing the obey key
    index, action, origin = policy.handle_keystroke(ord(KeyBindings.OBEY_KEY), 0)

    print(f"VM index: {index}")
    print(f"Action: {action}")
    print(f"Origin: {origin}")


if __name__ == "__main__":
    main()
