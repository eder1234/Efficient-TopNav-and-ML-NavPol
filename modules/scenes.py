class Scene:
    """
    A class to represent a virtual environment scene in Habitat.

    Attributes:
        name (str): The name of the scene.
    """

    # Class-level dictionary containing scene configurations
    SCENE_CONFIG = {
        'room': {
            'path': "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb",
            'visual_threshold_m': 0.853,  # Example threshold value
            'visual_threshold_m1sd': 0.9199,  # Example threshold value
            'visual_threshold_m2sd': 0.9869,  # Example threshold value
            'visual_threshold_m-1sd': 0.7861,  # Example threshold value
            # Add more properties as needed
        },
        'apartment': {
            'path': "data/scene_datasets/habitat-test-scenes/apartment_1.glb",
            'visual_threshold_m': 0.7708,  # Example threshold value
            'visual_threshold_m1sd': 0.8427,  # Example threshold value
            'visual_threshold_m2sd': 0.9145,  # Example threshold value
            'visual_threshold_m-1sd': 0.699,  # Example threshold value
            # Add more properties as needed
        },
        'castle': {
            'path': "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
            'visual_threshold_m': 0.8509,  # Example threshold value
            'visual_threshold_m1sd': 0.9030,  # Example threshold value
            'visual_threshold_m2sd': 0.955,  # Example threshold value
            'visual_threshold_m-1sd': 0.7989,  # Example threshold value
            'visual_threshold_min_g': 0.9162380695343018,
            # Add more properties as needed
        },
        'rep': {
            'path': "/home/rodriguez/Documents/GitHub/habitat/data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb",  # Use the scene ID as 'apt_1'
            'dataset_config': "data/replica_cad/replicaCAD.scene_dataset_config.json",
            # You can add other properties as needed
        }
    }

    def __init__(self, name):
        """
        Initializes the Scene object with the given name.

        Args:
            name (str): The name of the scene. Should be one of 
                        'van_gogh_room', 'apartment', or 'skokloster_castle'.

        Raises:
            ValueError: If the scene name is not recognized.
        """
        normalized_name = name.lower().replace(" ", "_")
        if normalized_name not in self.SCENE_CONFIG:
            raise ValueError(f"Scene '{name}' is not recognized. "
                             f"Available scenes: {list(self.SCENE_CONFIG.keys())}")
        self.name = normalized_name
        self.config = self.SCENE_CONFIG[self.name]

    def get_path(self):
        """
        Returns the file path of the scene.

        Returns:
            str: The file path to the scene's GLB file.
        """
        return self.config['path']

    def get_visual_threshold(self, th_type='m'):
        """
        Returns the visual threshold of the scene.

        Returns:
            float: The predefined visual threshold for the scene.
        """
        if th_type == 'm':
            return self.config['visual_threshold_m']
        elif th_type == 'm1sd':
            return self.config['visual_threshold_m1sd']
        elif th_type == 'm2sd':
            return self.config['visual_threshold_m2sd']
        elif th_type == 'm-1sd':
            return self.config['visual_threshold_m-1sd']
        elif th_type == 'min_g':
            return self.config['visual_threshold_min_g']
        else:
            raise ValueError(f"Invalid threshold type '{th_type}'. "
                             f"Valid types: 'm', 'm1sd', 'm2sd', 'm-1sd', 'min_g'")

    def get_property(self, property_name):
        """
        Retrieves a specified property of the scene.

        Args:
            property_name (str): The name of the property to retrieve.

        Returns:
            Any: The value of the requested property.

        Raises:
            AttributeError: If the property does not exist.
        """
        if property_name in self.config:
            return self.config[property_name]
        else:
            raise AttributeError(f"Property '{property_name}' does not exist for scene '{self.name}'.")

    @classmethod
    def get_available_scenes(cls):
        """
        Returns a list of available scene names.

        Returns:
            list: A list of available scene names as strings.
        """
        return list(cls.SCENE_CONFIG.keys())

    def __repr__(self):
        return f"Scene(name='{self.name}')"
