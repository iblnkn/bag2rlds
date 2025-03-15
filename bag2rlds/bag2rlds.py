#!/usr/bin/env python3
# bag2rlds.py
# A TFDS dataset builder that parses ROS 2 bagfiles into RLDS episodes.
# Copyright 2023, MIT License or similar.

import os
import glob
import yaml
from typing import Iterator, Tuple, Any, Dict

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# ROS 2
import rosbag2_py
import rclpy.serialization

# If you want to parse sensor_msgs/Image or geometry_msgs/Twist, import them:
from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist

def _ros_image_to_numpy(img_msg: Image) -> np.ndarray:
    """
    Convert a sensor_msgs/Image to a NumPy array (height x width x 3).
    Assumes 'rgb8' or 'bgr8' encoding. Adjust as needed for your actual image encodings.
    """
    height = img_msg.height
    width = img_msg.width
    channels = 3  # For rgb8/bgr8
    np_arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(height, width, channels)
    # If your data is 'bgr8', you may want np_arr = np_arr[..., ::-1]
    return np_arr


def _parse_action(msg_bytes: bytes) -> np.ndarray:
    """
    Example: parse geometry_msgs/Twist into a 10D array. Adjust as needed.
    Here we just do a placeholder.
    """
    # twist_msg = rclpy.serialization.deserialize_message(msg_bytes, Twist)
    # For instance: return [twist_msg.linear.x, twist_msg.linear.y, ... twist_msg.angular.z...]
    # We'll just return a dummy 10D vector for demonstration.
    return np.zeros((10,), dtype=np.float32)


class Bag2rldsDataset(tfds.core.GeneratorBasedBuilder):
    """
    A dataset builder that converts ROS 2 bagfiles to RLDS episodes,
    reading the configuration from a YAML file.
    """

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, config_path: str = '', **kwargs):
        """
        Additional argument: config_path, a path to a YAML file specifying:
          - train_bags_glob, val_bags_glob
          - observation_topics, action_topic, etc.
        Example usage:
          tfds build --builder bag2rlds.Bag2rldsDataset --builder_kwargs config_path=bag2rlds.yaml
        """
        super().__init__(*args, **kwargs)
        # Load the YAML config
        if not config_path:
            raise ValueError('No config_path provided. Please pass --builder_kwargs="config_path=xxx".')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Config file not found: {config_path}')

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        # For demonstration, we keep the universal sentence encoder
        # If you don't need language embeddings, remove this:
        self._embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...). Same schema as example."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(64, 64, 3),
                            dtype=tf.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(64, 64, 3),
                            dtype=tf.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(10,),
                            dtype=tf.float32,
                            doc='Robot state, [7 joint angles, 2 gripper pos, 1 door angle].'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(10,),
                        dtype=tf.float32,
                        doc='Robot action, e.g. [7 joint velocities, 2 gripper velocities, 1 terminate].'
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=tf.float32,
                        doc='Discount factor.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=tf.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=tf.bool,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=tf.bool,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=tf.bool,
                        doc='True if final step is terminal.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language instruction string.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=tf.float32,
                        doc='Language embedding from a universal-sentence-encoder.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original bagfile.'
                    ),
                }),
            })
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """
        Define data splits. We read them from the YAML config:
          bag2rlds:
            train_bags_glob: "path/to/train/*.db3"
            val_bags_glob:   "path/to/val/*.db3"
        """
        bag_conf = self._config['bag2rlds']
        return {
            'train': self._generate_examples(bag_conf['train_bags_glob']),
            'val': self._generate_examples(bag_conf['val_bags_glob']),
        }

    def _generate_examples(self, bags_glob: str) -> Iterator[Tuple[str, Any]]:
        """
        For each bagfile in the specified glob, parse messages, produce a single RLDS episode.
        If you want multiple episodes per bag, add logic to break them up.
        """
        bag_files = glob.glob(bags_glob)
        bag_files.sort()

        for bag_path in bag_files:
            episode_id, sample = self._parse_bagfile(bag_path)
            if sample is not None:
                yield episode_id, sample

    def _parse_bagfile(self, bag_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Actually open the bag with rosbag2_py, iterate messages, build RLDS steps.
        Return (episode_id, sample_dict).
        """
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # collect all messages
        messages = []
        while reader.has_next():
            (topic, data_bytes, t_ns) = reader.read_next()
            messages.append((topic, data_bytes, t_ns))

        # sort by timestamp
        messages.sort(key=lambda x: x[2])

        # We'll store steps in a list
        steps = []
        is_first_step = True

        # placeholders for current observation
        current_image = np.zeros((64, 64, 3), dtype=np.uint8)
        wrist_image = np.zeros((64, 64, 3), dtype=np.uint8)
        current_state = np.zeros((10,), dtype=np.float32)  # Fake default

        # For simpler config, read from self._config
        obs_topics = self._config['bag2rlds']['observation_topics']
        action_topic = self._config['bag2rlds']['action_topic']
        # If you have text instruction from a topic, parse it. For now, we do dummy.
        language_instruction = 'dummy instruction'
        language_embedding = self._embed([language_instruction])[0].numpy()

        for (topic, data_bytes, t_ns) in messages:
            # Check if this topic matches your observation topics
            if topic == obs_topics.get('camera', ''):
                img_msg = rclpy.serialization.deserialize_message(data_bytes, Image)
                resized = _ros_image_to_numpy(img_msg)
                # If needed, resize to (64,64):
                # (We do not show code here, but you'd do e.g. cv2.resize or PIL.)
                current_image = resized.astype(np.uint8)

            # If you have a second camera for 'wrist_image', do similarly:
            # elif topic == obs_topics.get('wrist_image', ''):
            #     ...

            # If you have a 'state_topic', parse that:
            # elif topic == obs_topics.get('state', ''):
            #     ...
            #     current_state = parse your state

            # Action:
            elif topic == action_topic:
                action_vec = _parse_action(data_bytes)
                # Create a step with the current observation
                step = {
                    'observation': {
                        'image': current_image,
                        'wrist_image': wrist_image,
                        'state': current_state,
                    },
                    'action': action_vec,
                    'discount': 1.0,
                    'reward': 0.0,
                    'is_first': is_first_step,
                    'is_last': False,
                    'is_terminal': False,
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                }
                steps.append(step)
                is_first_step = False

        if len(steps) > 0:
            # Mark final step
            steps[-1]['is_last'] = True
            steps[-1]['is_terminal'] = True
            steps[-1]['reward'] = 1.0  # if you treat the last step as success

            sample = {
                'steps': steps,
                'episode_metadata': {
                    'file_path': bag_path
                }
            }
            return bag_path, sample

        # If no steps found, skip
        return bag_path, None
