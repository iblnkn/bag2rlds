import numpy as np
import tqdm
import os

N_TRAIN_EPISODES = 100
N_VAL_EPISODES = 100

EPISODE_LENGTH = 10
#!/usr/bin/env python3
# create_example_data.py
# A script that creates synthetic ROS 2 bagfiles for demonstration,
# replacing the original create_example_data.py which made .npy files.

import os
import sys
import yaml
import numpy as np
import tqdm

import rosbag2_py
from rclpy.clock import Clock
from rclpy.duration import Duration
from rclpy.serialization import serialize_message

# Example ROS messages:
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

def main():
    """
    Create synthetic bagfiles that mimic 'episodes', storing random 'camera' images
    and random 'action' arrays. 
    The config YAML can specify:
      - n_train_episodes
      - n_val_episodes
      - episode_length
      - storage_id (e.g. 'sqlite3' or 'mcap')
      - output_dir (e.g. 'data')
    """

    if len(sys.argv) < 2:
        print("Usage: python create_synthetic_bag_data.py <path_to_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Pull out parameters from the YAML
    synth_conf = config.get('bag2rlds_synthetic', {})
    n_train = synth_conf.get('n_train_episodes', 100)
    n_val = synth_conf.get('n_val_episodes', 100)
    episode_length = synth_conf.get('episode_length', 10)
    storage_id = synth_conf.get('storage_id', 'sqlite3')
    output_dir = synth_conf.get('output_dir', 'data')
    random_seed = synth_conf.get('random_seed', None)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Prepare output directories for train/val
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print("Generating train bagfiles...")
    for i in tqdm.tqdm(range(n_train)):
        bag_name = f"episode_{i}.db3"  # or .mcap if using storage_id='mcap'
        bag_path = os.path.join(train_dir, bag_name)
        create_random_bagfile(bag_path, episode_length, storage_id)

    print("Generating val bagfiles...")
    for i in tqdm.tqdm(range(n_val)):
        bag_name = f"episode_{i}.db3"
        bag_path = os.path.join(val_dir, bag_name)
        create_random_bagfile(bag_path, episode_length, storage_id)

    print("Successfully created synthetic bag data!")


def create_random_bagfile(bag_path: str, episode_length: int, storage_id: str):
    """
    Creates a single bagfile with 'episode_length' steps of random data.
    We store:
      1) A 'camera' topic with sensor_msgs/Image (64x64, 'rgb8')
      2) An 'action' topic with std_msgs/Float32MultiArray (length=10)
    Each step is spaced by 0.2 seconds (5 Hz).
    """
    # Delete old bag if it exists
    if os.path.exists(bag_path) or os.path.exists(bag_path + ".metadata.yaml"):
        # We must remove the old bag or rosbag2 won't overwrite.
        remove_old_bag(bag_path)

    writer = rosbag2_py.SequentialWriter()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id=storage_id
    )
    converter_options = rosbag2_py.ConverterOptions('', '')
    writer.open(storage_options, converter_options)

    # Register two topics with the writer
    camera_topic_info = rosbag2_py.TopicMetadata(
        name='/camera',
        type='sensor_msgs/msg/Image',
        serialization_format='cdr'
    )
    writer.create_topic(camera_topic_info)

    action_topic_info = rosbag2_py.TopicMetadata(
        name='/action',
        type='std_msgs/msg/Float32MultiArray',
        serialization_format='cdr'
    )
    writer.create_topic(action_topic_info)

    # We'll simulate a timeline
    time_stamp = Clock().now()

    for step in range(episode_length):
        # 1) Create a random image
        img_msg = Image()
        img_msg.height = 64
        img_msg.width = 64
        img_msg.encoding = 'rgb8'
        img_msg.step = img_msg.width * 3  # stride in bytes
        random_image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        img_msg.data = random_image.tobytes()

        # 2) Create a random action
        action_msg = Float32MultiArray()
        # shape: e.g. 10 floats
        random_action = np.random.rand(10).astype(np.float32)
        action_msg.data = list(random_action)  # Must store as list in Float32MultiArray

        # Write them to bag
        writer.write(
            '/camera',
            serialize_message(img_msg),
            time_stamp.nanoseconds
        )
        writer.write(
            '/action',
            serialize_message(action_msg),
            time_stamp.nanoseconds
        )

        # increment time by 0.2s
        time_stamp = time_stamp + Duration(seconds=0.2)

    writer.reset()


def remove_old_bag(bag_path: str):
    """
    Removes any old DB3 or MCAP files + metadata associated with `bag_path`.
    Because rosbag2 won't overwrite an existing folder or file by default.
    """
    base = os.path.splitext(bag_path)[0]  # remove .db3 or .mcap
    exts = ['.db3', '.mcap', '.metadata.yaml', '']
    for ext in exts:
        candidate = base + ext
        if os.path.exists(candidate):
            if os.path.isfile(candidate):
                os.remove(candidate)
            else:
                # In case the storage plugin made a directory
                import shutil
                shutil.rmtree(candidate, ignore_errors=True)


if __name__ == '__main__':
    main()


def create_fake_episode(path):
    episode = []
    for step in range(EPISODE_LENGTH):
        episode.append({
            'image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'state': np.asarray(np.random.rand(10), dtype=np.float32),
            'action': np.asarray(np.random.rand(10), dtype=np.float32),
            'language_instruction': 'dummy instruction',
        })
    np.save(path, episode)


# create fake episodes for train and validation
print("Generating train examples...")
os.makedirs('data/train', exist_ok=True)
for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    create_fake_episode(f'data/train/episode_{i}.npy')

print("Generating val examples...")
os.makedirs('data/val', exist_ok=True)
for i in tqdm.tqdm(range(N_VAL_EPISODES)):
    create_fake_episode(f'data/val/episode_{i}.npy')

print('Successfully created example data!')
