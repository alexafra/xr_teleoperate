import os
import cv2
import json
import datetime
import numpy as np
import time
from .rerun_visualizer import RerunLogger
from queue import Queue, Empty
from threading import Thread
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

class EpisodeWriter():
    def __init__(self, task_dir, task_goal=None, task_desc = None, task_steps = None, frequency=30, image_size=[640, 480], rerun_log = True):
        """
        image_size: [width, height]
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps":"step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = frequency
        self.image_size = image_size

        self.rerun_log = rerun_log
        if self.rerun_log:
            logger_mp.info("==> RerunLogger initializing...\n")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit = "300MB")
            logger_mp.info("==> RerunLogger initializing ok.\n")
        
        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir) if 'episode_' in episode_dir and not episode_dir.endswith('.zip')]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info()

        self.is_available = True  # Indicates whether the class is available for new operations
        # Initialize the queue and worker thread
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False  # Flag to indicate when save_episode is triggered
        self.episode_start_monotonic = None
        self.episode_stop_monotonic = None

        self.episode_start_utc = None
        self.episode_stop_utc = None

        self.first_frame_timestamp_s = None
        self.last_frame_timestamp_s = None
        self.previous_frame_timestamp_s = None

        self.frame_timestamps_s = []
        self.max_frame_gap_s = 0.0
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")
    
    def is_ready(self):
        return self.is_available

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
                "version": "1.0.0" if version is None else version, 
                "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
                "author": "unitree" if author is None else author,
                "image": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "depth": {"width":self.image_size[0], "height":self.image_size[1], "fps":self.frequency},
                "audio": {"sample_rate": 16000, "channels": 1, "format":"PCM", "bits":16},    # PCM_S16
                "joint_names":{
                    "left_arm":   [],
                    "left_ee":  [],
                    "right_arm":  [],
                    "right_ee": [],
                    "body":       [],
                },

                "tactile_names": {
                    "left_ee": [],
                    "right_ee": [],
                }, 
                "sim_state": ""
            }

 
    def create_episode(self):
        """
        Create a new episode.
        Returns:
            bool: True if the episode is successfully created, False otherwise.
        Note:
            Once successfully created, this function will only be available again after save_episode complete its save task.
        """
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False  # Return False if the class is unavailable

        # Reset episode-related data and create necessary directories
        self.item_id = -1
        self.episode_id = self.episode_id + 1
        
        #time
        self.episode_start_monotonic = time.perf_counter()
        self.episode_stop_monotonic = None

        self.episode_start_utc = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

        self.episode_stop_utc = None

        self.first_frame_timestamp_s = None
        self.last_frame_timestamp_s = None
        self.previous_frame_timestamp_s = None
        self.frame_timestamps_s = []

        self.max_frame_gap_s = 0.0
        #time emd

        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.raw_depth_dir = os.path.join(self.episode_dir, 'raw_depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.raw_depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
            f.write('"data": [\n')
        self.first_item = True   # Flag to handle commas in JSON array

        if self.rerun_log:
            self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary = 60, memory_limit="300MB")

        self.is_available = False  # After the episode is created, the class is marked as unavailable until the episode is successfully saved
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True  # Return True if the episode is successfully created
        
    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None):
        #add time
        if self.episode_start_monotonic is None:
            raise RuntimeError(
                "Cannot add an item before create_episode()"
            )

        timestamp_s = (
            time.perf_counter()
            - self.episode_start_monotonic
        )

        if self.first_frame_timestamp_s is None:
            self.first_frame_timestamp_s = timestamp_s

        if self.previous_frame_timestamp_s is not None:
            frame_gap_s = (
                timestamp_s
                - self.previous_frame_timestamp_s
            )

            self.max_frame_gap_s = max(
                self.max_frame_gap_s,
                frame_gap_s,
            )

        self.previous_frame_timestamp_s = timestamp_s
        self.last_frame_timestamp_s = timestamp_s
        self.frame_timestamps_s.append(timestamp_s)
        #time end

        # Increment the item ID
        self.item_id += 1
        # Create the item data dictionary
        item_data = {
            'idx': self.item_id,
            'timestamp_s': timestamp_s, #time
            'colors': colors,
            'depths': depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
        }
        # Enqueue the item data
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            # Process items in the queue
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass
        
            # Check if save_episode was triggered
            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})

        # Save images
        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    logger_mp.info(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join('colors', color_name)

        # Save depths
        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.png'
                output_directory = (
                    self.raw_depth_dir
                    if depth_key.startswith("raw_depth")
                    else self.depth_dir
                )
                depth_path = os.path.join(output_directory, depth_name)

                if not cv2.imwrite(depth_path, depth):
                    logger_mp.info(
                        f"Failed to save depth image: {depth_path}"
                    )

                item_data['depths'][depth_key] = os.path.join(
                    os.path.basename(output_directory),
                    depth_name,
                )
        # Save audios
        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        # Update episode data
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(json.dumps(item_data, ensure_ascii=False, indent=4))
            self.first_item = False

        # Log data if necessary
        if self.rerun_log:
            curent_record_time = time.time()
            logger_mp.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{curent_record_time}")
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self):
        """
        Trigger the save operation. This sets the save flag, and the process_queue thread will handle it.
        """
        #time
        if self.need_save:
            return

        self.episode_stop_monotonic = time.perf_counter()

        self.episode_stop_utc = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        #time end
        self.need_save = True  # Set the save flag
        logger_mp.info(f"==> Episode saved start...")

    def _save_episode(self):
        """
        Save the episode data to a JSON file.
        """
        #timing
        timestamps = self.frame_timestamps_s
        frame_count = len(timestamps)

        if (
            self.episode_start_monotonic is not None
            and self.episode_stop_monotonic is not None
        ):
            recording_duration_s = (
                self.episode_stop_monotonic
                - self.episode_start_monotonic
            )
        else:
            recording_duration_s = 0.0

        if frame_count >= 2:
            sample_duration_s = (
                timestamps[-1]
                - timestamps[0]
            )

            measured_fps = (
                (frame_count - 1) / sample_duration_s
                if sample_duration_s > 0
                else 0.0
            )

            frame_gaps = [
                timestamps[i] - timestamps[i - 1]
                for i in range(1, frame_count)
            ]

            max_frame_gap_s = max(frame_gaps)
        else:
            sample_duration_s = 0.0
            measured_fps = 0.0
            max_frame_gap_s = 0.0

        timing = {
            "target_fps": float(self.frequency),
            "capture_start_utc": self.episode_start_utc,
            "capture_stop_utc": self.episode_stop_utc,
            "recording_duration_s": recording_duration_s,
            "sample_duration_s": sample_duration_s,
            "frame_count": frame_count,
            "measured_fps": measured_fps,
            "max_frame_gap_s": max_frame_gap_s,
        }

        with open(
            self.json_path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write("\n],\n")
            f.write(
                '"timing": '
                + json.dumps(
                    timing,
                    ensure_ascii=False,
                    indent=4,
                )
            )
            f.write("\n}")
        #timing end
        self.need_save = False     # Reset the save flag
        self.is_available = True   # Mark the class as available after saving
        logger_mp.info("==> Episode saved successfully to "
            f"{self.json_path}. "
            f"Frames={frame_count}, "
            f"measured_fps={measured_fps:.2f}, "
            f"max_gap={max_frame_gap_s:.3f}s")

    def close(self):
        """
        Stop the worker thread and ensure all tasks are completed.
        """
        self.item_data_queue.join()
        if not self.is_available:  # If self.is_available is False, it means there is still data not saved.
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()