#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import os
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step, adaptive_pi0_step=False, outside_pi0_step=50, inside_pi0_step=25):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)

        specified_path = f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=assets_id)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = int(pi0_step)
        self.adaptive_pi0_step = self._as_bool(adaptive_pi0_step)
        self.outside_pi0_step = int(outside_pi0_step)
        self.inside_pi0_step = int(inside_pi0_step)
        self.adaptive_decision_count = 0

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    def _adaptive_step_from_h_entry_bin(self, h_entry_bin):
        # bin0 means h == 0: current frame is inside/at the predicted key-state window.
        if int(h_entry_bin) == 0:
            return self.inside_pi0_step
        return self.outside_pi0_step

    def get_action_result(self):
        assert self.observation_window is not None, "update observation_window first!"
        result = self.policy.infer(self.observation_window)
        actions = result["actions"]
        h_entry_bin = result.get("keystate_h_entry_bin_pred", None)
        if h_entry_bin is None:
            chunk_len = self.pi0_step
        else:
            chunk_len = self._adaptive_step_from_h_entry_bin(h_entry_bin)
        result["adaptive_pi0_step"] = int(min(chunk_len, len(actions)))
        return result

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
