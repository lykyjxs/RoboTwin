import numpy as np
import torch
import dill
import os, sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name, model_name, checkpoint_id, pi0_step = (usr_args["train_config_name"], usr_args["model_name"],
                                                              usr_args["checkpoint_id"], usr_args["pi0_step"])
    return PI0(
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        adaptive_pi0_step=usr_args.get("adaptive_pi0_step", False),
        outside_pi0_step=usr_args.get("outside_pi0_step", 50),
        inside_pi0_step=usr_args.get("inside_pi0_step", 25),
    )


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    if model.adaptive_pi0_step:
        result = model.get_action_result()
        chunk_len = result["adaptive_pi0_step"]
        h_entry_bin = result.get("keystate_h_entry_bin_pred", None)
        type_pred = result.get("keystate_type_pred", None)
        model.adaptive_decision_count += 1
        if model.adaptive_decision_count <= 10 or h_entry_bin == 0:
            print(f"[adaptive_chunk] h_bin={h_entry_bin} type={type_pred} chunk={chunk_len}")
        actions = result["actions"][:chunk_len]
    else:
        actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
