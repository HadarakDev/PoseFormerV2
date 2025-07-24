# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import hashlib

import json
import numpy as np

import json
import numpy as np
import json
import numpy as np

def save_h36m_to_json(keypoints, scores, output_path):
    """
    Save Human3.6M-style keypoints to JSON format.

    Args:
        keypoints: np.ndarray of shape (M, T, 17, 2)
        scores: np.ndarray of shape (M, T, 17)
        output_path: Path to output JSON file
    """
    M, T, J, D = keypoints.shape
    assert J == 17 and D == 2, "Expected 17 joints with 2D coordinates"

    people_data = []

    for person_id in range(M):
        frames = []

        for t in range(T):
            joints = keypoints[person_id, t].tolist()  # (17, 2)
            joint_scores = scores[person_id, t].tolist() if scores is not None else [1.0] * 17

            frames.append({
                "frame": t,
                "joints": joints,
                "scores": joint_scores
            })

        people_data.append({
            "id": person_id,
            "keypoints": frames
        })

    output = {"people": people_data}

    with open(f"{output_path}pose2d_h36m.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ H36M-style keypoints saved to {output_path}pose2d_h36m.json")


def save_keypoints_coco_format(keypoints, scores, output_path, image_prefix="frame", width=1920, height=1080):
    """
    Save 2D keypoints and scores in COCO format for a single person per frame.
    
    Args:
        keypoints: np.array of shape (T, 17, 2) or (T, 17, 3)
        scores: np.array of shape (T, 17) or (T, 17, 1)
        output_path: Path to save JSON file
        image_prefix: Prefix to use for fake image filenames (e.g., "frame" -> "frame_0.jpg")
        width: Width of each dummy image
        height: Height of each dummy image
    """
    if keypoints.shape[0] == 1:
        keypoints = keypoints[0]
        scores = scores[0]
    T = keypoints.shape[0]
    annotations = []
    images = []

    for frame_id in range(T):
        kpts_2d = keypoints[frame_id]    # (17, 2) or (17, 3)
        kpts_score = scores[frame_id]    # (17,) or (17, 1)

        if kpts_2d.shape[1] < 2:
            raise ValueError(f"Keypoints at frame {frame_id} must have at least 2D (x, y) values")

        kpts_coco = []

        for j in range(17):
            x, y = map(float, kpts_2d[j, :2])  # Always take just (x, y)

            # Ensure scalar score
            s = float(kpts_score[j]) if np.ndim(kpts_score[j]) == 0 else float(kpts_score[j][0])

            # Map score to visibility
            if s < 0.3:
                v = 0
            elif s < 0.6:
                v = 1
            else:
                v = 2

            kpts_coco.extend([x, y, v])

        annotations.append({
            "id": frame_id,
            "image_id": frame_id,
            "category_id": 1,
            "keypoints": kpts_coco,
            "score": float(np.mean(kpts_score))
        })

        images.append({
            "id": frame_id,
            "file_name": f"{image_prefix}_{frame_id}.jpg",
            "width": width,
            "height": height
        })

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            "skeleton": [
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
                [1, 3], [2, 4], [3, 5], [4, 6]
            ]
        }]
    }

    with open(f"{output_path}pose2d_coco.json", "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"✅ COCO-format keypoints saved to: {output_path}/pose2d_coco.json")


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    # model.state_dict(model_dict).requires_grad = False
    return model
