# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

from .bbox import BBox
from typing import List


class Frame:
    def __init__(
        self,
        frame_id,
        cur_sample_token=None,
        timestamp=None,
        transform_matrix=None,
        scene_id=None,
        audit_gt_tokens=None,
    ):

        self.frame_id = frame_id
        self.scene_id = scene_id
        self.cur_sample_token = cur_sample_token
        self.timestamp = timestamp
        self.transform_matrix = transform_matrix

        self.bboxes: List[BBox] = []
        if audit_gt_tokens is not None:
            self.audit_gt_tokens = list(audit_gt_tokens)
