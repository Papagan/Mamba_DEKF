# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import re
import os

from tracker.frame import Frame
from tracker.bbox import BBox
from utils.nusc_utils import obtain_box_bottom_corners, filter_bboxes_with_nms


class BaseVersionTrackingDataset:
    def __init__(self, scene_id, scene_data, cfg):
        self.scene_id = scene_id
        self.scene_data = scene_data
        self.cfg = cfg

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, index):
        frame_info = self.scene_data[index]
        frame_id = frame_info["frame_id"]
        timestamp = frame_info["timestamp"]
        cur_sample_token = frame_info["cur_sample_token"]
        transform_matrix = frame_info["transform_matrix"]
        bboxes = frame_info["bboxes"]
        audit_gt_tokens = frame_info.get("audit_gt_tokens")
        if audit_gt_tokens is not None and len(audit_gt_tokens) != len(bboxes):
            raise ValueError("audit_gt_tokens must align with frame bboxes")
        indexed_bboxes = [
            (bbox, audit_gt_tokens[index] if audit_gt_tokens is not None else None)
            for index, bbox in enumerate(bboxes)
            if bbox["category"] in self.cfg["CATEGORY_MAP_TO_NUMBER"]
        ]
        if self.cfg["TRACKING_MODE"] == "ONLINE":
            input_score = self.cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"]

        else:
            input_score = self.cfg["THRESHOLD"]["INPUT_SCORE"]["OFFLINE"]
        filtered = [
            (bbox, token)
            for bbox, token in indexed_bboxes
            if bbox["detection_score"]
            > input_score[self.cfg["CATEGORY_MAP_TO_NUMBER"][bbox["category"]]]
        ]

        if self.cfg["DATASET"] == "nuscenes":
            if filtered:
                token_by_bbox_id = {id(bbox): token for bbox, token in filtered}
                nms_bboxes = filter_bboxes_with_nms(
                    [bbox for bbox, _ in filtered], self.cfg
                )
                filtered = [
                    (bbox, token_by_bbox_id[id(bbox)]) for bbox in nms_bboxes
                ]

        filtered_tokens = (
            [token for _, token in filtered] if audit_gt_tokens is not None else None
        )
        cur_frame = Frame(
            frame_id=int(frame_id),
            scene_id=self.scene_id,
            cur_sample_token=cur_sample_token,
            timestamp=timestamp,
            transform_matrix=transform_matrix,
            audit_gt_tokens=filtered_tokens,
        )
        for bbox, _ in filtered:
            clean_bbox = dict(bbox)
            for token_name in ("gt_token", "instance_token", "annotation_token"):
                clean_bbox.pop(token_name, None)
            cur_frame.bboxes.append(BBox(frame_id, clean_bbox))
        return cur_frame
