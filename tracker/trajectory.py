# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track
# Trajectory: Pure Data Container
#
# [REFACTOR] All Kalman filter instances (EKF_CV, EKF_CA, EKF_CTRA,
# KF_YAW, KF_SIZE) have been REMOVED. Trajectory is now a stateless
# data container that stores track_id, historical bboxes, status flags,
# and lifecycle metadata. All state estimation is handled externally
# by DecoupledAdaptiveKF + TemporalMamba in the tracker layer.
# ------------------------------------------------------------------------

import numpy as np
import copy
import warnings

from .bbox import BBox
from typing import List
from scipy.optimize import curve_fit, OptimizeWarning

np.set_printoptions(formatter={"float": "{:0.4f}".format})


def linear_func(x: float, a: float, b: float) -> float:
    return a * x + b


class Trajectory:
    """
    Pure data container for a single tracked object.

    Responsibilities:
      - Store track_id, category, historical bboxes, matched scores
      - Manage lifecycle flags (status_flag, unmatch_length, etc.)
      - Compute diff/curve velocity from bbox history (geometry only)
      - Size estimation via EMA (early) and rigid-body locking (mature)

    NOT responsible for:
      - Kalman filtering (predict / update) — handled by MambaDecoupledEKF
      - State fusion — handled by the outer tracker
    """

    def __init__(
        self,
        track_id: int,
        init_bbox: BBox = None,
        first_bbox: bool = True,
        cfg: dict = None,
    ) -> None:
        self.track_id = track_id
        self.category_num: int = cfg["CATEGORY_MAP_TO_NUMBER"][init_bbox.category]

        self.first_bbox = first_bbox
        self.track_length: int = 1
        self.unmatch_length: int = 0
        self.out_range_length: int = 0
        self.is_output: bool = False
        self.first_updated_frame: int = init_bbox.frame_id
        self.last_updated_frame: int = init_bbox.frame_id
        self.cfg = cfg
        self.bboxes: List[BBox] = [init_bbox]
        self.matched_scores: List[float] = []

        # ---- Size locking (rigid-body prior) ----
        # Physical objects don't change size. We fuse early observations via
        # EMA (track_length <= 10) and lock the estimate permanently afterwards.
        self.smoothed_lwh: List[float] = list(init_bbox.lwh)

        # ---- config thresholds ----
        self.frame_rate: float = cfg["FRAME_RATE"]
        self.cost_mode: str = cfg["MATCHING"]["BEV"]["COST_MODE"][self.category_num]
        self._cache_bbox_len: int = cfg["THRESHOLD"]["TRAJECTORY_THRE"]["CACHE_BBOX_LENGTH"][
            self.category_num
        ]
        self._max_predict_len: int = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "PREDICT_BBOX_LENGTH"
        ][self.category_num]
        self._max_unmatch_len: int = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "MAX_UNMATCH_LENGTH"
        ][self.category_num]
        self._confirmed_track_length: int = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "CONFIRMED_TRACK_LENGTH"
        ][self.category_num]
        self._delet_out_track_length: int = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "DELET_OUT_VIEW_LENGTH"
        ][self.category_num]
        self._confirmed_det_score: float = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "CONFIRMED_DET_SCORE"
        ][self.category_num]
        self._confirmed_match_score: float = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "CONFIRMED_MATCHED_SCORE"
        ][self.category_num]
        self._is_filter_predict_box: float = cfg["THRESHOLD"]["TRAJECTORY_THRE"][
            "IS_FILTER_PREDICT_BOX"
        ][self.category_num]

        # 0: initialization / 1: confirmed / 2: obscured / 4: dead
        self.status_flag: int = 1

    # ------------------------------------------------------------------
    # Lifecycle methods (called by base_tracker.py)
    # These manage bbox history and status flags ONLY.
    # The actual state prediction/update is done externally by
    # DecoupledAdaptiveKF. The outer tracker writes fused results
    # back into bbox fields (global_xyz_lwh_yaw_predict, *_fusion, etc.)
    # BEFORE or AFTER calling these methods.
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """
        Called by tracker before association.

        Previously ran KF predict internally. Now a no-op placeholder —
        the outer tracker calls DecoupledAdaptiveKF.predict() and writes
        predicted state into self.bboxes[-1] fields directly.
        """
        pass

    def update(self, bbox: BBox, matched_score: float) -> BBox:
        """
        Called when this trajectory is matched to a detection.

        Manages bbox history, lifecycle, and derived velocities.
        The outer tracker is responsible for running DecoupledAdaptiveKF.update()
        and writing fused state back into bbox fields.

        Args:
            bbox           : Matched detection bbox.
            matched_score  : Association cost for this match.

        Returns:
            The appended bbox (with updated metadata).
        """
        bbox.track_id = self.track_id
        self.track_length += 1
        bbox.track_length = self.track_length
        self.last_updated_frame = bbox.frame_id
        self.unmatch_length = 0
        self.bboxes.append(bbox)

        if len(self.bboxes) > self._cache_bbox_len:
            self.bboxes.pop(0)

        self.matched_scores.append(matched_score)

        # ---- Size EMA & Locking (rigid-body prior) ----
        # Early life (<= 10 frames): EMA fuses noisy measurements.
        # Mature (> 10 frames): lock — ignore detection size entirely.
        current_lwh = list(bbox.lwh)
        if self.track_length <= 10:
            alpha = 0.2  # weight on new measurement
            self.smoothed_lwh = [
                alpha * current_lwh[i] + (1.0 - alpha) * self.smoothed_lwh[i]
                for i in range(3)
            ]
        # Overwrite bbox.lwh with the smoothed/locked estimate.
        # This propagates into all downstream consumers (fusion, output, eval).
        bbox.lwh_ema = list(self.smoothed_lwh)

        # derived velocity estimates from raw bbox history (no KF needed)
        self.bboxes[-1].global_velocity_diff = self.cal_diff_velocity()
        self.bboxes[-1].global_velocity_curve = self.cal_curve_velocity()
        self.bboxes[-1].matched_score = matched_score

        # status promotion
        if self.track_length > self._confirmed_track_length or (
            matched_score > self._confirmed_match_score
            and self.bboxes[-1].det_score > self._confirmed_det_score
        ):
            self.status_flag = 1

        return self.bboxes[-1]

    def unmatch_update(self, frame_id: int) -> None:
        """
        Called when this trajectory has no matching detection.

        Creates a fake (predicted) bbox to maintain the timeline.
        The outer tracker writes the KF-predicted state into the fake bbox
        BEFORE calling this method.

        Args:
            frame_id : Current frame id.
        """
        self.unmatch_length += 1

        fake_bbox = copy.deepcopy(self.bboxes[-1])
        # Exponential score decay for coasted predictions.
        # Pure KF predictions should not inherit the last observation's full score —
        # the longer a track goes without a real detection, the lower its confidence.
        last_real_score = 0.0
        for b in reversed(self.bboxes):
            if not getattr(b, "is_fake", False):
                last_real_score = b.det_score
                break
        decay = 0.8 ** self.unmatch_length
        fake_bbox.det_score = last_real_score * decay
        fake_bbox.is_fake = True
        fake_bbox.frame_id = frame_id

        # Note: the outer tracker should have already written predicted
        # xyz/lwh/yaw into fake_bbox fields via DecoupledAdaptiveKF output.
        # If not yet integrated, the deep-copied values serve as fallback.

        self.bboxes.append(fake_bbox)
        self.matched_scores.append(0)
        self.bboxes[-1].matched_score = 0
        self.bboxes[-1].unmatch_length = self.unmatch_length

        self.bboxes[-1].global_velocity_diff = self.cal_diff_velocity()
        self.bboxes[-1].global_velocity_curve = self.cal_curve_velocity()

        if len(self.bboxes) > self._cache_bbox_len:
            self.bboxes.pop(0)

        # lifecycle state machine
        if self.status_flag == 0 and self.track_length > self._confirmed_track_length:
            self.status_flag = 4

        if self.status_flag == 1 and self.unmatch_length > self._max_unmatch_len:
            self.status_flag = 2

        if self.status_flag == 2 and self.unmatch_length > self._max_predict_len:
            self.status_flag = 4

    # ------------------------------------------------------------------
    # Post-processing (offline interpolation)
    # ------------------------------------------------------------------

    def logit(self, y: float) -> float:
        if y == 0:
            return -10000
        if y <= 0 or y >= 1:
            raise ValueError("Input must be in the range (0, 1).")
        return np.log(y / (1 - y))

    def filtering(self) -> None:
        """
        Offline interpolation of missing detections + quality-aware score assignment.

        Interpolation: linearly fills gaps in global_xyz_lwh_yaw_fusion for fake bboxes.

        Score assignment: only high-confidence real detections (original score >= 0.4)
        determine the final trajectory score. Stage 2 ByteTrack rescue dets (0.1-0.4)
        and fake coasted bboxes are excluded — they exist to maintain track continuity
        through occlusion, not to represent object existence probability.
        """
        # snapshot original scores before logit transform
        original_scores = [bbox.det_score for bbox in self.bboxes]

        if_has_unmatched = 0
        unmatch_bbox_sum = 0
        start_xyz_lwh_yaw = None
        start_frame = 0

        last_xyz_lwh_yaw_fusion = None
        for bbox in self.bboxes:
            frame_id = bbox.frame_id
            bbox.det_score = self.logit(bbox.det_score)
            if (self.first_updated_frame <= frame_id <= self.last_updated_frame
                    and bbox.is_fake and self.is_output):
                bbox.is_interpolation = True
                if if_has_unmatched == 0:
                    start_xyz_lwh_yaw = last_xyz_lwh_yaw_fusion
                    if_has_unmatched = 1
                    unmatch_bbox_sum = 1
                    start_frame = frame_id
                else:
                    unmatch_bbox_sum += 1
            elif not bbox.is_fake and if_has_unmatched == 1:
                end_frame = frame_id - 1
                end_xyz_lwh_yaw = bbox.global_xyz_lwh_yaw_fusion
                gap = (end_xyz_lwh_yaw - start_xyz_lwh_yaw) / (unmatch_bbox_sum + 1)
                last_xyz_lwh_yaw = start_xyz_lwh_yaw
                if unmatch_bbox_sum >= 2:
                    for bbox_tmp in self.bboxes:
                        if start_frame <= bbox_tmp.frame_id <= end_frame:
                            last_xyz_lwh_yaw += gap
                            bbox_tmp.global_xyz_lwh_yaw_fusion[0] = last_xyz_lwh_yaw[0]
                            bbox_tmp.global_xyz_lwh_yaw_fusion[1] = last_xyz_lwh_yaw[1]
                            bbox_tmp.global_xyz_lwh_yaw_fusion[2] = last_xyz_lwh_yaw[2]
                if_has_unmatched = 0
            last_xyz_lwh_yaw_fusion = bbox.global_xyz_lwh_yaw_fusion

        # ---- quality-aware score: only high-confidence real dets ----
        quality_logit_scores = []
        for bbox, orig_score in zip(self.bboxes, original_scores):
            if not bbox.is_fake and orig_score >= 0.4:
                quality_logit_scores.append(bbox.det_score)

        if quality_logit_scores:
            final_score = sum(quality_logit_scores) / len(quality_logit_scores)
        else:
            best_orig = max(original_scores) if original_scores else 0.5
            final_score = self.logit(best_orig)

        for bbox in self.bboxes:
            bbox.det_score = final_score

    # ------------------------------------------------------------------
    # Derived velocity (pure geometry, no KF)
    # ------------------------------------------------------------------

    def cal_diff_velocity(self) -> list:
        """Finite-difference velocity from last two bboxes."""
        if len(self.bboxes) > 1:
            prev_bbox = self.bboxes[-2]
            cur_bbox = self.bboxes[-1]
            time_diff = (cur_bbox.frame_id - prev_bbox.frame_id) / self.frame_rate
            if time_diff > 0:
                position_diff = np.array(cur_bbox.global_xyz[:2]) - np.array(prev_bbox.global_xyz[:2])
                return (position_diff / time_diff).tolist()
            return [0.0, 0.0]
        return [0.0, 0.0]

    def cal_curve_velocity(self) -> list:
        """Linear-fit velocity from last three bboxes."""
        if len(self.bboxes) > 2:
            x_vals = [bb.frame_id for bb in self.bboxes[-3:]]
            y_vals_x = [bb.global_xyz[0] for bb in self.bboxes[-3:]]
            y_vals_y = [bb.global_xyz[1] for bb in self.bboxes[-3:]]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    popt_x, _ = curve_fit(linear_func, x_vals, y_vals_x)
                    popt_y, _ = curve_fit(linear_func, x_vals, y_vals_y)
                return [popt_x[0], popt_y[0]]
            except RuntimeError:
                return [0.0, 0.0]
        return [0.0, 0.0]
