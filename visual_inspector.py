from itertools import cycle
from typing import List
from helpers import get_time_elapsed_ms
from pprint import pprint
import json

def is_object_present(detections: List[int], object_id: int):
    return object_id in detections

class VisualInspector:
    def __init__(
        self,
        start_marker_object_id: int,
        end_marker_object_id: int,
        process_object_ids: List[int],
        stream_fps: float,
    ):
        self.start_marker_object_id = start_marker_object_id
        self.end_marker_object_id = end_marker_object_id
        self.process_object_ids = process_object_ids
        self.NUM_STEPS = len(process_object_ids)

        self.stream_fps = stream_fps

        self.refresh_state()
        

    def refresh_state(self) -> None:
        self.state = {
            "prev_step": -1,
            "seen_objects": [],
            "num_steps_completed": 0,
            "cycle_started": False,
            "cycle_ended": False,
            "steps_started": False,
            "steps_completed": False,
            "cycle_start_frame_num": 0,
            "cycle_end_frame_num": 0,
            "step_start_frame_num": 0,
            "step_end_frame_num": 0,
            "is_step_completed": [False] * self.NUM_STEPS,
            "step_times": [None] * self.NUM_STEPS,
            "step_sequence": [],
        }

    def cycle_started(self) -> bool:
        return self.state["cycle_started"]

    def handle_cycle_start(self, frame_num: int):
        self.state["cycle_started"] = True
        self.state["cycle_start_frame_num"] = frame_num
        self.state["seen_objects"].append(self.start_marker_object_id)
        print("CYCLE STARTED")

    def check_for_start_marker(self, detections: List[int], frame_num: int) -> None:
        if is_object_present(detections=detections, object_id=self.start_marker_object_id):
            self.handle_cycle_start(frame_num=frame_num)

    def check_cycle_start(self, detections: List[int], frame_num: int):
        if self.cycle_started():
            return True

        

        

    def process_detections(self, detections: List[int], frame_num: int):
        if not self.cycle_started():
            self.check_for_start_marker()
        
        if not self.cycle_started():
            return


    

