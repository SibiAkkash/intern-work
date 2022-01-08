from typing import List
from helpers import get_time_elapsed_ms
from pprint import pprint
import json
import numpy as np
from pathlib import Path
import string
import random
import cv2
import mysql.connector


def is_object_present(detections: List[int], object_id: int):
    return object_id in detections


# ----------------- FLOW -------------------------------------------------------------------------------------
# start of cycle is presence of start marker
# start of step is presence of corresponding bounding box
# when we see a new bounding box, the previous step is over, calculate time taken
# if this new bounding box is the end marker, the cycle has finished, refresh state
# Step order doesnt matter, cycle start and end is based on presence of start and end marker respectively.
# ------------------------------------------------------------------------------------------------------------


def write_to_db(
    cnx: mysql.connector.connection_cext.CMySQLConnection,
    sequence: List[int],
    step_times: List[float],
    cycle_video_path: str,
):
    cursor = cnx.cursor()
    add_cycle = "INSERT INTO Cycle_seqs (station_id, sequence, video_path) VALUES (%s, %s, %s)"

    # create new cycle
    data_cycle = (2, json.dumps(sequence), cycle_video_path)
    cursor.execute(add_cycle, data_cycle)

    cycle_id = cursor.lastrowid

    # add step times
    add_times = "INSERT INTO Cycles VALUES (%s, %s, %s)"
    for step_num in range(len(step_times)):
        data_times = (cycle_id, step_num, step_times[step_num])
        cursor.execute(add_times, data_times)

    cnx.commit()
    cursor.close()


class VisualInspector:
    def __init__(
        self,
        start_marker_object_id: int,
        end_marker_object_id: int,
        process_object_ids: List[int],
        stream_fps: float,
        db_connection: mysql.connector.connection_cext.CMySQLConnection,
        video_save_dir: str = "cycles",
    ):
        self.start_marker_object_id = start_marker_object_id
        self.end_marker_object_id = end_marker_object_id
        self.process_object_ids = process_object_ids
        self.NUM_STEPS = len(process_object_ids)

        self.stream_fps = stream_fps
        self.save_dir = Path(video_save_dir)
        self.vid_writer = None

        self.cnx = db_connection

        self.refresh_state()

    def refresh_state(self) -> None:
        # generate random name for cycle video
        cycle_vid_name = "".join(random.choice(string.ascii_letters) for _ in range(10))
        extension = "mp4"
        save_path = f"{str(self.save_dir / cycle_vid_name)}.{extension}"

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
            "cycle_vid_save_path": save_path,
        }

    def _create_vid_writer(self):
        # create vid writer is called only from by handle_cycle_start()
        # we are guarenteed to have a new save_path and the first frame
        # MJPG, mp4v
        codec = "mp4v"
        fourcc = cv2.VideoWriter_fourcc(*codec)
        w, h = self.current_frame.shape[1], self.current_frame.shape[0]
        self.vid_writer = cv2.VideoWriter(
            self.state["cycle_vid_save_path"], fourcc, self.stream_fps, (w, h), True
        )

    def save_frame(self):
        print("saving frame")
        if self.vid_writer:
            self.vid_writer.write(self.current_frame)

    def cycle_started(self) -> bool:
        return self.state["cycle_started"]

    def _handle_cycle_start(self, frame_num: int):
        self.state["cycle_started"] = True
        self.state["cycle_start_frame_num"] = frame_num
        self.state["seen_objects"].append(self.start_marker_object_id)
        print("CYCLE STARTED")
        pprint(self.state)
        # reset vid writer at start of each cycle
        self._create_vid_writer()
        self.save_frame()

    def _handle_cycle_end(self, frame_num: float):
        # if there was any step, do necessary processing
        if self.state["prev_step"] != -1:
            self._handle_state_change(frame_num=frame_num)

        # save last frame of cycle
        self.save_frame()
        # release vid writer pointer
        self.vid_writer.release()

        # TODO can this be async ?
        write_to_db(
            cnx=self.cnx,
            sequence=self.state["step_sequence"],
            step_times=self.state["step_times"],
            cycle_video_path=self.state["cycle_vid_save_path"],
        )

        self.refresh_state()
        print("CYCLE ENDED")
        pprint(self.state)

    def _handle_state_change(
        self,
        frame_num: int,
        next_step_num: int = -1,
        next_object_id: int = -1,
        is_last_step: bool = False,
    ):
        # check if this is the first object other than the start marker
        if len(self.state["seen_objects"]) == 1:
            print("FIRST OBJECT SEEN")
            # this means this is the first process object to be seen
            self.state["steps_started"] = True
            self.state["step_start_frame_num"] = frame_num
            self.state["prev_step"] = next_step_num
            self.state["seen_objects"].append(next_object_id)

        else:
            print("not first object")
            # calc step time for previous step
            prev_step = self.state["prev_step"]

            # previous step ends here
            self.state["step_end_frame_num"] = frame_num

            # calculate cycle time for the previous step
            time_taken = get_time_elapsed_ms(
                start_frame=self.state["step_start_frame_num"],
                end_frame=self.state["step_end_frame_num"],
                fps=self.stream_fps,
            )
            print(f"Time taken to complete step {prev_step}: {time_taken} ms")
            self.state["step_times"][prev_step] = round(time_taken, 2)

            self.state["is_step_completed"][prev_step] = True
            self.state["step_sequence"].append(prev_step)
            self.state["num_steps_completed"] += 1

            # if not is_last_step:
            # set start frame number for next step
            self.state["step_start_frame_num"] = frame_num + 1

            # set prev_step to the current step
            self.state["prev_step"] = next_step_num

            self.state["seen_objects"].append(next_object_id)

        pprint(self.state)

    def _get_step_number(self, detections: List[int]):
        """if list contains objects we havent seen, return the step number and object id"""
        # TODO this assumes we will only see 1 new object from the previous frame
        for object_id in detections:
            # return (step number, object_id) if we see a new object that we haven't seen yet
            if (
                object_id not in self.state["seen_objects"]
                and object_id in self.process_object_ids
            ):
                return self.process_object_ids.index(object_id), object_id

        return -1, -1

    def process_detections(
        self, detections: List[int], frame_num: int, current_frame: np.ndarray
    ):

        self.current_frame = current_frame

        if not self.cycle_started():
            start_marker_found = is_object_present(
                detections=detections, object_id=self.start_marker_object_id
            )
            if start_marker_found:
                self._handle_cycle_start(frame_num)
            # if cycle hasn't started yet, we want to return
            # if the cycle did start, we still want to return and process only from next frame
            return

        end_marker_found = is_object_present(
            detections=detections, object_id=self.end_marker_object_id
        )

        if end_marker_found:
            self._handle_cycle_end(frame_num)
            return

        # at this step, the cycle has started, it hasn't ended
        # save all these frames
        self.save_frame()

        # cycle started, we see a object(the start marker could still be in frame) that is not the end marker
        # check if there is an object other than previously seen objects (including start marker)
        next_step_num, next_object_id = self._get_step_number(detections)

        # we haven't seen a new object
        if next_step_num == -1:
            return

        self._handle_state_change(
            next_step_num=next_step_num,
            next_object_id=next_object_id,
            frame_num=frame_num,
            is_last_step=False,
        )


# def show_steps(image):
#     # box background
#     cv2.rectangle(
#         img=image,
#         color=(100, 100, 100),
#         pt1=RECT_POINT_1,
#         pt2=RECT_POINT_2,
#         thickness=-1,
#     )
#     # steps status
#     for step in range(NUM_STEPS + 1):
#         if step == NUM_STEPS:
#             s = f"Sequence: {state['sequence']}"
#             color = DONE_COLOUR if state["cycle_ended"] else NOT_DONE_COLOUR
#         else:
#             s = f"Step {step + 1}, time taken: {state['step_times'][step]} ms"
#             color = DONE_COLOUR if state["is_step_completed"][step] else NOT_DONE_COLOUR

#         cv2.putText(
#             img=image,
#             text=s,
#             org=(X_OFFSET, Y_OFFSET + Y_PADDING * step),
#             fontFace=0,
#             fontScale=0.5,
#             color=color,
#             thickness=1,
#         )
