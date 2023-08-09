import yaml
from tools.logger import default_logger as logger
from typing import Dict
import os


class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y


class Rect:
    def __init__(self, x, y, width, height) -> None:
        self.x = x
        self.y = y
        self.w = width
        self.h = height


class AlignmentProjectConfig:
    def __init__(self) -> None:
        self.linescan_settings_path: str = None
        self.linescan_folder: str = None
        self.gocator_folder: str = None
        self.keypointsfg: Dict[Point] = {}
        self.keypointsbg: Dict[Point] = {}
        self.calibration_ball: Rect = None
        self.calibration_ball_diameter_mm: float = 40
        self.linescan_flipped: bool = False
        self.gocator_flipped: bool = False
        self.colormap_start: float = None
        self.colormap_end: float = None
        self.output_path: str = None
        self._isDirty: bool = False

    def set_output_path(self, output_path):
        self.output_path = output_path

    def to_yaml(self, path):
        data = dict(
            linescan_folder=self.linescan_folder,
            linescan_settings_path=self.linescan_settings_path,
            gocator_folder=self.gocator_folder,
            keypointsfg={
                key: dict(x=self.keypointsfg[key].x, y=self.keypointsfg[key].y)
                for key in self.keypointsfg
            },
            keypointsbg={
                key: dict(x=self.keypointsbg[key].x, y=self.keypointsbg[key].y)
                for key in self.keypointsbg
            },
            calibration_ball=dict(
                x=self.calibration_ball.x,
                y=self.calibration_ball.y,
                w=self.calibration_ball.w,
                h=self.calibration_ball.h,
            )
            if self.calibration_ball is not None
            else None,
            calibration_ball_diameter_mm=self.calibration_ball_diameter_mm,
            linescan_flipped=self.linescan_flipped,
            gocator_flipped=self.gocator_flipped,
            colormap_start=self.colormap_start,
            colormap_end=self.colormap_end,
        )
        try:
            with open(path, "w+") as outfile:
                yaml.dump(data, outfile, default_flow_style=False)

            logger.info(f"[Aliengment]Exported project to {path}")
            self.output_path = path
            return True
        except:
            logger.error(f"[Aliengment]Failed to exported project to {path}")
            return False

    def setDirty(self, isDirty=True):
        self._isDirty = isDirty

    def isDirty(self):
        return self._isDirty

    def from_yaml(self, path):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                self.linescan_folder = data["linescan_folder"]
                self.linescan_settings_path = data["linescan_settings_path"]
                self.gocator_folder = data["gocator_folder"]
                self.keypointsbg = {
                    key: Point(
                        x=data["keypointsbg"][key]["x"], y=data["keypointsbg"][key]["y"]
                    )
                    for key in data["keypointsbg"]
                }

                self.keypointsfg = {
                    key: Point(
                        x=data["keypointsfg"][key]["x"], y=data["keypointsfg"][key]["y"]
                    )
                    for key in data["keypointsfg"]
                }
                self.calibration_ball = (
                    Rect(
                        x=data["calibration_ball"]["x"],
                        y=data["calibration_ball"]["y"],
                        width=data["calibration_ball"]["w"],
                        height=data["calibration_ball"]["h"],
                    )
                    if data["calibration_ball"] is not None
                    else None
                )
                self.calibration_ball_diameter_mm = data["calibration_ball_diameter_mm"]
                self.linescan_flipped = data["linescan_flipped"]
                self.gocator_flipped = data["gocator_flipped"]
                self.colormap_start = data["colormap_start"]
                self.colormap_end = data["colormap_end"]

            self.output_path = path
        except yaml.YAMLError as exc:
            print(exc)
