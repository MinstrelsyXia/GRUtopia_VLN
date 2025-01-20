# SensorParams
from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel


class CameraConfigEnum(str, Enum):
    no_rgb = 'no_rgb'
    point_cloud = 'point_cloud'
    distance2camera = 'distance2camera'
    bounding_box_2d = 'bounding_box_2d'
    distance2image_plane = 'distance2image_plane'


class SensorParams(BaseModel):
    """
    Sensor config validator
    """
    name: str
    enable: Optional[bool] = True
    size: Optional[Tuple[int, int]] = None  # Camera only
    camera_config: Optional[List[CameraConfigEnum]] = None  # Camera only
    scan_rate: Optional[int] = None  # RPS. Lidar only