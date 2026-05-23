from pydantic import BaseModel

class SegmentRequest(BaseModel):
    bboxes: list[list[float]]