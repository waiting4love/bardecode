from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


@dataclass
class DecodedBarcode:
    format: str
    text: str
    bbox: list[int]
    detection_score: float


@dataclass
class ImageResult:
    image: str
    width: int
    height: int
    barcodes: list[DecodedBarcode] = field(default_factory=list)
    undecoded: list[dict] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class Report:
    tool: str
    version: str
    results: list[ImageResult] = field(default_factory=list)
