from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class TraceabilityID(BaseModel):
    lot_id: str
    wafer_id: Optional[str] = None
    die_id: Optional[str] = None

class LogEntry(BaseModel):
    timestamp: datetime
    level: str
    module: Optional[str]
    message: str
    raw: str
    file: str
    severity: int

class WaferBinMap(BaseModel):
    lot_id: str
    wafer_id: str
    bin_map: List[List[int]]
    defect_type: Optional[str] = None