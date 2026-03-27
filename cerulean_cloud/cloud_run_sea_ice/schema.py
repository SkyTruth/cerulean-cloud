"""Schema for the sea-ice sync worker."""

from typing import Optional

from pydantic import BaseModel


class SyncRequest(BaseModel):
    """Optional request payload for manual or scheduler-triggered syncs."""

    force: bool = False


class SyncResponse(BaseModel):
    """Response returned by the sync worker."""

    status: str
    ran: bool
    cadence_days: int
    anchor_date: str
    reason: Optional[str] = None
    object_name: Optional[str] = None
