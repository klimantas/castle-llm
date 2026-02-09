"""Pydantic models for structured LLM outputs."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, constr


class Action(BaseModel):
    """Structured action output from the LLM."""
    
    reasoning: Optional[constr(max_length=200)] = Field(
        None, description="Optional short reasoning (<=200 chars)"
    )
    action: Literal[
        "MONITOR",
        "ANALYSE",
        "DECOY_APACHE",
        "DECOY_FEMITTER",
        "DECOY_HARAKASMPT",
        "DECOY_SMSS",
        "DECOY_SSHD",
        "DECOY_SVCHOST",
        "DECOY_TOMCAT",
        "DECOY_VSFTPD",
        "REMOVE",
        "RESTORE",
        "SLEEP",
    ]
    host: Literal[
        "Defender",
        "Enterprise0",
        "Enterprise1",
        "Enterprise2",
        "Op_Host0",
        "Op_Host1",
        "Op_Host2",
        "Op_Server0",
        "User0",
        "User1",
        "User2",
        "User3",
        "User4",
        "nohost",
    ] = Field(
        default="nohost",
        description="Host to perform the action on. Default is 'nohost' for actions that do not require a host argument.",
    )
