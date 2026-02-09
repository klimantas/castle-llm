"""Action conversion and mapping utilities."""

from typing import Optional, Any
from CybORG.Shared.Actions import *
from .models import Action


# Mapping of action names to CybORG action constructors
ACTION_MAP = {
    "SLEEP": lambda h, s: Sleep(),
    "MONITOR": lambda h, s: Monitor(agent="Blue", session=s),
    "ANALYSE": lambda h, s: Analyse(hostname=h, agent="Blue", session=s),
    "DECOY_APACHE": lambda h, s: DecoyApache(hostname=h, agent="Blue", session=s),
    "DECOY_FEMITTER": lambda h, s: DecoyFemitter(hostname=h, agent="Blue", session=s),
    "DECOY_HARAKASMTP": lambda h, s: DecoyHarakaSMPT(hostname=h, agent="Blue", session=s),
    "DECOY_SMSS": lambda h, s: DecoySmss(hostname=h, agent="Blue", session=s),
    "DECOY_SSHD": lambda h, s: DecoySSHD(hostname=h, agent="Blue", session=s),
    "DECOY_SVCHOST": lambda h, s: DecoySvchost(hostname=h, agent="Blue", session=s),
    "DECOY_TOMCAT": lambda h, s: DecoyTomcat(hostname=h, agent="Blue", session=s),
    "DECOY_VSFTPD": lambda h, s: DecoyVsftpd(hostname=h, agent="Blue", session=s),
    "REMOVE": lambda h, s: Remove(hostname=h, agent="Blue", session=s),
    "RESTORE": lambda h, s: Restore(hostname=h, agent="Blue", session=s),
}


def convert_action_to_cyborg(output: Action, session: int = 0) -> Optional[Any]:
    """Convert a parsed Action model to a CybORG action.
    
    Args:
        output: Pydantic Action model containing action type and host.
        session: Session ID for the action (default: 0).
        
    Returns:
        A CybORG action object, or None if the action type is invalid.
    """
    action_factory = ACTION_MAP.get(output.action.upper())
    return action_factory(output.host, session) if action_factory else None


def get_fallback_action(session: int = 0) -> Any:
    """Get a safe fallback action when parsing fails.
    
    Args:
        session: Session ID for the action.
        
    Returns:
        A Monitor action as a safe default.
    """
    return Monitor(agent="Blue", session=session)
