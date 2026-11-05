"""Every registered data adapter. Adding an adapter here gives it the shared contract check."""

from whatif.adapters import Adapter
from whatif.elections import ELECTIONS
from whatif.polls import DAWUM

ADAPTERS: tuple[Adapter, ...] = (
    DAWUM,
    *(adapter for election in ELECTIONS for adapter in (election.results, election.structure)),
)
