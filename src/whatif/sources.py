"""Every registered data adapter. Adding an adapter here gives it the shared contract check."""

from whatif.data.adapters import Adapter
from whatif.data.eurostat import INFLATION, UNEMPLOYMENT
from whatif.data.elections import ELECTIONS
from whatif.events import GDELT, PAGEVIEWS
from whatif.data.polls import DAWUM

ADAPTERS: tuple[Adapter, ...] = (
    DAWUM,
    *(adapter for election in ELECTIONS for adapter in (election.results, election.structure)),
    INFLATION,
    UNEMPLOYMENT,
    GDELT,
    PAGEVIEWS,
)
