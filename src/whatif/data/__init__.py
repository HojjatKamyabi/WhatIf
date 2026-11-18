"""WhatIf's data layer: German political and public data, with licence and provenance built in.

One module per source. Every adapter refuses to download until its licence is recorded, stores
each download with its source, licence and date, and is read offline afterwards. Nothing here
depends on the rest of WhatIf, so it can become a package of its own once others use it.
"""
