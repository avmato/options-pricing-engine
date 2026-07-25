"""optlab -- a no-arbitrage audit of listed option quotes.

The library answers one question: when a published option chain looks like it
contains free money, how much of that is the market, how much is the
modelling convention, and how much is simply the bid-ask spread?

Layers
------
``optlab.core``
    Black-76 pricing, Greeks and implied volatility, all parameterised by the
    forward rather than by an assumed rate and dividend yield.
``optlab.market``
    Quote metrics, filtering, and the forward curve implied by put-call
    parity.
``optlab.audit``
    Static arbitrage checks, each run at the mid and at executable prices.
``optlab.study``
    Analyses assembled from the layers above.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
