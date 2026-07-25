# Method notes

Short derivations for the claims the README makes. Nothing here is original; the point is
that every number in the repository can be traced to a line of algebra.

## 1. The forward is the only sensible parameterisation

A long call and a short put at the same strike and expiry pay

$$(S_T - K)^+ - (K - S_T)^+ = S_T - K$$

in every state of the world. The package is a forward contract struck at `K`, so its value
today is `D(F − K)` where `F` is the forward price and `D` the discount factor:

$$C - P = D(F - K).$$

Read as a function of `K` this is a straight line: **intercept** `D·F`, **slope** `−D`.

Both quantities are therefore observable from the option quotes alone. That matters because
the alternative — supplying `r` and `q` by hand — requires two numbers nobody observes, and
gets them wrong in a way that is invisible until it contaminates the smile (§3).

## 2. Why the slope is useless and the intercept is not

The rate follows from the slope as `r = −log(D)/T`. Differentiating,

$$\frac{\partial r}{\partial D} = -\frac{1}{DT}.$$

At `T = 1/52`, a slope error of `10⁻³` becomes a rate error of `0.052`, i.e. 5.2 percentage
points. The intercept carries no such factor, and averaging over ~70 strikes shrinks its
error further.

Empirically the joint fit on this snapshot returns `D ∈ [1.006, 1.023]` — discount factors
above one, implying rates of −4% to −41%. Since the same regression has `R² > 0.9998`, this
is not a bad fit; it is a well-fit line whose slope is not identified to the precision the
rate demands.

**Conclusion.** Take `D` from the money market. Imply only `F` from the options, strike by
strike, as `F = K + (C − P)/D`, and use the dispersion of those per-strike estimates as a
data-quality measure.

## 3. A wrong forward opens a fake call/put gap

Let `V(σ, F)` be the Black-76 price. Holding the quote fixed and perturbing the forward by
`δF`, the implied volatility moves by

$$\delta\sigma \approx -\frac{\partial V/\partial F}{\partial V/\partial \sigma}\,\delta F
= -\frac{\Delta}{\mathcal{V}}\,\delta F.$$

Calls have `Δ > 0` and puts `Δ < 0`, while vega is positive for both. So a single error in
`F` moves call and put implied volatilities in **opposite directions**, producing a
call/put "disagreement" that looks like a market feature and is not one.

This is the mechanism behind Finding 3 in the README: assuming 4% and no dividends
misprices the 83-day forward by $2.24 and opens a gap that implying the forward closes.

## 4. The six static checks

For strikes `K₁ < K₂ < K₃` at one expiry, with `w = +1` for calls and `−1` for puts:

| Check | Relation | Portfolio if violated |
|---|---|---|
| Price bounds | `max(w(S−K),0) ≤ V ≤ S` (call) or `≤ K` (put) | buy below the floor / sell above the cap |
| Strike monotonicity | `C` falls, `P` rises in `K` | the vertical spread, for a credit |
| Vertical cap | spread `≤ K₂ − K₁` | sell the spread above its maximum payoff |
| Butterfly convexity | `w₁V(K₁) − V(K₂) + w₃V(K₃) ≥ 0` | buy the butterfly for a credit |
| Box spread | `D(K₂−K₁) ≤ box ≤ K₂−K₁` | buy below the floor / sell above the cap |
| Calendar | longer-dated American ≥ shorter-dated | buy the long, sell the short, for a credit |

Butterfly weights `w₁ = (K₃−K₂)/(K₃−K₁)` and `w₃ = (K₂−K₁)/(K₃−K₁)` handle unequal strike
spacing. A negative butterfly price is the same statement as a negative risk-neutral
density around `K₂`: convexity in strike and a valid implied distribution are one condition.

### Exercise style

American exercise widens the band in three places:

- **Floor.** The option can be exercised today, so `V ≥ max(w(S−K), 0)`, which for deep
  in-the-money puts is strictly above the discounted European payoff.
- **Cap.** `C ≤ S` and `P ≤ K`, not `D·F` and `D·K`.
- **Short box.** A long box below `D(K₂−K₁)` is arbitrage under either convention, since the
  exercise right can only add value. A *short* box can be assigned early, so it is capped at
  the undiscounted `K₂ − K₁`. Testing it against the European value is the single largest
  source of false positives in the suite: 310 of 1,574 on this snapshot.

## 5. Mid versus executable

Each check runs twice. At the `mid` basis both legs are priced at `(bid+ask)/2`; at the
`executable` basis legs you buy are priced at the ask and legs you sell at the bid. The
difference is the entire question: a portfolio that is only profitable at the midpoint is
not an arbitrage, it is the spread.

Put-call parity is reported as a diagnostic rather than as an arbitrage, because closing the
trade requires the underlying. The box spread is its executable counterpart: same
information, no stock leg.

## 6. Implied volatility

`V` is strictly increasing in `σ`, so the inversion is a bracketed root-find. Three details
matter:

1. **Out-of-the-money transformation.** An in-the-money quote is mostly intrinsic value.
   Inverting it directly means resolving a small extrinsic component inside a large number;
   parity maps it to the out-of-the-money twin first, `p = c − (F − K)` in the forward
   measure, and the cancellation disappears.
2. **Safeguarded Newton.** The Newton step `σ ← σ − (V(σ) − V*)/𝒱` converges quadratically
   but vega vanishes in the wings. Each iteration maintains a valid bracket and falls back
   to bisection whenever the Newton step leaves it.
3. **Identifiability.** When vega falls below the rounding error of the transformation
   itself, the price does not determine a volatility. Those quotes are returned as
   `NOT_IDENTIFIED`, not as a number.

Convergence is judged on the volatility step rather than the price residual, because a wing
option worth `10⁻⁹` would otherwise be declared converged while its price is still 10%
wrong.
