# Feasibility policy

Single adopted model: minimum-downside risk minimization under a required net expected return. The risk--mean scalarized objective of the earlier revision is removed and is not retained.

Revision LP scales: [1.0, 0.95, 0.9, 0.85]; same across k, RVaR/CVaR, endpoint/interval/heterogeneous experiments.

Only infeasibility triggers target reduction. No fallback; failures retain NaN weights.

Each trajectory starts in cash, then references its own last accepted weights, reindexed to the current pool. Forced exits from a changing stock universe enter the objective and minimum-return constraint.

Bootstrap samples always use the reference trajectory's initial weights for that window. No bootstrap sample affects the next sample or next window.

3n variables; 2n+3 inequalities; 3n lower bounds, n finite upper bounds. Purchases and sales both incur a 0.30% fee; costs are charged per rebalance.
