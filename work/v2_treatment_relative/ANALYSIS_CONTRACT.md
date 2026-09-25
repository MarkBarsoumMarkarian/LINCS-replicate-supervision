# V2 treatment-relative empirical-Bayes feasibility contract

Frozen before any new cross-compound outcome is calculated.

## Objective

Determine whether plate-matched treatment effects plus empirical-Bayes variance
moderation can support a powered, reproducible assay-resolution experiment on
the existing LINCS MCF7 10 micromolar, 6 hour Level-2 data.

This is a method-development gate. It may use self-comparisons and previously
opened controls, but it must not open a new candidate-pair outcome.

## Data and experimental unit

- Matrix: 5,157 profiles by 978 directly measured genes.
- Experimental unit: one compound on one RNA plate, after averaging replicate
  treatment wells on that plate.
- Primary control: vehicle wells from the same RNA plate.
- Representation: `log2(Level2 intensity)` minus the within-plate vehicle
  median for each gene.
- Untreated controls are excluded from the primary analysis.

## Development compounds

Only the following already exposed, deeply replicated compounds can train or
tune the primary method:

- BRD-K81418486, vorinostat
- BRD-A19037878, trichostatin A
- BRD-A75409952, wortmannin
- BRD-A19500257, geldanamycin

All other compounds remain unavailable for method selection in this gate.

## Primary statistical engine

Fit gene-wise linear models to development-compound plate signatures with
limma. Use empirical-Bayes posterior residual variances to form fixed gene
precision weights. The weights are learned only from the four development
compounds and are frozen before any future candidate pair is opened.

The primary whole-signature distance is the weighted root-mean-square
difference between two independently estimated treatment-effect vectors:

`sqrt(mean((effect_A - effect_B)^2 / posterior_variance))`.

This signed-data metric replaces Jensen-Shannon divergence; Jensen-Shannon is
not valid for signed treatment-relative effects.

The practical equivalence margin is fixed at the largest compound-specific
95th percentile of 1,000 same-compound, disjoint 6-versus-6 plate splits. It is
not recalibrated downward as the tested sample size increases. Six plates per
side is the prespecified reference design; the margin therefore represents the
worst observed active-compound disagreement at that minimum design.

For an evaluated comparison, a 95% nonparametric bootstrap interval is formed
for the frozen distance. The label is Confusable when the upper bound is below
the margin, Distinguishable when the lower bound is above it, and Inconclusive
otherwise. All inequalities are strict.

## Feasibility tests

1. Same-compound split-half stability for each development compound.
2. Leave-one-compound-out calibration transfer.
3. The already exposed vorinostat-versus-wortmannin comparison as a separated
   control.
4. Power curves at 6, 8, 10, 12, 14, and 16 plates per side.

Simulation seed: 20,260,923. Calibration uses 1,000 draws per compound. Power
uses 200 outer trials and 399 bootstrap draws per trial. The separated control
uses plate-paired vorinostat-versus-wortmannin differences; same-compound
experiments use disjoint plate groups.

The feasibility gate passes only if the frozen method simultaneously achieves:

- at least 80% correct same-compound/equivalent decisions at a plate count
  attainable by at least one untouched candidate pair;
- at least 90% correct separated-control decisions at that plate count;
- no more than 5% wrong-opposite-branch decisions in either direction; and
- materially better same-compound power than the V1 active-profile result.

If this gate fails, no new biological pair will be opened under this method.

## Secondary arm

GSVA/ssGSEA pathway aggregation is prespecified as a secondary representation
only after the primary feasibility gate is evaluated. It cannot be substituted
post hoc to rescue an unfavorable candidate-pair result.

## Locked items

- No DESeq2: the input is continuous Level-2 intensity, not integer counts.
- No scVI/deep generative model in the primary analysis.
- No pair choice based on expression outcomes.
- No claim that an inconclusive result proves similarity or difference.
- No guarantee of a preferred biological label.
