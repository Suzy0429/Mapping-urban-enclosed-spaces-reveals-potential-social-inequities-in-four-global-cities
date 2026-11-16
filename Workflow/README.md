# Workflow

Documentation of the complete processing chain, together with runnable code for
the two stages of it that are specific to this study and are released here: the
street-view sampling geometry and the estimation of the enclosure threshold.

## Environment

| Component | Version used |
|---|---|
| Python | 3.9.21 |
| Python for step 1 | ArcGIS Pro's own Python, 3.9 (arcpy, numpy 1.20.1, pandas 1.4.4) |
| numpy | 2.0.1 |
| pandas | 2.2.3 |
| scipy | 1.13.1 |
| shapely | 2.0.7 |
| networkx | 3.2.1 |
| gensim | 4.3.2 |
| scikit-learn | 1.9.1 |

Step 2 has no ArcGIS dependency; step 1 needs arcpy. Both take their inputs and
their output directory as arguments, so no path is hard-coded.

## Execution order

```
01_sampling_points.py          -> points_<city>.csv     one row per sampling point
   |
   v
   street-view retrieval       Baidu Maps and Google Street View; paper Section 2.2
   v
   semantic segmentation       Mask2Former, Cheng et al. (2022), Swin-L IN21k
   v
   closure degree FBW          share of a parcel's sampling points whose image
   |                           contains Fence, Wall and Building; Section 2.3
   v
02_threshold_estimation.py     -> threshold_report.txt  T, its identification
   |                           interval, bootstrap intervals
   v
   functional classification   word embeddings over the block corpus and a random
   |                           forest; Section 2.5, Supplementary Section 9
   v
   connectivity                interface permeability kappa and topological
   |                           embeddedness tau, then C = tau * kappa;
   |                           Section 2.6, Supplementary Section 12
   v
   social potential Phi        coefficient-of-variation weights over the four
   |                           indicators; Supplementary Section 13
   v
   four-quadrant analysis      each city's own mean Phi and mean C; Section 3
```

## Steps released here

### 01_sampling_points.py

Places sampling points along a 10 m outward buffer of every parcel edge at 50 m
intervals, and computes for each point the clockwise bearing from due north to
the parcel's visual centroid. A street-view image is later requested at that
point with a 90° field of view centred on that bearing, so the boundary is
photographed frontally rather than obliquely.

```bash
"<ArcGIS Pro>/bin/Python/envs/arcgispro-py3/python.exe" 01_sampling_points.py \
    --parcels parcels_beijing.shp --out-dir out --city beijing
```

| Parameter | Value | Meaning |
|---|---|---|
| `--buffer` | 10 m | outward buffer along the parcel edge |
| `--spacing` | 50 m | target spacing between sampling points |
| `--min-points` | 4 | floor per buffer ring, so small parcels are still sampled |
| `--utm-epsg` | derived | UTM zone of the layer centre, so the metric step is locally correct |

The number of points on a ring is the integer nearest to (ring length / 50), and
the points are then spaced at (ring length / that integer) — the published
behaviour. `parcel_id` is carried from the parcel layer into the buffer and the
centroid layer, so points and centroids are matched by parcel rather than by
feature order.

Output columns: `point_id`, `parcel_id`, `lon`, `lat`, `angle_deg`.

### 02_threshold_estimation.py

Fits the enclosure threshold T on the hand-labelled parcels by exhaustive search
over the midpoints of the sorted closure degrees, taking the value that
maximises classification accuracy. The rule the threshold is applied through is
`ue = (FBW >= T) & (road == 1)`.

```bash
python 02_threshold_estimation.py \
    --train Data/UEthreshold_train.csv --out-dir out
```

| Parameter | Value | Meaning |
|---|---|---|
| `--train` | `Data/UEthreshold_train.csv` | the 700 hand-labelled parcels (`FBW`, `ue`) |
| `--validation` | optional | independently labelled parcels (`city`, `FBW`, `ue`, `road`), if available |
| `--parcels` | optional | all parcels (`city`, `id`, `FBW`, `road`, `area`), if available |
| `--bootstrap` | 10,000 | resamples for the threshold intervals |
| `--seed` | 20261023 | seed for the bootstrap |

Results on the hand-labelled parcels: T = 0.7350, accuracy 0.9900, F1 0.9909,
Youden's J 0.9807, all three criteria selecting the same value; the
identification interval is (0.7300, 0.7400]. The percentile interval from that
set is `[0.7350, 0.7350]` — degenerate, because resampling the same 700 parcels
cannot create a closure degree inside the empty band — and the script says so
rather than reporting it as precision. The non-degenerate interval in
Supplementary Table 13 comes from the independently labelled validation set.

## Steps documented but not released

The three stages below implement the study's methodological contributions. Their
procedure, parameters and seeds are documented here so that the chain can be
followed end to end; the code and the parcel-level identification results are
not redistributed, and are available from the corresponding authors on
reasonable request.

### Functional classification

Point-of-interest, area-of-interest and land-use features are joined to the city
blocks with an intersects predicate, each block becomes a text tally of the
feature classes it contains, the tallies are embedded with Word2Vec, and a
random forest trained on the manually labelled blocks assigns one of five
functional types to every block.

| Parameter | Value |
|---|---|
| feature-class column | `fclass` |
| block identifier | `OBJECTID` |
| tokenisation | `gensim.utils.simple_preprocess` — lowercase; split on non-word characters; underscores kept; tokens containing a digit, shorter than 2 or longer than 15 characters dropped |
| Word2Vec | CBOW, d = 100, window 5, negative 5, min_count 1, epochs 5, alpha 0.025, subsample 1e-3, seed 42, single worker |
| block vector | mean of the token vectors, all-zero where the block has no token |
| random forest | 200 trees, `max_features` = sqrt, `max_depth` = None, `min_samples_split` = 2, `min_samples_leaf` = 1, bootstrap on, no class weights, `random_state` = 42 |

The counts in the block text are not features. `simple_preprocess` drops any
token containing a digit, so `"restaurant: 3, cafe: 1"` tokenises to
`["restaurant", "cafe"]`: what the embedding sees is which feature classes the
block contains. Blocks whose text tokenises to nothing receive an all-zero
vector and are predicted through the classifier's default path; the affected
shares are quantified in the Supplementary Information.

### Connectivity

A parcel's topological embeddedness τ is the length-weighted mean edge
betweenness of the road segments inside it. The road network is turned into a
primal graph whose vertices are road vertices and whose edges are the segments
between consecutive vertices, edge weight is segment length, and betweenness is
approximated from k = 1000 sampled source nodes. Non-zero values are then
log10-transformed and min-max normalised within the city; parcels whose internal
network carries no betweenness keep zero.

Its interface permeability κ is the weighted density of access points along the
enclosure boundary: every road feature crossing the boundary contributes the
weight of its road class, the weights are summed and divided by the boundary
length, and the result is min-max normalised within the city.

| Parameter | Value |
|---|---|
| betweenness sources `k` | 1000 |
| betweenness seed | 0 |
| edge weight | segment length |
| working CRS | EPSG:3857 |
| road-class weights | motorway, motorway_link, primary, primary_link, trunk, trunk_link 10; secondary, secondary_link, tertiary, tertiary_link 8; living_street, residential, service 6; cycleway, bridleway, pedestrian 4; footway, path, steps, track, unclassified, unknown 2 |

Spatial connectivity is the product of the two, C = τ · κ. Both components lie
in [0, 1] after the within-city normalisation, so C does as well. C is defined
for enclosed parcels; an open parcel has no enclosure boundary for κ to
describe, and the four-quadrant analysis is restricted to enclosed parcels
throughout. Internal roads are never removed from the network: the graph is
built from the full road network and τ is read from the segments lying inside
the parcel, so τ describes the centrality the internal network would carry in
the mapped street network. Supplementary Section 12 reports the check in which
the network is rebuilt with every enclosed parcel's edges removed.

### Social potential

Social potential Φ is the weighted Euclidean distance from the origin over four
min-max normalised indicators — green-space density, infrastructure density,
building density and housing price. The weights are each indicator's coefficient
of variation divided by the sum of the coefficients, computed with a floor of
1 × 10⁻⁴ added to the normalised values so that the denominator is stable; the
distance itself uses the normalised values without the floor. The four weights
are 0.4283, 0.2487, 0.1483 and 0.1748 for green space, infrastructure, buildings
and housing price respectively.

## Verification

Step 2 reproduces the threshold reported in the paper on the hand-labelled
parcels: T = 0.7350, accuracy 0.9900, and the identification interval
(0.7300, 0.7400]. Applying the rule at that threshold to the full parcel set
returns the published enclosure shares of 54.5, 23.1, 5.4 and 7.8 per cent for
Beijing, London, Mumbai and New York. At B = 10,000 with seed 20261023 the
training-set bootstrap is degenerate exactly as Supplementary Table 13 reports,
and the validation-set interval reproduces as 0.7351 [0.7343, 0.7359] pooled.

## Random seeds

| Step | Seed | Where |
|---|---|---|
| 02 threshold bootstrap | 20261023 | `--seed` |
| 02 | none | the search itself is deterministic |
| 01 | none | deterministic |
| functional classification, Word2Vec | 42 | its script |
| functional classification, random forest and folds | 42 | its script |
| connectivity, betweenness source sampling | 0 | its script |

The word-embedding step is the only stochastic part of the functional
classification; the forest is deterministic given `random_state` = 42 and the
embeddings it consumes.
