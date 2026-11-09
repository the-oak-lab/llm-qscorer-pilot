# Jev scorer and the Jev-fitted model

This folder holds a slope regression fitted on Jev's checklist scores. The model in `models/spacing-exp4` is fitted on Gemini Flash scores.

## How Jev scores the checklist

Jev (`typesafe/jev-1.14`) is a classifier model. It returns one yes/no probability per feature and generates no text. OpenRouter serves it through `/api/alpha/decisions`.

| | Gemini Flash (current pipeline) | Jev |
|---|---|---|
| Output | generated JSON with one 0/1 answer per feature | one probability per feature |
| Calls per question | 1 call per sample (`samples` 1 to 3), all 45 features | 1 request with all 45 features |
| Run-to-run agreement | Spearman 0.74 between 2 runs (predicted slope) | mean probability change 0.012 between 2 runs |
| Cost per question (holdout runs) | $0.016 per sample | $0.00024 |
| Median latency per call (holdout runs) | 20.9 s | 0.3 s |

The tuned Jev request (`jev_variant="tuned"`) changes three things in what Jev reads.

1. Each feature sends its statement as the question and its definition as the criteria.
2. Ten compound features (`configs/jev-splits.json`) are also sent as separate parts. The code joins the parts with a logical OR. The joined and split forms stay in the output so that you can compare them.
3. Four features (f11, f19, f22, f27) use rewordings chosen in `scripts/test_jev_phrasings.py` (`configs/jev-phrasings.json`).

## How the model was fitted

- Data: the researchers' spacing-exp4 file `questions-exp4.csv` (341 questions in 4 item sets). This file supplies the observed slopes and labels. Options are in the original order.
- Inputs: Jev's raw probabilities from the tuned request (the `.joined` columns), one request per question.
- Fit: the repo's ridge regression `qscorer.predict.fit`, unchanged. It is weighted by 1/slope_sd², demeaned within set, and alpha is chosen by leave-one-set-out CV. The chosen alpha is 1.0.
- Sign stability: 1000 bootstrap refits.
- Script: `python scripts/fit_jev_model.py`. Scoring all 341 questions cost $0.08.

The model does not use the Gemini calibration (`configs/jev-calibration-gemini.json`). That calibration maps Jev's probabilities to Gemini's answers. It lowered held-out accuracy in every test (see below).

## Results

All numbers are held out, by leave-one-set-out, unless the row says in-sample. "Low" means P(slope > 0) ≤ 0.70. The base rate of low questions is 25%.

### Against the researchers' slopes (341 questions, `scripts/compare_researchers.py`)

| Predictor | Within-set Spearman | AUC high vs low | Low share, bottom 10% | Low share, bottom 20% |
|---|---|---|---|---|
| Gemini `checklist_cv` (researchers) | 0.233 | 0.636 | 0.47 | 0.40 |
| Jev raw, Jev-fitted model | 0.179 | 0.590 | 0.56 | 0.43 |
| Jev with Gemini calibration | 0.132 | 0.563 | 0.47 | 0.35 |
| Jev raw, shuffled option order (339) | 0.154 | 0.578 | 0.53 | 0.40 |
| `stage1_known` (researchers' baseline) | 0.218 | 0.651 | 0.24 | 0.25 |
| `stage1_improve` (researchers' baseline) | 0.122 | 0.541 | 0.41 | 0.31 |
| Gemini shipped model, in-sample | 0.488 | 0.795 | 0.62 | 0.60 |

Paired bootstrap (2000 resamples within set), Jev raw minus Gemini `checklist_cv`:

- Within-set Spearman: −0.052 (95% CI −0.162 to +0.061)
- AUC: −0.043 (−0.116 to +0.030)
- Low share in the bottom 10%: +0.075 (−0.118 to +0.265)

The held-out predictions of Jev and Gemini correlate at Spearman 0.57.

### Agreement with Gemini on unseen questions (holdout of 100, `scripts/holdout_table.py`)

These questions were not used to tune Jev. Each scorer is compared with the average of 2 Gemini runs, through the Gemini model.

| Scorer | Feature kappa | Slope Spearman | Mean abs slope gap | Bottom-20 shared |
|---|---|---|---|---|
| Gemini run 2 against run 1 (noise floor) | 0.78 | 0.74 | 0.040 | 14 |
| Haiku 5.5, 1 run | 0.55 | 0.46 | 0.071 | 9 |
| Jev, untuned (cut 0.7) | 0.53 | 0.45 | 0.064 | 6 |
| Jev, tuned and calibrated to Gemini | 0.57 | 0.53 | 0.053 | 9 |

### Each scorer with its own fitted model (200 questions with slopes from the DataShop export, `scripts/fit_by_scorer.py`)

| Scorer | CV within-set Spearman | Held-out Spearman (95% CI) | AUC high vs low |
|---|---|---|---|
| Gemini, 2-run average | 0.193 | 0.17 (0.02 to 0.31) | 0.619 |
| Haiku 5.5, 1 run | 0.138 | 0.12 (−0.02 to 0.26) | 0.589 |
| Jev raw | 0.153 | 0.09 (−0.06 to 0.23) | 0.558 |

### Coefficients

13 of 45 features have a stable sign (≥ 0.9 of bootstrap refits) in the Jev model. The Gemini model has 20. Nine features are stable in both models with the same sign.

| Feature | Jev coef | Gemini coef | Direction |
|---|---|---|---|
| f18 Correct-Option Length and Elaboration Asymmetry | +0.253 | +0.096 | more learning |
| f20 Claim-and-Rationale Response Options | +0.101 | +0.067 | more learning |
| f01 Provision of Reference Data in the Stem | +0.100 | +0.076 | more learning |
| f44 Distractor-Addressing and Refutational Feedback | +0.092 | +0.050 | more learning |
| f10 Scenario-Based Stem Contextualization | −0.110 | −0.050 | less learning |
| f14 Everyday Intuitive Solvability | −0.134 | −0.090 | less learning |
| f31 Transferable Rule and Mechanistic Derivation in Feedback | −0.140 | −0.027 | less learning |
| f02 Quantitative Calculation Requirement | −0.171 | −0.030 | less learning |
| f15 Elaborated Verbal Response Options | −0.177 | −0.087 | less learning |

f03 (+0.181) and f39 (+0.108) are stable in Jev only. Their Gemini signs are negative.

## Summary of the data

- Jev with its own model ranks questions less well than the researchers' Gemini model (within-set Spearman 0.18 against 0.23). The bootstrap CIs of the difference include zero.
- In the bottom 10% of predictions, Jev's share of low questions is 0.56 and Gemini's is 0.47. The CI of the difference includes zero.
- Calibrating Jev to Gemini's answers raised agreement with Gemini (holdout slope Spearman 0.45 to 0.53). It lowered agreement with observed learning (within-set Spearman 0.18 to 0.13).
- Haiku agrees with Gemini about as much as Jev does (holdout slope Spearman 0.46 against 0.45 untuned). Different scorers give different feature readings.
- Option order changes Jev's probabilities by 0.017 on average.
- One Jev request costs about 65 times less than one Gemini sample, and its median latency is about 70 times lower.

## Limits

- All fits and tests use one experiment (spacing-exp4, high-school science). Transfer to other content is not tested.
- The holdout shows agreement with Gemini, not with observed learning. The Gemini model was fitted on those questions, so its observed-learning numbers there are in-sample.
- The 200-question fits use slopes computed from the DataShop export with `scripts/run_slopes.py`. They correlate at r = 0.996 with the researchers' slopes, and their split-half reliability is 0.48.
