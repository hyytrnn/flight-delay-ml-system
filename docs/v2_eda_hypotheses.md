# V2 EDA Hypotheses

Each hypothesis is falsifiable and must be evaluated with support counts and
rolling temporal folds. The September 2025 final holdout remains closed.

| ID | Hypothesis | EDA evidence needed | Feature or experiment | T-24h status |
|---|---|---|---|---|
| H1 | Late local departures have higher delay risk across seasons | hour rate and hour by season | cyclical hour and route-season interaction | GREEN schedule proxy |
| H2 | Carrier effects vary by origin rather than globally | carrier-origin support and rate | carrier_origin interaction | YELLOW BTS schedule proxy |
| H3 | Route effects persist after controlling for support | route rate with Wilson interval | carrier_route and route frequency | YELLOW |
| H4 | U.S. holiday proximity changes delay prevalence | monthly and holiday-distance rates | holiday and days-to-holiday | GREEN calendar |
| H5 | High planned origin load raises delay risk | load-bin rates by airport and hour | P2 retrospective congestion proxy | YELLOW until schedule snapshot exists |
| H6 | Airport geography adds information beyond airport IDs | elevation and distance ablation | F3 static geography | GREEN when reference version is valid |
| H7 | Model quality varies more by time than headline mean suggests | fold prevalence and worst-fold PR-AUC | rolling-origin model selection | GREEN evaluation protocol |
| H8 | Weather improves ranking on matched coverage | weather coverage and extremes | issued-run F5 ablation | GREEN only when issue time is verified |
| H9 | Class weighting improves recall but may harm calibration | fold PR-AUC Brier and reliability | P5 weighting pilot | GREEN train-fold only |
| H10 | Flow forecasts help only if out-of-fold and T-24h | flow autocorrelation and forecast residuals | auxiliary flow_hat | YELLOW until source timing is proven |

