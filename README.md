# Flipkart Grid 2.0 — Traffic Demand Prediction

> **Final Score:** 91.75866 / 100 (Public Leaderboard)  
> A two-stage ensemble combining LightGBM and Seq2Seq Transformers for citywide traffic demand forecasting.

## Overview

This solution predicts normalized traffic demand at the granularity of `(geohash, 15-min timestamp)` for urban road cells. Given a full day (day-48) of traffic history and early-morning observations (day-49 slots 0–8), we forecast demand for the remaining slots (9–55) of day-49.