# CO2-Sequestration-Simulation
Accelerating CO2 Sequestration Simulation with Generative Machine Learning Models


# Data preperation - 

1. Load dataset correlation_wide.csv from Google Drive.
2. Identify static features (minerals, temp, shift, year).
3. Deduplicate to one series per unique static configuration.
4. Trim all series to 101 timesteps.
5. Run KMeans (k=8) on normalized trajectories → cluster labels.
   
Outputs:

merged_df: static features + cluster label.
df_output: long-format CO₂ series (file_id, timestep, CO2).

#  TFT model 
Data used by the model - 

  Target (time series): Total_CO2_capture — CO₂ value at each timestep 
  Static features (constant within a series):
      Categorical: temp, shift
      Continuous: MikeSorghum, Quartz, Plagioclase, Apatite, Ilmenite, Diopside_Mn, Diopside, Olivine, Alkali-feldspar, Montmorillonite, Glass
  Known-future covariates (time-varying, known ahead):  time_idx, time_sin, time_cos (+ year if varying).
  Inputs expected from data prep:
    merged_df — one row per file_id with the static features above
    df_output — long-format CO₂ series: ['file_id','timestep','CO2'] (exactly 101 steps)

Datasets (disjoint by series) - 

  Train: 70% of file_ids — used to fit the model (sliding windows within each train series).
  Validation: 10% of file_ids — used for checkpointing/early stopping only on the final horizon.
  Test: 20% of file_ids — held-out evaluation only on the final horizon.
  No file_id appears in more than one split.

X→Y forecasting protocol - 

  Total length fixed at 101.
  Choose an encoder length X (e.g., 80) → Y = 101 − X (e.g., 21).
  Model inputs per batch:
    Encoder (past X): past CO₂ (encoder_target) + static features + time covariates
    Decoder (future Y): known-future covariates only (decoder_cont/cat/time_idx)
    Labels: true CO₂ for the Y steps (decoder_target) — used only for loss/metrics, never as inputs.
  Default X–Y splits run: (80,21), (60,41), (50,51), (40,61), (20,81), (10,91), (5,96).

Model layers - 

Static encoders & static context
Embed static categorical features and project static continuous features; produce a context vector that conditions downstream modules (lets dynamics depend on geology/conditions).

Variable Selection Networks (VSNs)
Per-timestep soft selection over input variables using Gated Residual Networks (GRNs) → compact, informative representation before sequence modeling.

LSTM encoder–decoder (1 layer, tiny)
Encodes past X dynamics and conditions decoder over Y; captures short/medium-range temporal patterns.

Temporal self-attention (decoder steps)
Single-head attention over the Y horizon; enforces coherence across the forecasted segment.

Gated Residual Networks (GRNs) & gating
Nonlinear mixing with residual paths and gates for stability; used in VSNs, post-attention, and context conditioning.

Output head (point forecasts)
output_size=1 with RMSE loss → predictions in original CO₂ units by default (no target normalization).

Config 
hidden_size=64, hidden_continuous_size=64, lstm_layers=1, attention_head_size=4, dropout=0.1, lr=1.5e-3, batch_size=512, epochs=300, early stopping confidence = 50.




