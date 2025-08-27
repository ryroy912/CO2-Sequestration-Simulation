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
## Data used by the model - 

  Target (time series): Total_CO2_capture — CO₂ value at each timestep 
  Static features (constant within a series):
      Categorical: temp, shift
      Continuous: MikeSorghum, Quartz, Plagioclase, Apatite, Ilmenite, Diopside_Mn, Diopside, Olivine, Alkali-feldspar, Montmorillonite, Glass
  Known-future covariates (time-varying, known ahead):  time_idx, time_sin, time_cos (+ year if varying).
  Inputs expected from data prep:
    merged_df — one row per file_id with the static features above
    df_output — long-format CO₂ series: ['file_id','timestep','CO2'] (exactly 101 steps)

## Datasets (disjoint by series) - 

  Train: 70% of file_ids — used to fit the model (sliding windows within each train series).
  Validation: 10% of file_ids — used for checkpointing/early stopping only on the final horizon.
  Test: 20% of file_ids — held-out evaluation only on the final horizon.
  No file_id appears in more than one split.

## X→Y forecasting protocol - 

  Total length fixed at 101.
  Choose an encoder length X (e.g., 80) → Y = 101 − X (e.g., 21).
  Model inputs per batch:
    Encoder (past X): past CO₂ (encoder_target) + static features + time covariates
    Decoder (future Y): known-future covariates only (decoder_cont/cat/time_idx)
    Labels: true CO₂ for the Y steps (decoder_target) — used only for loss/metrics, never as inputs.
  Default X–Y splits run: (80,21), (60,41), (50,51), (40,61), (20,81), (10,91), (5,96).

## Model layers - 

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

## Config 
hidden_size=64, hidden_continuous_size=64, lstm_layers=1, attention_head_size=4, dropout=0.1, lr=1.5e-3, batch_size=512, epochs=300, early stopping confidence = 50.


#  Nlinear model 

## Data used by the model : 

   Target (time series): Total_CO2_capture — CO₂ value at each timestep (101 steps per file_id).
   Inputs expected from data prep:
      df_output — long-format CO₂ series: ['file_id','timestep','CO2'] (exactly 101 steps)
      merged_df is produced by prep but unused here.

## Datasets (disjoint by series) : 

Series are split by file_id using train_test_split:
   First split: 80% → train+val, 20% → test
   Second split (on the 80%): 80% → train, 20% → val
   Net result: Train 64% / Val 16% / Test 20% (no file_id overlap)

## X→Y forecasting protocol : 
   For each split (X, Y) (e.g., (80, 21) on a 101-step series):
   Build X = first X timesteps, Y = last Y = 101−X timesteps per series:
      X = CO2[:, :X]         # inputs
      Y = CO2[:, X:]         # labels
   Splits run by default: (80,20), (60,40), (40,60), (20,80), (10,90), (5,95)

## Model architecture : 

Class: NLinear(seq_len=X, pred_len=Y, individual=False)
Flow (univariate):
Input squeeze: (B, X, 1) → (B, X)
**** Level removal: seq_last = x[:, -1:]; x ← x − seq_last *** (Most important step 1/2) - subtract the last observed value of the input window from the whole window!
Makes the linear map operate on a level-normalized sequence (helps stability).
Linear map: out = W · x + b, where W ∈ ℝ^{Y×X}
If individual=True (not used here), a per-channel linear layer is applied (for multivariate).
Add back level: out ← out + seq_last *** (Most important step 2/2) - here the value is added back
Reshape: (B, Y) → (B, Y, 1)
What the parameters capture:
   W learns a direct linear relationship from the last X points to the next Y points (one-shot).
   The residual connection via seq_last gives a simple, adaptive level baseline the linear map can refine.

## Config 

    'epochs': 500,
    'batch_size': 64,
    'learning_rate': 0.001,
    'individual': False,

# LSTM model : 

## Data used by the model

Target (time series): Total_CO2_capture — CO₂ value at each timestep (101 steps per file_id).
Static features (constant within a series): temp, shift, MikeSorghum, Quartz, Plagioclase, Apatite, Ilmenite, Diopside_Mn, Diopside, Olivine, Alkali-feldspar, Montmorillonite, Glass

## Datasets (disjoint by series) 
Train/Test split by file_id: 80% train+val / 20% test (no overlap).
Validation from training set: random 10% of the training series for validation.
Effective proportions: Train ≈ 72% / Val ≈ 8% / Test 20% (by series).

## X→Y forecasting protocol
For each split (X, Y) (e.g., (80, 21) on a 101-step series):
Build X = first X timesteps, Y = last Y = 101−X timesteps per series:
   X = CO2[:, :X]         # inputs
   Y = CO2[:, X:]         # labels
Model inputs (per batch):
   x_seq: [B, X, 1] past CO₂ sequence
   x_static: [B, static_dim] static feature vector per series
Model output: [B, Y] (one-shot multi-horizon)
Splits run by default:
   (80,20), (60,40), (40,60), (20,80), (10,90), (5,95)


## Model architecture - 

1. Static encoder (MLP):
      static_fc: Linear(static_dim→H) → ReLU → Linear(H→H)
      Encodes static categorical/continuous features into a context vector [B, H]
2. Time conditioning with statics:
         Expand static context across time and concatenate with each timestep of x_seq:
         lstm_input = concat([x_seq, static_context], dim=-1) → shape [B, X, 1+H]
3. LSTM encoder (sequence-to-sequence):
         lstm: LSTM(input_size=1+H, hidden_size=H, num_layers=2, batch_first=True)
         Produces hidden states [B, X, H].
4. Sequence pooling:
      Take the last timestep hidden: last_hidden = lstm_out[:, -1, :] ([B, H]
5. Output head (MLP, MIMO):
       output_fc: Linear(H→H) → ReLU → Linear(H→Y)
       Direct multi-horizon prediction of the next Y values (no autoregressive loop)


## Config 
Optimizer / Loss: Adam (lr=1e-3), nn.MSELoss()
Epochs / Batch size: 500 / 64
Validation: best model selected by validation MSE on the 10% validation subset of training series.
Test metric: averaged MSE over all test batches (held-out 20% series).

# DSSM - TS Gluon model 

## Data used by the model : 
   Series ID: file_id (one independent time series per id)
   Time index: timestep (0…100), total length per series = 101
   Target (observed): CO2 (from Total_CO2_capture)
   Static continuous features (replicated across time):
   MikeSorghum, Quartz, Plagioclase, Apatite, Ilmenite, Diopside_Mn, Diopside, Olivine, Alkali-feldspar, Montmorillonite, Glass, temp, shift, year (Only features present in the CSV are used; temp/shift/year are coerced to numeric and NaNs → 0.)
   No static categoricals are used in the model.

## Datasets
   Series split by file_id: ~70% train, 10% validation, 20% test (disjoint sets).
   Windowing per series (total length 101):
      Train: target length = X (first X steps).
      Val/Test: target length = X+Y (so the evaluator can score the last Y).
   Known covariates: the static features are replicated across time and passed as feat_dynamic_real.
      For each item, feat_dynamic_real has shape [F_static, T] and T must equal the target length of that dataset (X for train, X+Y for val/test)

## Forecasting X - > Y  

   During training, the model never sees future targets (Y) nor future covariates beyond time X.
   Validation/test include the full X+Y targets only so GluonTS can compute metrics on the final Y horizon.
   Example (X=80, Y=21):
      Train: target length 80; dynamic reals length 80.
      Val/Test: target length 101; dynamic reals length 101.
      Metric computed on the final 21 steps of each val/test series.

## Model architecture :

DeepState is a global deep state-space model (Rangapuram et al., NeurIPS 2018):
RNN encoder (LSTM): processes the historical target (and covariates) to produce time-varying state-space parameters for each step.
Config here: num_layers=2, num_cells=40, dropout=0.1.
Linear State Space Model (SSM) + Kalman Filter:
Learns latent dynamics (transition & emission) per time step.
Produces a probabilistic forecast by rolling the SSM forward for Y steps.
Trainer (GluonTS/MXNet): minibatch training by maximizing the likelihood of the observed series.
Intuition:
   The RNN captures global, non-linear patterns shared across series and conditions the SSM.
   The SSM provides stable linear dynamics and uncertainty via the Kalman filter/smoother.
   Together: robust multi-step forecasting with calibrated uncertainty.

## Configuration 

CFG = {
  "freq": "D",                  # date frequency label (synthetic here)
  "learning_rate": 1e-3,
  "num_layers": 2,
  "num_cells": 40,
  "dropout_rate": 0.1,
  "epoch_grid": [10, 20, 40],   # train separate models for each; pick best on VAL
  "num_batches_per_epoch": 100, # controls epoch length; raise for final runs (e.g., 200)
  "val_frac": 0.10,
  "test_frac": 0.20
} 

Per split:

      past_length = X
      prediction_length = Y
      use_feat_dynamic_real = True      # statics replicated across time
      use_feat_static_cat  = False
      cardinality = [1]                 # required by GluonTS even when no static cats

## Files and running 
prep.py — loads CSV, builds:
merged_df (file_id + static continuous features)
df_output (file_id, timestep, CO2)
deepstate_run.py — builds datasets, trains DeepState with validation selection, saves DeepState_RESULTS.csv.

Data prep (writes merged_df.parquet / df_output.parquet) : !conda run -n mx19 python -u prep.py
Train & evaluate DeepState (streams logs, writes DeepState_RESULTS.csv) : !conda run -n mx19 python -u deepstate_run.py



