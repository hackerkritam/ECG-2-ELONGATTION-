# PhysioDyn: Prompt Pack for VS Code (Copilot / Claude Code)

Paste the prompts in order. Run and verify each step before moving to the next.
Prompt 0 is the standing context: paste it first, or save it as CLAUDE.md / .github/copilot-instructions.md.

---

## PROMPT 0: Project context (paste first / save as CLAUDE.md)

```
You are helping me implement the code for a research paper called "PhysioDyn:
Physiologically-Constrained Generative Models for Long-Duration ECG Synthesis and
Robust HRV Reconstruction". Use Python 3.10+, PyTorch, and a clean modular layout.

Architecture (three modules):
1. BMM (Beat Morphology Module): conditional 1D U-Net generator + projection critic,
   trained with WGAN-GP on single ECG beats (200 samples, 5 AAMI classes N,S,V,F,Q).
2. IDM (IBI Dynamics Module): GRU encoder -> latent Neural ODE (torchdiffeq, RK4)
   -> GRU decoder, models RR-interval sequences of length 15, 30, 200.
3. Composer: places BMM beats at times given by IDM RR intervals (time-warp +
   crossfade stitching), then adds respiration modulation, baseline wander, muscle noise.

Data: MIT-BIH Arrhythmia Database via the `wfdb` package (PhysioNet).
Preprocessing: resample to 250 Hz, Butterworth bandpass 0.5-40 Hz, z-score normalize.

Project rules:
- Project layout: physiodyn/{data,models,losses,metrics,train,eval,utils}/, configs/*.yaml,
  scripts/, outputs/{checkpoints,figures,tables,logs}/
- All hyperparameters come from YAML configs, never hard-coded.
- Set seeds everywhere (python, numpy, torch, cuda) with a single utility.
- Type hints and short docstrings. Use tqdm and logging. Support CPU and GPU (auto device).
- Never fabricate results. Every number or figure must come from actually running the code.
- Save checkpoints and metrics as JSON/CSV so tables can be regenerated.
- Before writing code, give a short file-by-file plan. Then write the code.
```

---

## PROMPT 1: Scaffolding and environment

```
Create the project skeleton for PhysioDyn: the folder layout from my context, a
requirements.txt (torch, torchdiffeq, numpy, scipy, pandas, matplotlib, seaborn, wfdb,
scikit-learn, pyyaml, tqdm, neurokit2 or antropy, fastdtw or tslearn), a
physiodyn/utils/seed.py, a physiodyn/utils/config.py that loads YAML into a dataclass or
dict, and three config files: configs/bmm.yaml, configs/idm.yaml, configs/composer.yaml.

Fill the configs with these values from the paper.

BMM: epochs 20, signal length 200, num_classes 5, batch_size 64, latent_dim 128,
label_embed_dim 64, lr 1e-4, Adam betas (0.0, 0.9), lambda_gp 10, lambda_l1 10,
n_critic 5, n_synthetic 25000, seeds [0,1,2,3] (up to 8).

IDM: ODE function MLP (64 hidden, Tanh), dynamics dz/dt = -alpha*z + MLP(z), alpha 0.1,
solver rk4 via torchdiffeq, rtol 1e-4, atol 1e-5, batch_size 8, Adam, noise sigma 0.01,
grad clip 1.0, lambda_temp 1.0 (called lambda_spike in the paper's Table 2),
lambda_latent 1e-4, lambda_freq 0.1, sequence lengths [15, 30, 200] (curriculum),
latent_dim and GRU hidden_dim configurable (default 64).

Composer: fs 250, respiration freq range 0.2-0.3 Hz, amplitude Ar 0.05, baseline wander
freq 0.05-0.5 Hz amplitude Ab 0.1, muscle noise std 0.02, crossfade length 10 samples,
bandpass 0.5-40 Hz.

Also add a Makefile or scripts/ with a README stub explaining how to run each stage.
```

---

## PROMPT 2: Data download, preprocessing, beat extraction

```
Implement physiodyn/data/ for the MIT-BIH Arrhythmia Database.

1. download.py: use wfdb.dl_database('mitdb', dl_dir=...) to download all 48 records.
2. preprocess.py:
   - Load each record (use lead MLII where available; skip records without it, e.g. log which).
   - Resample from 360 Hz to 250 Hz (scipy.signal.resample_poly 25/36).
   - Bandpass 0.5-40 Hz (4th order Butterworth, zero-phase filtfilt).
   - Z-score normalize per record.
   - Map R-peak sample indices from 360 Hz to 250 Hz consistently.
3. beats.py (for BMM): segment 200-sample windows centered on each annotated R-peak
   (100 before, 100 after). Map annotation symbols to AAMI classes:
   N: N,L,R,e,j | S: A,a,J,S | V: V,E | F: F | Q: /,f,Q
   Skip non-beat annotations. Output arrays X [n,200], y [n] in {0..4}.
   Print class counts (paper reports N 90,588 / S 2,781 / V 7,235 / F 802 / Q 33).
   Report any difference from these counts; do not force them.
4. rr.py (for IDM): from the .atr annotations, compute RR_i = (r_{i+1}-r_i)/fs in SECONDS
   using the original fs (360). Aggregate intervals into windows, then standardize each
   window to a fixed length L in {15, 30, 200}: truncate if longer, edge-pad (repeat last
   value) if shorter. Use NON-overlapping windows.
5. Split by RECORD (patient-wise), not by beat, e.g. the de Chazal DS1/DS2 split or a
   configurable seeded record split, to avoid leakage. Save train/val/test .npz files.
6. A quick sanity script that plots a few beats per class and an RR series.

Add unit tests (pytest) for: output shapes, class mapping, RR in plausible range 0.2-3 s
before normalization, and no record appears in two splits.
```

---

## PROMPT 3: BMM models (U-Net WGAN-GP)

```
Implement physiodyn/models/bmm.py following the paper (Section 4).

Generator G(z, y):
- z ~ N(0, I) of dim 128, class embedding e_y = Embedding(K=5, 64).
- h0 = concat[z, e_y] -> MLP -> reshape to [B, C, T0].
- 1D U-Net encoder-decoder with skip connections (channel-wise concat),
  h^(l+1)=Enc_l(h^(l)), hhat^(l)=Dec_l(hhat^(l+1)) concat h^(l).
- Final Conv1x1 -> output [B, 1, 200]. Choose depth/channels so the output length is exactly 200
  (use upsampling + padding/cropping as needed). Make it configurable.

Critic D(x, y) with projection conditioning:
- Feature extractor phi(x) = f_theta(x) in R^d (1D conv stack, spectral norm NOT used;
  use no BatchNorm in the critic because of the gradient penalty; use LayerNorm/none).
- D(x,y) = w^T phi(x) + <phi(x), V(y)>.

Include a shape test and a parameter-count printout.
```

---

## PROMPT 4: BMM training (WGAN-GP) with fiducial-aware losses

```
Implement physiodyn/train/train_bmm.py.

Losses (paper Eq. 12-15):
- Critic: L_D = E[D(G(z,y),y)] - E[D(x,y)] + lambda_gp * GP
  (the paper writes the sign the other way; implement the standard WGAN-GP so that
  the critic maximizes D(real)-D(fake) and document this in a comment).
  GP uses x_hat = eps*x + (1-eps)*x_fake, eps~U(0,1), penalty (||grad||_2 - 1)^2.
- Generator: L_G = -E[D(G(z,y),y)] + lambda_rec * ||G(z,y) - x||_1, with lambda_rec = 10.
  Note: the L1 term needs a paired target x. Pair each generated sample with a real beat
  of the same class from the batch and document this choice clearly in the code.

Training:
- Adam(lr=1e-4, betas=(0.0, 0.9)), n_critic=5, batch size 64, 20 epochs, class-conditional
  sampling from the real data, optional class-balanced sampler (config flag) since
  Q has only 33 beats.
- Log critic loss, generator loss, and gradient penalty per epoch to CSV. Save a checkpoint
  each epoch and the final one. Seeds loop: train for seeds in the config list and save
  each run in outputs/checkpoints/bmm/seed_{s}/.
- Then generate 25,000 synthetic beats (class proportions matching the training set, or
  configurable) and save to .npz.

Also plot the loss curves (critic and generator vs epoch).
```

---

## PROMPT 5: BMM metrics and Figure 2 / Table 3

```
Implement physiodyn/metrics/beat_metrics.py and scripts/eval_bmm.py.

Metrics comparing real vs generated beats (document exactly how pairing is done):
- RMSE, MSE, MAE, PRD (%) = 100*sqrt(sum((x-xh)^2)/sum(x^2)) between each generated beat
  and its nearest real beat of the same class (nearest by Euclidean distance), averaged.
- DTW (use fastdtw or tslearn) between generated and nearest real beats.
- Frechet Distance: fit Gaussians on feature embeddings (use a small 1D-CNN classifier
  trained on real beats as the feature extractor, penultimate layer) and compute FD.
- MMD with an RBF kernel (median-heuristic bandwidth) on flattened beats or embeddings.

Do all of it per seed and report mean +/- std over seeds in a DataFrame; save to
outputs/tables/table3_ours.csv plus a LaTeX version.

Figures:
- Figure 2 (right panel in the paper): mean +/- std band of real vs fake beats overlaid
  (real in blue, fake in red, shaded std).
- Figure 2 (left panel): one generated example per class (N, S, V, F, Q) in a 5-panel layout.

Do not hard-code baseline numbers from other papers as my results. Put the published baselines
(BiLSTM-CNN GAN, 1CNN-BiLSTM GAN, 4CNN GAN, MC-WGAN, BC-GAN) in a separate
configs/baselines_table3.yaml with a "copied from cited papers" note, and merge them only
when building the final comparison table.
```

---

## PROMPT 6: IDM model (GRU-Neural ODE)

```
Implement physiodyn/models/idm.py per Algorithm 1 in the paper.

- GRU encoder over x_{1:T} (input dim 1 RR value, optional covariates concatenated: sleep/wake,
  respiration, activity, subject-ID embedding; keep covariates optional and off by default
  because MIT-BIH does not provide them).
- h_T -> z0 = W_e h_T (linear).
- Latent dynamics: dz/dt = -alpha*z + MLP_theta(z), alpha = 0.1 (contractive), MLP with 64
  hidden units and Tanh. Integrate with torchdiffeq.odeint (method='rk4', rtol/atol from config)
  over a time grid of T points (fixed sequence length). Use odeint_adjoint as a config option.
- Decoder: GRU over the latent trajectory z_{1:T} -> Linear -> xhat_{1:T}.
- Add Gaussian noise injection sigma=0.01 to the encoder input during training.
- A generate(n, T, temperature) method that samples z0 ~ N(0, I) (or from a fitted prior on
  the encoded z0 values) and decodes an RR sequence of any horizon T by changing the time grid.
  Implement the prior fit as a small Gaussian over encoded z0 on the training set.

Include a shape test, and a test that the model can overfit a single batch to near zero loss.
```

---

## PROMPT 7: IDM losses and curriculum training

```
Implement physiodyn/losses/idm_losses.py and physiodyn/train/train_idm.py.

Losses (paper Eq. 16-20):
- L_recon = mean |x - xhat|  (L1)
- L_temp = mean over t of || (x_{t+1}-x_t) - (xhat_{t+1}-xhat_t) ||^2
- L_latent = mean over t of ||z(t)||_2^2
- L_freq = mean over k of ( |FFT(x)_k| - |FFT(xhat)_k| )^2 (use torch.fft.rfft)
- Total: L = L_recon + lam1*L_temp + lam2*L_latent + lam3*L_freq,
  with lam1 = 1.0, lam2 = 1e-4, lam3 = 0.1 (from config).

Training:
- Adam, batch size 8, grad clip 1.0.
- Curriculum learning: train sequentially on T = 15, then 30, then 200. Warm-start each stage
  from the previous stage's weights, with per-stage epochs/learning-rate/hidden-dim in
  the config (hidden-dim changes mean you need to keep dims fixed across stages, or document how weights
  are transferred; recommend fixing dims).
- Log each loss component separately per epoch, keep the best-validation-loss checkpoint
  per stage, and report best losses (paper reports 0.0343 / 0.1242 / 0.1364 for 15/30/200;
  I want to see what I actually get).
- Normalization: decide and document whether RR values are z-scored (the paper's Table 4
  shows negative min RR values, which implies z-score normalization was applied to RR) or in raw
  seconds. Make it a config flag, and always inverse-transform back to seconds before
  computing HRV metrics.
```

---

## PROMPT 8: HRV metrics, Table 4, Figures 3-5

```
Implement physiodyn/metrics/hrv.py and scripts/eval_idm.py.

HRV metrics on RR sequences in SECONDS (state the units in the output):
Mean RR, STD, Min, Max, RMSSD, SDNN, pNN50 (% of successive differences > 50 ms),
Sample Entropy (m=2, r=0.2*std), Approximate Entropy (m=2, r=0.2*std),
Poincare SD1 = std(diff)/sqrt(2), SD2 = sqrt(2*SDNN^2 - SD1^2).

Evaluation:
- For each of T in {15, 30, 200}: generate N sequences (config, e.g. 1000), compute metrics
  per sequence, and report mean +/- std across sequences and seeds. Compute the same metrics for
  real windows of the same length.
- Output Table 4 as CSV + LaTeX: columns Gen 15 RR, Gen 30 RR, Gen 200 RR, Original RR.
- Also report reconstruction-mode results (encode real -> decode) vs generation-mode results
  (sample z0) separately, because they answer different questions.
- Statistical comparison: KS test and Wasserstein distance between real and generated
  distributions for each HRV metric.

Figures:
- Fig 3: grouped bar chart of Mean RR, SDNN, RMSSD, SD1, SD2, ApEn, real vs fake, for each of the three lengths.
- Fig 4 (length 15): RR density (KDE) real vs fake, violin plot, and KDE of per-sequence STD.
- Fig 5: overlays of real vs reconstructed sequences (top-4 for length 15, top-3 for 30,
  top-1 for 200) with the similarity score shown in the legend (define the score, e.g. 1 - normalized MAE,
  and document it).

Flag explicitly in the printout if the length-200 generation has lost variability
(e.g. generated SDNN much lower than real SDNN).
```

---

## PROMPT 9: Composer

```
Implement physiodyn/models/composer.py per Section 6 (Eq. 21-22).

Inputs: RR sequence (seconds), a list of beat waveforms from the BMM (200 samples at 250 Hz),
and fs=250.

Steps:
1. Compute R-peak times t_i = cumsum(RR). Allocate an output buffer of length ceil(t_N * fs) + margin.
2. W(b_i, RR_i): time-warp each 200-sample beat so its length is proportional to the RR
   interval (use scipy.signal.resample or linear interpolation; keep the R-peak at the beat
   center, so warp the pre-R and post-R parts separately so the QRS width stays physiological).
3. S: place beats with crossfade stitching (config length, raised-cosine window) so there are no
   discontinuities.
4. Modulation (Eq. 22): x <- x*(1 + Ar*sin(2*pi*fr*t)) + Ab*sin(2*pi*fb*t) + eps(t),
   with random phases, and eps = Gaussian muscle noise. Add optional sparse spike artifacts
   (the paper's figure 7 mentions spikes).
5. Final bandpass 0.5-40 Hz to produce the "cleaned" version and return both noisy and cleaned.
6. Beat selection: sample classes from a configurable distribution (default mostly N with a small
   fraction of S/V), draw z ~ N(0,I) per beat from the trained BMM generator.

Provide compose(rr_seconds, generator, class_probs, seed) -> dict(noisy, clean, r_peaks, classes).
Test: R-peak times in the output match cumsum(RR) within 1 sample, and the output has no NaN.
```

---

## PROMPT 10: Composer evaluation, Figures 6-7

```
Implement scripts/eval_composer.py.

1. Generate RR sequences of length 15, 30, 200 with the IDM, and compose full ECGs
   (durations come out of the RR sums, roughly 16 s, 31 s, 122 s in the paper; report the actual durations).
2. Figure 7: for each duration, a noisy and cleaned ECG panel (amplitude vs time in seconds).
3. Figure 6: bar charts comparing synthetic ECG vs real ECG on Heart Rate (bpm), RR interval (s),
   QRS duration (s), signal std dev, plus a radar plot. Compute real reference stats by
   aggregating all MIT-BIH records (as in the paper) with clear definitions: heart rate from
   R-peaks, QRS duration with a simple detector (e.g. neurokit2 delineation) and document the method.
4. Run an R-peak detector (e.g. Pan-Tompkins / neurokit2) on the synthetic signal and compare detected
   R-peak times to the intended ones, and report the detection F1 and the HRV recovered from the
   synthetic ECG vs the HRV of the input RR series. This closes the loop for the paper's HRV
   reconstruction claim.
5. Save a metrics CSV.
```

---

## PROMPT 11: End-to-end runner and reproducibility

```
Create scripts/run_all.py with stages: download -> preprocess -> train_bmm -> eval_bmm ->
train_idm -> eval_idm -> compose -> eval_composer -> make_tables, each runnable on its own via
CLI flags (--stage, --seeds, --config-dir, --quick). Add a --quick mode (tiny data, 1 epoch)
for smoke testing on CPU. Cache intermediate results and skip finished stages.
Write a final README.md with install steps, the order of commands, expected runtime, GPU memory notes,
and a mapping from each paper table/figure to the script that produces it.
Add a results/RESULTS.md auto-generated summary listing the actual numbers produced.
```

---

## PROMPT 12: Review and debugging (use as needed)

```
Review the whole repository against the PhysioDyn paper. Check: tensor shapes, train/eval mode,
no leakage between train/test records, WGAN-GP sign conventions, gradient penalty computed
w.r.t. interpolated samples with create_graph=True, seeds set, units (seconds vs z-scored) for RR,
and that all metrics are computed on held-out data. List bugs by severity, then fix them.
Run pytest and the --quick pipeline and report what fails.
```

---

## Notes: issues in the paper that will affect your code and results

1. **Sampling rate:** the paper says everything is resampled to 250 Hz, but RR uses fs = 360 Hz. RR in seconds is unaffected by resampling if you convert correctly; just be consistent (Prompt 2).
2. **Negative RR in Table 4** (Min -1.07, etc.): RR intervals cannot be negative, so the generated values were evidently in z-scored units, and the "Original RR" column is in different units (mean 0.80 s, max 4.41). Compare HRV metrics only in seconds (Prompt 7-8) or reviewers will catch this.
3. **Gen 200 collapses:** mean RR 0.47 s vs 0.80 real, sample entropy 0.55 vs 5.72. The paper's own Fig 5C shows loss of variability. Expect to have to fix this (more epochs, scheduled sampling, a latent-noise prior, or a stronger frequency loss) rather than just reproduce it.
4. **Table 3:** your FD (0.608) and MMD (0.0497) are worse than several baselines, and PRD is not the best, yet the whole column is bolded. Report honestly; baselines also use different data splits and definitions, so they are not strictly comparable.
5. **lambda_spike** appears in Table 2 but no spike loss exists in the equations. Prompt 1 maps it to the temporal-difference loss; confirm what you actually meant.
6. **Fig 2 caption** says "left figure" for the mean +/- std plot, but it is the right panel. Batch size is 64 for the GAN and 8 for the IDM, so keep them separate.
7. **L1 reconstruction loss in the GAN generator** needs paired targets, which a noise-driven generator does not have by default (Prompt 4 handles this explicitly).
8. **Results come from running the code**, so do not copy numbers from the paper into your results tables.
