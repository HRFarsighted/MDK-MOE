# Code-to-paper mapping

| Manuscript component | Supplied implementation |
| --- | --- |
| Embedding and normalization | `models/MDK_MoE.py: PatchEmbed`, `layers/Embed.py: PositionalEmbedding`, `RevIN/RevIN.py` |
| FFT decomposition | `FrequencyDecomposer.batch_fft` |
| Moving-average decomposition | `moving_avg`, instantiated with `configs.moving_avg` |
| Wavelet decomposition | `WaveletDecomposer`, a Morlet-kernel convolution instantiated with `configs.level` |
| Cross-view MMI | `FrequencyContrastiveLoss`; returned as `loss_mi` by `Model.forward` |
| MoE gating | `TopKGating` |
| Kalman fusion | `StableRecursiveKalmanFusion` |
| Projection | `Linear_x`, residual addition, dropout, `MLP1`, RevIN denormalization |
| Training objective | `Exp_Main._RobustPredLoss` and `Exp_Main.train` |
| Data and metrics | `data_provider/` and `utils/metrics.py` |

## Points to reconcile in the camera-ready materials

1. **`level` and `moving_avg`:** Appendix C.4 describes `level` as the MA window length. The supplied model passes `level` to the wavelet kernel; the MA window uses `moving_avg`.
2. **FFT ratio:** Appendix C.4 describes `threshold_ratio` in terms of high-frequency components. The code uses it to select symmetric low-frequency FFT bins and defines high frequency as the residual.
3. **Prediction-loss scaling:** The manuscript writes Smooth L1 + MSE, while `_RobustPredLoss` uses `0.5 * SmoothL1 + 0.5 * MSE`. This changes the relative scaling of auxiliary losses when their weights are fixed.
4. **Top-k gating:** `TopKGating.decompostion_tp` transforms logits differently above and below a rank threshold, then softmaxes all six logits. It does not zero out nonselected experts; describe this carefully when discussing sparse expert selection.
5. **MMI weight naming:** The CLI's `covar_loss` multiplies the returned MMI/contrastive objective, despite its legacy name.

These notes record source-level differences. They are not empirical conclusions about accuracy or changes to the model. Reconcile the exact implementation used for the reported experiments before updating either the paper or public claims.
