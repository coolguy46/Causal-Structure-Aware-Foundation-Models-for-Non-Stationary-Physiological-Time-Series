The Idea: Causal Structure-Aware Foundation Models for Non-Stationary Physiological Time Series
One-line pitch: A foundation model that learns why physiological signals change — not just that they change — by jointly modeling causal graph structure and temporal dynamics, enabling zero-shot generalization across devices, subjects, and conditions.

Why this gap exists
Foundation models for time series remain largely unexplored for domain-specific biomedical signals like EEG, with current work showing generalist models surprisingly outperforming EEG-specific ones ladybugdb — but nobody has explained why, and no one has exploited it. The key open problems flagged by NeurIPS 2025 workshop organizers were agents, interpretability, and context-informed predictions ladybugdb, none of which are solved for physiological signals. The NeurIPS 2025 paper list shows heavy coverage of EEG autoencoders, time series forecasting, and causal discovery — but each in isolation. The combination is open.

The core technical contribution
Here's the architecture and what makes each piece non-trivial:
Problem formulation (novel): Treat a multi-channel biosignal (EEG, ECG, EMG, etc.) not as a flat multivariate time series, but as observations from an underlying latent causal dynamical system — where the causal graph changes as physiological state changes (sleep → wake, healthy → seizure, rest → exercise). The model must simultaneously infer the latent graph and forecast/classify, and the graph must be the mechanism of generalization, not a side output.
Architecture (novel combination):

A spectral tokenizer that converts raw signals into frequency-band tokens — explicitly separating physiological rhythms (delta, alpha, beta bands) before encoding, giving the model physics-aligned inductive bias
A causal graph inference module that runs per-window, producing a sparse directed adjacency matrix over channels — grounded in Granger-style conditional independence but learned end-to-end
A graph-conditioned transformer whose attention mask is the inferred causal graph, not a learned dense attention — this is the theoretical crux: attention is constrained by inferred causality, so the model cannot overfit spurious correlations
A non-stationarity adapter (lightweight, per-subject) that shifts the causal graph prior without retraining the backbone

Loss function (novel): A joint objective with three terms: (1) reconstruction loss on the spectral tokens, (2) a causal consistency loss penalizing interventional inconsistencies in the inferred graph across windows, and (3) a downstream task loss. The causal consistency term is the theoretical novelty — it's derived from the do-calculus and enforces that the inferred graph predicts correctly under simulated interventions.
The theoretical claim (NeurIPS-grade): You prove that models trained with causally-constrained attention generalize across distribution shifts (new subjects, new devices) with provably lower worst-case error than models with unconstrained attention, under mild assumptions about the data-generating process. This connects to the invariant risk minimization literature but specialized to time-varying graphs.

Why this is new
Looking across the NeurIPS 2025 paper list, papers like THD-BAR (topology-hierarchical brain autoregressive modeling) and BrainOmni (EEG/MEG foundation model) exist, but none combine causal graph inference + spectral tokenization + theoretical generalization guarantees. The causal discovery literature at NeurIPS 2025 (the "Identifying Macro Causal Effects" and "Near-Optimal Experiment Design" papers) is entirely separate from the biosignal foundation model work. This paper explicitly bridges them.


What the paper actually proves / shows
Theorem (informal): Under the assumption that the true data-generating process is a time-varying structural causal model, a graph-conditioned transformer trained with causal consistency loss achieves worst-case generalization error bounded by O(√(d/n)) where d is the causal graph degree — versus O(√(D/n)) for unconstrained attention where D ≫ d. This is the NeurIPS-grade theoretical contribution.
Empirical claims (concrete and falsifiable):

Zero-shot transfer across subjects on sleep staging (Sleep-EDF dataset) beats SOTA by 5-10% F1
Zero-shot transfer across EEG montage configurations (23-ch → 2-ch) via adapter without full retraining
On seizure detection, the inferred causal graphs match clinical ground truth connectivity better than correlation-based baselines, validated via interventional experiments (do-calculus node knockouts match known neurophysiology)
Within-modality transfer (EEG sleep→EEG seizure, ECG task A→ECG task B) shows causal structure improves adaptation; cross-modality transfer (EEG→ECG) is not claimed, as causal graphs differ fundamentally across organs
Ablation shows the causal consistency loss alone accounts for most of the generalization gain


Why reviewers would accept this
It hits every NeurIPS criterion simultaneously: a genuine theoretical contribution (the generalization bound), a methodological novelty (causally-constrained attention), a compelling application domain with reproducible benchmarks (Sleep-EDF, CHB-MIT, PTB-XL), and it speaks to multiple hot threads — foundation models, causal ML, and non-stationary learning — without being derivative of any single one.
The one thing you'd need to be careful about: clearly differentiating from BrainOmni and the THD-BAR paper in your related work. The key differentiator is the causal graph as the mechanism of generalization, not as an auxiliary output. No existing paper makes that specific claim with a proof.
Would you like help formalizing the theorem, designing the ablation study, or drafting the related work positioning?