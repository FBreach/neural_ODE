# Archived functions

This archive records 45 function definitions retained as comments in the source.
The original function and class bodies, including decorators, remain prefixed
with `# `. They cannot run or be imported. Methods removed from classes are no
longer available through those classes; `DiffeqSolverParallel` keeps its own
active `forward` method.

The retained workflow uses ODE-RNN or RNN encoding, parallel Euler/RK2/RK4
integration, reconstruction pretraining, GMM fitting, mixture-prior fine-tuning,
prediction, evaluation, and plotting. Exported API functions remain available.
The archived ADNI and post-hoc GMM model names now raise the existing
`ValueError: Unknown LatentODE model_name: ...` validation error. External code
or saved models that depend on archived private classes require restoration.

Each source link points to the commented function definition. Restoration must
include the related class, imports, factory configuration, and callers described
below; uncommenting an isolated function may not be sufficient.

## Unused clustering helpers (3 functions)

No repository workflow calls these helpers. `train_fit_gmm` and `VAEClustering.init_prior_from_gmm` handle mixture fitting; `plot_latent_space` handles latent visualization. Restoration requires their existing PyTorch/NumPy dependencies and the commented `KMeans` import for `kmeans`.

- [`neural_ode._core.vaeclustering.sample_latent_space`](src/neural_ode/_core/vaeclustering.py#L745)
- [`neural_ode._core.vaeclustering.kmeans`](src/neural_ode/_core/vaeclustering.py#L786)
- [`neural_ode._core.vaeclustering.init_gmm_from_kmeans`](src/neural_ode/_core/vaeclustering.py#L813)

## Unused deterministic reconstruction (3 functions)

`compute_all_losses_pretrain_auto` has no repository callers and is the only caller of the separate deterministic reconstruction methods. Prediction and evaluation already use `get_reconstruction(use_sampling=False)`. Restore the abstract declaration, concrete implementation, and loss method together if the old deterministic objective is needed.

- [`neural_ode._core.latent_ode.LatentODE.get_reconstruction_det`](src/neural_ode/_core/latent_ode.py#L105)
- [`neural_ode._core.vaeclustering.VAEClustering.get_reconstruction_det`](src/neural_ode/_core/vaeclustering.py#L370)
- [`neural_ode._core.vaeclustering.VAEClustering.compute_all_losses_pretrain_auto`](src/neural_ode/_core/vaeclustering.py#L510)

## Unused linear decoder (2 functions)

No active code imports or instantiates `encoder_decoder.Decoder`. The retained `vaeclustering.Decoder` emits both a mean and standard deviation. Restoring this class uses the existing PyTorch and utility imports.

- [`neural_ode._core.encoder_decoder.Decoder.__init__`](src/neural_ode/_core/encoder_decoder.py#L355)
- [`neural_ode._core.encoder_decoder.Decoder.forward`](src/neural_ode/_core/encoder_decoder.py#L365)

## Unused prior sampling (3 functions)

`LatentODE.sample_traj_from_prior` immediately raises `NotImplementedError`; the rest of its body is unreachable. The solver and ODE helpers belong to that unused path. Restoration requires `torchdiffeq`, its commented `odeint` import, a valid `z0_prior`, and compatible solver time grids. Simply uncommenting the code does not implement prior sampling.

- [`neural_ode._core.diffeq_solver.DiffeqSolver.sample_traj_from_prior`](src/neural_ode/_core/diffeq_solver.py#L41)
- [`neural_ode._core.latent_ode.LatentODE.sample_traj_from_prior`](src/neural_ode/_core/latent_ode.py#L185)
- [`neural_ode._core.ode_func.ODEFunc.sample_next_point_from_prior`](src/neural_ode/_core/ode_func.py#L50)

## Superseded solver method (1 function)

The factory constructs `DiffeqSolverParallel`, which overrides this `forward` method. The base constructor remains active. Restoration requires the commented `torchdiffeq` dependency and `odeint` import, and callers supplying the shared time grid expected by this method.

- [`neural_ode._core.diffeq_solver.DiffeqSolver.forward`](src/neural_ode/_core/diffeq_solver.py#L26)

## Poisson implementation (3 functions)

No active factory or repository workflow constructs `ODEFunc_w_Poisson`. Restoration requires the class, a compatible intensity network and augmented state, and deliberate factory wiring. The commented `poisson` factory option previously changed the encoder dimension without constructing this class.

- [`neural_ode._core.ode_func.ODEFunc_w_Poisson.__init__`](src/neural_ode/_core/ode_func.py#L57)
- [`neural_ode._core.ode_func.ODEFunc_w_Poisson.extract_poisson_rate`](src/neural_ode/_core/ode_func.py#L82)
- [`neural_ode._core.ode_func.ODEFunc_w_Poisson.get_ode_gradient_nn`](src/neural_ode/_core/ode_func.py#L113)

## ADNI implementation (27 functions)

No repository example, notebook, test, or workflow selects ADNI. The implementation targets `gm_score`, `CENTILOID`, `ADAS13`, and `MMSE`; its loss requires `raw_data_to_predict`, absent from standard batches. The two KL helpers below have callers only within the ADNI implementation; ordinary training retains its own KL calculations. Restoration requires the ADNI classes, feature validation and constants, imports, factory selection and options, normalization setup, and a compatible data adapter. Diagnosis supervision also needs `dx_times`, `dx_labels`, and `dx_mask`.

- [`neural_ode._core.latent_ode.LatentODEADNIMixture.__init__`](src/neural_ode/_core/latent_ode.py#L210)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.set_dx_class_weights`](src/neural_ode/_core/latent_ode.py#L262)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.set_adni_normalization`](src/neural_ode/_core/latent_ode.py#L275)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.adni_prediction_to_raw`](src/neural_ode/_core/latent_ode.py#L298)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture._reconstruct_from_latent`](src/neural_ode/_core/latent_ode.py#L308)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.get_reconstruction_det`](src/neural_ode/_core/latent_ode.py#L347)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.get_reconstruction`](src/neural_ode/_core/latent_ode.py#L367)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.mixed_reconstruction_log_prob`](src/neural_ode/_core/latent_ode.py#L388)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.forecast_error_metrics`](src/neural_ode/_core/latent_ode.py#L414)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.dx_class_log_probs`](src/neural_ode/_core/latent_ode.py#L438)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.dx_class_probs`](src/neural_ode/_core/latent_ode.py#L462)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture._zero_dx_scores`](src/neural_ode/_core/latent_ode.py#L466)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture._dx_scores`](src/neural_ode/_core/latent_ode.py#L477)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.dx_logits_at_times`](src/neural_ode/_core/latent_ode.py#L514)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.dx_loss_and_metrics`](src/neural_ode/_core/latent_ode.py#L525)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture._compute_adni_losses`](src/neural_ode/_core/latent_ode.py#L571)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.compute_all_losses`](src/neural_ode/_core/latent_ode.py#L693)
- [`neural_ode._core.latent_ode.LatentODEADNIMixture.compute_all_losses_pretrain`](src/neural_ode/_core/latent_ode.py#L714)
- [`neural_ode._core.vaeclustering.validate_adni_feature_names`](src/neural_ode/_core/vaeclustering.py#L26)
- [`neural_ode._core.vaeclustering.ADNIMixedDecoder.__init__`](src/neural_ode/_core/vaeclustering.py#L82)
- [`neural_ode._core.vaeclustering.ADNIMixedDecoder._positive_std`](src/neural_ode/_core/vaeclustering.py#L113)
- [`neural_ode._core.vaeclustering.ADNIMixedDecoder.mmse_thresholds`](src/neural_ode/_core/vaeclustering.py#L116)
- [`neural_ode._core.vaeclustering.ADNIMixedDecoder.mmse_probabilities`](src/neural_ode/_core/vaeclustering.py#L123)
- [`neural_ode._core.vaeclustering.ADNIMixedDecoder.forward`](src/neural_ode/_core/vaeclustering.py#L131)
- [`neural_ode._core.vaeclustering.ADNIMixedDecoder.feature_log_probs`](src/neural_ode/_core/vaeclustering.py#L168)
- [`neural_ode._core.vaeclustering.VAEClustering.mixture_prior_kl`](src/neural_ode/_core/vaeclustering.py#L326)
- [`neural_ode._core.vaeclustering.VAEClustering.standard_normal_kl`](src/neural_ode/_core/vaeclustering.py#L357)

## Post-hoc GMM variant (3 functions)

No repository workflow selects this variant. Both subclass overrides delegate to `compute_all_losses_standard_vae`. With a nonzero KL coefficient, it regularizes against N(0,I); a fitted GMM supplies cluster assignments without entering the training loss. Default mixture-prior fine-tuning and ordinary `train_fit_gmm` remain active. Restoration requires the subclass, its loss helper, import, model-name constant, and factory branch.

- [`neural_ode._core.latent_ode.LatentODEPostHocGMM.compute_all_losses`](src/neural_ode/_core/latent_ode.py#L742)
- [`neural_ode._core.latent_ode.LatentODEPostHocGMM.compute_all_losses_pretrain`](src/neural_ode/_core/latent_ode.py#L758)
- [`neural_ode._core.vaeclustering.VAEClustering.compute_all_losses_standard_vae`](src/neural_ode/_core/vaeclustering.py#L550)

## Related commented configuration and imports

- `create_latent_ode_model.py`: ADNI/post-hoc class imports, model-name
  constants, accepted-name entries, factory branches, and `extra_model_kwargs`.
  The active factory constructs `LatentODE` for both retained encoders.
- `create_latent_ode_model.py`: private `poisson`, `lambda_dx`,
  `dx_decoder_type`, and `feature_names` options, plus the Poisson encoder
  dimension adjustment.
- `latent_ode.py`: ADNI imports, `Results`, and the `torch.nn` and
  `torch.nn.functional` imports used only by archived code.
- `vaeclustering.py`: ADNI feature constants, `KMeans`, and the original
  `hidden_dims_list` import. The active utility import retains `create_mlp`.
- `diffeq_solver.py`: the `torchdiffeq.odeint` import.
- `pyproject.toml`: the `torchdiffeq` dependency. Restore it before enabling
  either archived integration path.

The remaining solver tolerances and base constructor are retained for
compatibility. No dataset, generated output, or checkpoint is changed.
