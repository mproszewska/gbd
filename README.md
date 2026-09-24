# Graph Bridge Diffusion

Repository is based on [DeFoG](https://github.com/manuelmlmadeira/DeFoG/).

Discrete graph diffusion/flow matching for structure-constrained molecular design (scaffold decoration, motif extension, linker generation, superstructure generation, scaffold morphing), evaluated on [MolGenBench](https://github.com/CAODH/MolGenBench/) and [SAFE](https://github.com/datamol-io/safe).

## Setup

1. Install [DeFoG](https://github.com/manuelmlmadeira/DeFoG/) and [MolGenBench](https://github.com/CAODH/MolGenBench/) following their respective instructions.
2. Install the required libraries:
```bash
pip install torch==2.4.0 torch-geometric==2.3.1 rdkit==2023.03.2
```

## Training

### Bridge
```bash
python main.py +experiment=moses_scaffold dataset=moses_scaffold \
    model.keep_scaffold_fixed=True model.skip_scaffold_prob=0.2 model.sample_fragments_prob=0.5
```

### No-bridge (ablation)
```bash
python main.py +experiment=moses_scaffold dataset=moses_scaffold \
    model.keep_scaffold_fixed=False model.skip_scaffold_prob=1.0 model.sample_fragments_prob=0.0
```

## Evaluation

```bash
python main.py +experiment=moses_scaffold dataset=moses_scaffold \
    hydra.run.dir="${CHECKPOINT%%/checkpoints*}/" \
    general.test_only=${CHECKPOINT} \
    sample.eta=10 sample.omega=0 \
    sample.time_distortion=polydec \
    sample.sample_steps=50 sample.eval_task=molgenbench
```

- `sample.eval_task`: set to `molgenbench` for MolGenBench scaffold decoration, or to `safe` for SAFE evaluation.

- `${CHECKPOINT}` should point to a trained model checkpoint from the training step above (bridge or no-bridge).

