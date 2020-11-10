# pet.jl
Pattern-Exploiting Training in Julia. A replication of "It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners"

## Setting up

```bash
./download-data.sh
```

## Running Baselines

### All baselines
```bash
julia baselines.jl
```

### Specific baseline
Here are the possible flags:
```bash
julia baselines.jl --dataset BoolQ/CB/COPA/MultiRC/ReCoRD/RTE/WiC/WSC/all --method Random/MostCommon/all
```

For more details, you can always do:
```bash
julia baselines.jl --help
```
