# pet.jl
Pattern-Exploiting Training in Julia. A replication of "It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners"

## Setting up

```bash
./download-data.sh
./download-albert.sh
```

## Replicating experiments

Scripts for the PET and iPET versions of each dataset can be found under src/scripts.

To run an experiment, for example boolq with PET, simply do:

```bash
cd src && julia scripts/boolq_pet.jl
```


## Running Baselines

### All baselines
```bash
cd src && julia baselines.jl
```

### Specific baseline
Here are the possible flags:
```bash
cd src && julia baselines.jl --dataset BoolQ/CB/COPA/RTE/WiC/WSC/all --method Random/MostCommon/all
```

For more details, you can always do:
```bash
julia src/baselines.jl --help
```

# ALBERT

## Requirements

- PyCall

Python Dependencies

- transformers

## Example
```bash
cd src && julia albert_example.jl
```
Input: The capital of France is [MASK].

Output: the capital of france is paris .

To use your own input, uncomment out the lines in src/albert_example.jl


## Running tests

### All ALBERT related tests
```bash
cd src/albert && julia albert_tests.jl
```
