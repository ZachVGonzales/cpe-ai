# cpe-ai
Sustainable LLMs 

## Creating Env

### Create Conda Env (Optional)

```bash
conda create -n lean4 python=3.11 -y
conda activate lean4
```

### Install LEAN4

```bash
curl -L https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh
```

* NOTE: restart terminal after this

### Confirm Installation

```bash
lean --version
lake --version
```

## Testing Lean Code

The following lists the steps necessary to test LEAN code (compilation & run)

1. generate a lean code file (see data/OPC/example.lean for example)

2. using the provided test script run the following

```bash
# list all options:
python scripts/data/lean-check.py --help
```

```bash
# build and run:
python scripts/data/lean-check.py --file data/OPC/filename... --run
```