# Module Refactoring Guide - Completion Steps

## Overview
The codebase has been partially refactored into thematic modules. This guide completes the refactoring by moving the remaining worker files and updating imports.

## Step 1: Move Worker Files to Training Package

### 1a. Create Dynamic Worker (`training/dynamic/worker.py`)
**Source:** `src/worker_p2p.py`

**Copy the file content and replace these imports:**
```python
# Old imports (REMOVE):
from src.robust_aggregators import RobustAggregator
from src.byzWorker import ByzantineWorker
from src.byz_attacks import ByzantineAttack
from . import models, misc

# New imports (ADD):
from robustness.aggregators import RobustAggregator
from training.byzantine import ByzantineWorker
from robustness.attacks import ByzantineAttack
from data import models
from src.utils.tensor_utils import flatten, unflatten
from src.utils.math_utils import clip_vector
```

**Keep:** All method implementations unchanged

### 1b. Create Fixed-Graph Worker (`training/fixed_graph/worker.py`)
**Source:** `src/fx_worker_p2p.py`

**Copy the file content and replace these imports:**
```python
# Old imports (REMOVE):
from src.robust_aggregators import RobustAggregator
from src.byzWorker import ByzantineWorker, DecByzantineWorker
from src.byz_attacks import ByzantineAttack
from src.robust_summations import cs_plus, gts, cs_he
from . import models, misc
from utils.gossip import LaplacianGossipMatrix

# New imports (ADD):
from robustness.aggregators import RobustAggregator
from training.byzantine import ByzantineWorker, DecByzantineWorker
from robustness.attacks import ByzantineAttack
from robustness.summations import cs_plus, gts, cs_he
from data import models
from src.utils.tensor_utils import flatten, unflatten
from src.utils.math_utils import clip_vector
from topology.gossip import LaplacianGossipMatrix
```

**Keep:** All method implementations unchanged

## Step 2: Move Analysis File

### 2a. Complete `analysis/study.py`
**Source:** `src/study.py`

**Action:** Copy entire file content - it needs minimal import changes
- The file imports `tools` and other system utilities which remain the same
- Just verify the file works with the rest of the refactored code

## Step 3: Clean Up src/misc.py

The current `src/misc.py` should be refactored to keep only:
- `check_make_dir()` - directory creation
- `process_commandline()` - CLI argument parsing
- `print_conf()` - configuration printing
- `topk()`, `sigmoid()` - accuracy metrics
- `store_result()`, `make_result_file()` - result file I/O

**Remove from src/misc.py** (now in specialized modules):
- `get_default_root()` → use `data.dataset_utils.get_default_root()`
- `draw_indices()` → use `data.dataset_utils.draw_indices()`
- `flatten()`, `unflatten()` → use `src.utils.tensor_utils`
- `clip_vector()`, `line_maximize()`, etc. → use `src.utils.math_utils`
- All smoothed_weiszfeld, compute_distances, etc. → use `src.utils.math_utils`

## Step 4: Update Training Scripts

### 4a. Update `train_p2p.py`
Replace imports:
```python
# Old
from src.worker_p2p import P2PWorker
from src import dataset, misc
from src import fxgraph

# New
from training.dynamic.worker import P2PWorker
from data import Dataset, make_train_test_datasets
from data import models
from topology import fxgraph
from src import misc  # for CLI utilities
```

### 4b. Update `fx_train_p2p.py`
Replace imports:
```python
# Old
from src.fx_worker_p2p import P2PWorker
from src import dataset, misc
from src import fxgraph

# New
from training.fixed_graph.worker import P2PWorker
from data import Dataset, make_train_test_datasets  
from data import models
from topology import fxgraph
from src import misc  # for CLI utilities
```

### 4c. Update Launcher Scripts
In `run.py`, `mnist_run.py`, `fx_mnist_run.py`, `fx_cifar10_run.py`:
```python
# Old
from src import dataset

# New
from data import Dataset, make_train_test_datasets
```

### 4d. Update `experiments/common.py`
Replace imports:
```python
# Old
from src import misc, study

# New
from src import misc  # for CLI utilities  
from analysis import study
from src.utils.math_utils import *  # if needed for plotting
```

### 4e. Update `comp_plot.py`
Replace imports:
```python
# Old
from src import study

# New
from analysis import study
```

## Step 5: Verify Structure

Run this command to check the new structure:
```bash
tree -d -L 2 /home/ale/Projects/RPEL-BF2D/
```

Expected output:
```
├── analysis/
│   ├── __init__.py
│   └── study.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── dataset_utils.py
│   └── models.py
├── robustness/
│   ├── __init__.py
│   ├── aggregators.py
│   ├── attacks.py
│   └── summations.py
├── topology/
│   ├── __init__.py
│   ├── fxgraph.py
│   ├── gossip.py
│   └── graph.py
├── training/
│   ├── __init__.py
│   ├── byzantine.py
│   ├── dynamic/
│   │   ├── __init__.py
│   │   └── worker.py
│   └── fixed_graph/
│       ├── __init__.py
│       └── worker.py
└── src/
    ├── utils/
    │   ├── __init__.py
    │   ├── math_utils.py
    │   └── tensor_utils.py
    └── misc.py
```

## Step 6: Test Imports

After moving files, validate imports by running:
```bash
cd /home/ale/Projects/RPEL-BF2D
python -c "from training.dynamic.worker import P2PWorker; print('✓ Dynamic worker imports OK')"
python -c "from training.fixed_graph.worker import P2PWorker; print('✓ Fixed-graph worker imports OK')"
python -c "from robustness.aggregators import RobustAggregator; print('✓ Robustness imports OK')"
python -c "from data import Dataset, make_train_test_datasets; print('✓ Data imports OK')"
python -c "from topology import create_graph; print('✓ Topology imports OK')"
```

## Step 7: Run Validation

Execute the validation command to check for errors:
```bash
# This will show any import or syntax errors in the refactored code
python -m py_compile training/dynamic/worker.py
python -m py_compile training/fixed_graph/worker.py
python -m py_compile analysis/study.py
```

## Notes

- All source code logic remains **unchanged**, only organization improved
- The `src/` directory is reduced to utilities and configuration
- Top-level import statements make the package structure clear
- Backward compatibility can be maintained by adding compatibility imports if needed
- The experiments/ package remains as the orchestration layer

## Common Issues & Solutions

**Issue:** `ModuleNotFoundError: No module named 'robustness'`
**Solution:** Ensure you're running from the project root directory

**Issue:** Circular imports
**Solution:** Check that training/ doesn't import from src/worker_p2p (should import from training.dynamic.worker instead)

**Issue:** Missing utility functions
**Solution:** Check that src/utils/math_utils.py has all functions (clip_vector, flatten, etc.) imported in code
