# SpheriC
Python package for spherical collapse calculations including monolithic and self-similar collapse models.  
   
## How to Use These Files
### Step 1: Download the code
download with 
`git clone ....`
or download zip file and extract its content.

### Step 2: Setup path
Assuming you cloned or extracted the package to SPHERIC_DIR (e.g., '/home/user/SpheriC'), you need to add it to your python path.

In `bash` (replace `$SPHERIC_DIR` with your actual path)
```bash
export PYTHONPATH=$SPHERIC_DIR:$PYTHONPATH
```

Or only add it at runtime in the beginning of your python code:
```python
import sys
# Replace with the actual path to the SpheriC directory
SPHERIC_DIR = '/home/user/SpheriC' 
sys.path.append(SPHERIC_DIR)
```

### Step 3: Use the Code 
Once the path has been setup, you can use it:

```python
# spherical collapse solver
from SpheriC import SCSolver
sc = SCSolver(OmegaM=0.1)
sc.DeltaVirial(a=0.5)

# self-similar collapse solver 
from Spheric import ReducedOrbit
orbit = ReducedOrbit(epsilon=0.3)
orbit.solve()
```
For more detailed examples on the usage, see the enclosed `examples/Example.ipynb` jupyter notebook.

## References:
If you make use of this code, please cite the following paper:

- Jiaxin Han, The many boundaries of the stratified dark matter halo, RAA, 2026 (arxiv:coming_soon)

## Contact:
Jiaxin Han (jiaxin.han #at# sjtu.edu.cn)
