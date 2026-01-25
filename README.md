A Python port of [PiMPC](https://github.com/SOLARIS-JHU/PiMPC.jl).

## Installation
Clone this repository
```bash
git clone https://github.com/shaoanlu/PiMPC-python.git
```
And install in edit mode
```bash
pip install -e .
```

## Usage
Refer to [examples](examples/):
- `examples/afti16_example.py` (linear AFTI-16 aircraft model)
- `examples/unicycle_nmpc_example.py` (nonlinear unicycle with time-varying linearization)

![](assets/example_afti16.png)

![](assets/example_unicycle.png)

## Testing
```bash
pytest tests/test_pimpc.py
# ================================================= test session starts # ==================================================
# platform linux -- Python 3.10.14, pytest-9.0.2, pluggy-1.6.0
# rootdir: /mnt/c/Users/shaoa/Documents/GitHub/PiMPC-python
# configfile: pyproject.toml
# collected 6 items
# 
# tests/test_pimpc.py ......                                                                                       # [100%]
# 
# ================================================== 6 passed in 0.43s # ===================================================
```
