# Data

The files in this directory were obtained from https://github.com/barmoral/evaluator-hmix-dens
and modified to create tests inspired by [Issue 575](https://github.com/openforcefield/openff-evaluator/issues/575).

As @barmoral's calculations were done with multiple properties, the results were *renamed* and
the working paths were edited using the following example code below. The files were then manually renamed from 6422* to 6421.

```python
import pathlib
import json
import re

directory = pathlib.Path("6422_conditional_group_component_1/6422_production_simulation_component_1")
clip = "working-data/SimulationLayer/e8c168c73b7b4db0b5d49131e4fc571f/"
for filename in directory.glob("6422*.json"):
    with filename.open("r") as f:
        contents = f.read()
    new_contents = re.sub(r'6422\|', r'6421|', contents)
    new_contents = re.sub(r'6422_', '6421_', new_contents)
    new_contents = re.sub(r'6420\|', r'6421|', new_contents)
    new_contents = re.sub(r'6420_', '6421_', new_contents)
    new_contents = re.sub(clip, '', new_contents)
    with filename.open("w") as f:
        f.write(new_contents)
```

