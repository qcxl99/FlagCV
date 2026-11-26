"""OpenCV CUDA baseline recipe for multiply."""

if __package__:
    from .__recipe import recipe
else:
    from __recipe import recipe

from workloads import multiply as workload

_recipe = recipe(workload)
OP_NAME = _recipe.OP_NAME
DTYPES = _recipe.DTYPES
IS_INPLACE = _recipe.IS_INPLACE
native = _recipe.native
grid = _recipe.grid
build_inputs = _recipe.build_inputs
key_shape = _recipe.key_shape
config = _recipe.config
