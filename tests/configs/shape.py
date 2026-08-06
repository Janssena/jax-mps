import jax
from jax import numpy as jnp
from jax import random

from .util import OperationTestConfig, complex_standard_normal


def make_shape_op_configs():
    # Flip and transpose ops for both real and complex inputs
    for complex in [False, True]:
        with OperationTestConfig.module_name(
            "shape-complex" if complex else "shape-real"
        ):
            yield from [
                OperationTestConfig(
                    jnp.flip,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (16,), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.fliplr,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (8, 16), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.flipud,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (8, 16), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.transpose,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (4, 8, 16), complex
                    ),
                ),
                OperationTestConfig(
                    jnp.transpose,
                    lambda key, complex=complex: complex_standard_normal(
                        key, (4, 8, 16), complex
                    ),
                    (1, 0, 2),
                    static_argnums=(1,),
                ),
            ]

    with OperationTestConfig.module_name("shape"):
        yield from [
            OperationTestConfig(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                lambda key: random.normal(key, (3, 4)),
                lambda key: random.normal(key, (5, 4)),
            ),
            OperationTestConfig(
                lambda x: jnp.reshape(x, (20,)),
                lambda key: random.normal(key, (4, 5)),
            ),
            OperationTestConfig(
                lambda x: jnp.pad(x, ((1, 1), (2, 2))),
                lambda key: random.normal(key, (3, 3)),
            ),
            # Pad with interior padding
            OperationTestConfig(
                lambda x: jax.lax.pad(x, 0.0, [(1, 1, 1), (0, 0, 2)]),
                lambda key: random.normal(key, (3, 4)),
            ),
            # Pad with negative edge padding (trimming)
            OperationTestConfig(
                lambda x: jax.lax.pad(x, 0.0, [(-1, 2, 0), (0, -1, 0)]),
                lambda key: random.normal(key, (4, 5)),
                name="pad-negative-edge",
            ),
            # broadcast_in_dim with non-ascending broadcast_dimensions: an
            # implicit transpose fused into the broadcast, as jax.lax's own
            # docstring for this function demonstrates. StableHLO permits this
            # (broadcast_dimensions need only be unique, not sorted) and JAX
            # lowers it verbatim -- reachable from plain jax.jit, no optimizer
            # needed. Regression for jax-mps issue: HandleBroadcastInDim used
            # to reshape without transposing, silently returning the input
            # with two axes swapped whenever broadcast_dimensions was unsorted.
            OperationTestConfig(
                lambda x: jax.lax.broadcast_in_dim(
                    x, (2, 3, 4), broadcast_dimensions=(1, 0)
                ),
                lambda key: random.normal(key, (3, 2)),
                name="broadcast_in_dim-nonascending-dims",
                # The forward value is correct on mps (that is what this config
                # verifies). The gradient xfails because of an upstream JAX bug,
                # not the plugin: jax._src.lax.lax._broadcast_in_dim_transpose_rule
                # reduce_sums the cotangent over the pure-broadcast axes but
                # never transposes the surviving axes back into operand order by
                # the inverse of broadcast_dimensions. For ascending dims that is
                # a no-op (so grads work); for a permutation it returns the
                # transpose of the correct gradient, so jax.grad raises
                # "Expected cotangent type f32[3,2] ... but got f32[2,3]". This
                # happens during JAX tracing before any device dispatch and
                # reproduces identically under JAX_TEST_MODE=cpu (no plugin).
                # Writing the same math as transpose + ascending broadcast
                # differentiates correctly, confirming the fused primitive's VJP
                # is the culprit. strict=True so this flips to a failure (xpass)
                # if JAX ever fixes it, prompting removal of the marker.
                # TODO(upstream): file against openxla/jax and link the issue here.
                grad_xfail="Expected cotangent type",
            ),
        ]
