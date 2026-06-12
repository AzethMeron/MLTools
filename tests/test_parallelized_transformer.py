"""ParallelizedTransformer: ordering, error propagation, reuse after failure.

Transforms must be module-level functions so the 'spawn' context can pickle
them into worker processes.
"""
import pytest

from MLTools.DataProcessing import ParallelizedTransformer

pytestmark = pytest.mark.slow


def double_it(x):
    return x * 2


def fail_on_negative(x):
    if x < 0:
        raise ValueError(f"negative input: {x}")
    return x + 1


@pytest.fixture(scope="module")
def doubler():
    return ParallelizedTransformer(double_it, num_workers=2)


@pytest.fixture(scope="module")
def failer():
    return ParallelizedTransformer(fail_on_negative, num_workers=2)


def test_results_in_input_order(doubler):
    data = list(range(50))
    assert doubler.run(data) == [x * 2 for x in data]


def test_empty_input(doubler):
    assert doubler.run([]) == []


def test_single_item(doubler):
    assert doubler.run([21]) == [42]


def test_reusable_across_runs(doubler):
    assert doubler.run([1, 2]) == [2, 4]
    assert doubler.run([3]) == [6]


def test_worker_exception_propagates(failer):
    """A raising transform used to kill the worker and deadlock run().
    Now it must raise in the parent with the original traceback."""
    with pytest.raises(RuntimeError, match="negative input"):
        failer.run([1, -5, 3])


def test_usable_after_exception(failer):
    """Workers must survive a failed item: subsequent runs still work."""
    with pytest.raises(RuntimeError):
        failer.run([-1])
    assert failer.run([10, 20]) == [11, 21]


def test_invalid_worker_count():
    with pytest.raises(ValueError):
        ParallelizedTransformer(double_it, num_workers=0)
