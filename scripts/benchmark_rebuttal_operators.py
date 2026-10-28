"""Matched steady-state diagnostics for rebuttal TODOs 5, 6(b), and 7.

Run through run_rebuttal_diagnostics.py for environmental controls and kernel logs.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
import torch.nn.functional as functional

from qpu_xla import Device
from qpu_xla.benchmark import collect_metadata
from qpu_xla.models.vision.resnet18 import ResNet18Artifact, ResNet18Runtime, torchvision_preprocess
from qpu_xla.models.vision.runtime import VisionMode, _dynamic_w8a8_conv
from qpu_xla.ops import Conv2dW8A8Plan, bias_activation


def timed(fn):
    """Return elapsed host time and the callable result."""
    start = perf_counter_ns()
    value = fn()
    return perf_counter_ns() - start, value


def conv(args):
    """Time current W8A8 plans, keeping the matching oracle outside timing."""
    import qpu_xla.ops.conv2d_w8a8 as implementation

    results = []
    rng = np.random.default_rng(20261028)
    cases = [
        ("pointwise", (1, 128, 80, 80), (64, 128, 1, 1), 0, 1),
        ("dense-3x3", (1, 64, 80, 80), (64, 64, 3, 3), 1, 1),
        ("depthwise-3x3", (1, 128, 80, 80), (128, 1, 3, 3), 1, 128),
    ]
    for name, shape, weight_shape, padding, groups in cases:
        if args.conv_case != "all" and args.conv_case != name:
            continue
        source_values = rng.standard_normal(shape, dtype=np.float32)
        weights = rng.standard_normal(weight_shape, dtype=np.float32)
        expected = _dynamic_w8a8_conv(
            source_values, weights, stride=(1, 1), padding=(padding, padding), dilation=(1, 1), groups=groups
        )
        source_torch, weights_torch = torch.from_numpy(source_values), torch.from_numpy(weights)

        def cpu():
            return functional.conv2d(source_torch, weights_torch, padding=padding, groups=groups)

        with Device.open(data_area_size=128 * 1024 * 1024) as device, device.queue() as queue:
            source = device.tensor(shape, np.float32)
            source.numpy()[:] = source_values
            plan = Conv2dW8A8Plan(device, source_shape=shape, weight=weights, padding=padding, groups=groups)
            destination = device.tensor(plan.destination_shape, np.float32)
            original_window, original_pointwise = implementation._im2col_nchw, implementation._pointwise_nchw_matrix
            layout_spans = []

            def wrap(fn):
                def instrumented(*a, **kw):
                    start = perf_counter_ns()
                    value = fn(*a, **kw)
                    layout_spans.append(perf_counter_ns() - start)
                    return value

                return instrumented

            implementation._im2col_nchw = wrap(original_window)
            implementation._pointwise_nchw_matrix = wrap(original_pointwise)

            def qpu():
                layout_spans.clear()
                start = perf_counter_ns()
                plan.execute(destination, source, queue=queue).wait()
                wall = perf_counter_ns() - start
                # Validation is outside the timed interval, including every
                # retained invocation so a reset cannot silently corrupt it.
                np.testing.assert_allclose(destination.numpy(), expected, rtol=1e-5, atol=1e-4)
                events = [
                    {
                        "name": e.name,
                        "category": e._category,
                        "submitted_ns": e.submitted_ns,
                        "started_ns": e.started_ns,
                        "finished_ns": e.finished_ns,
                    }
                    for e in queue._events
                ]
                queue.consume_completed_event_counts()
                return {"wall_ns": wall, "layout_window_ns": sum(layout_spans), "events": events}

            try:
                qpu()
                error = float(np.max(np.abs(destination.numpy() - expected)))
                np.testing.assert_allclose(destination.numpy(), expected, rtol=1e-5, atol=1e-4)
                cpu_values = cpu().numpy().copy()
                runs = []
                for _ in range(args.warmup):
                    cpu()
                    qpu()
                for index in range(args.repeat):
                    order = ["cpu", "qpu"]
                    random.Random(20261028 + index).shuffle(order)
                    sample = {"order": order}
                    for mode in order:
                        if mode == "cpu":
                            sample["cpu_ns"] = timed(cpu)[0]
                        else:
                            sample["qpu"] = qpu()
                    runs.append(sample)
                results.append(
                    {
                        "case": name,
                        "source_shape": shape,
                        "weight_shape": weight_shape,
                        "padding": padding,
                        "groups": groups,
                        "direct_depthwise_reduction": plan._direct_depthwise,
                        "max_abs_error_matching_w8a8": error,
                        "max_abs_error_fp32": float(np.max(np.abs(destination.numpy() - cpu_values))),
                        "metadata": collect_metadata(device),
                        "samples": runs,
                    }
                )
            finally:
                implementation._im2col_nchw, implementation._pointwise_nchw_matrix = (
                    original_window,
                    original_pointwise,
                )
                plan.close()
                source.buffer.close()
                destination.buffer.close()
        print(f"completed {name}", flush=True)
    return {
        "task": "convolution-breakdown",
        "boundary": "prepared steady-state mapped-input to NCHW output; dynamic lowering and quantization included",
        "cpu_reference": "native Torch FP32 conv2d (different representation)",
        "profiling": (
            "layout wrappers and existing host/QPU queue timestamps; "
            "QPU event includes submission/wait, not device-only arithmetic"
        ),
        "results": results,
    }


def epilogue(args):
    """Compare the existing fused Torch operator with the QPU epilogue."""
    shape = (16, 262128)
    rng = np.random.default_rng(20261028)
    values = rng.standard_normal(shape, dtype=np.float32)
    biases = rng.standard_normal(shape[1], dtype=np.float32)
    source_torch, bias_torch = torch.from_numpy(values), torch.from_numpy(biases)
    cpu_output = torch.empty_like(source_torch)
    expected = np.maximum(values + biases, 0)
    with Device.open(data_area_size=64 * 1024 * 1024) as device, device.queue() as queue:
        source, bias, destination = (device.tensor(s, np.float32) for s in (shape, (shape[1],), shape))
        source.numpy()[:] = values
        bias.numpy()[:] = biases

        def cpu():
            torch.ops.aten._add_relu.out(source_torch, bias_torch, out=cpu_output)

        def qpu():
            bias_activation(destination, source, bias, relu=True, queue=queue).wait()

        cpu()
        qpu()
        np.testing.assert_array_equal(cpu_output.numpy(), expected)
        np.testing.assert_array_equal(destination.numpy(), expected)
        samples = []
        for _ in range(args.warmup):
            cpu()
            qpu()
        for index in range(args.repeat):
            order = ["cpu", "qpu"]
            random.Random(20261028 + index).shuffle(order)
            sample = {"order": order}
            for mode in order:
                sample[mode + "_ns"] = timed(cpu if mode == "cpu" else qpu)[0]
            samples.append(sample)
            queue.consume_completed_event_counts()
        metadata = collect_metadata(device)
        for tensor in (source, bias, destination):
            tensor.buffer.close()
    return {
        "task": "fused-bias-relu-control",
        "shape": shape,
        "samples": samples,
        "cpu_implementation": str(torch.ops.aten._add_relu.out._schema),
        "metadata": metadata,
        "boundary": (
            "steady-state persistent source/bias and preallocated output; QPU submission and completion included"
        ),
        "max_abs_error_cpu": 0.0,
        "max_abs_error_qpu": 0.0,
    }


def resnet(args):
    """Measure the complete model on one matched preprocessed image."""
    artifact = ResNet18Artifact.open(args.artifact)
    source = torchvision_preprocess(args.image)
    with ExitStack() as stack:
        device = stack.enter_context(Device.open(data_area_size=256 * 1024 * 1024))
        cpu = stack.enter_context(ResNet18Runtime.from_artifact(artifact, mode=VisionMode.CPU_FP32))
        qpu = stack.enter_context(ResNet18Runtime.from_artifact(artifact, mode=VisionMode.QPU_FP32, device=device))
        expected = cpu.predict_logits(source)
        actual = qpu.predict_logits(source)
        error = float(np.max(np.abs(actual - expected)))
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        assert actual.argmax() == expected.argmax()
        samples = []
        for _ in range(args.warmup):
            cpu.predict_logits(source)
            qpu.predict_logits(source)
        for index in range(args.repeat):
            order = ["cpu", "qpu"]
            random.Random(20261028 + index).shuffle(order)
            sample = {"order": order}
            for mode in order:
                runtime = cpu if mode == "cpu" else qpu
                before = runtime.context.telemetry.to_dict()
                duration, result = timed(lambda: runtime.predict_logits(source))
                after = runtime.context.telemetry.to_dict()
                sample[mode + "_ns"] = duration
                sample[mode + "_coverage"] = {k: after[k] - before[k] for k in before}
                if mode == "qpu":
                    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
            samples.append(sample)
        return {
            "task": "resnet18-whole-model",
            "artifact": str(args.artifact.resolve()),
            "artifact_manifest": artifact.metadata,
            "image": str(args.image.resolve()),
            "source_shape": list(source.shape),
            "samples": samples,
            "max_abs_error": error,
            "argmax": int(actual.argmax()),
            "metadata": collect_metadata(device),
            "boundary": (
                "preprocessed normalized FP32 input to complete 1000-class logits; "
                "excludes image decode/preprocessing and persistent-plan construction"
            ),
            "scope": "one fixed image, repeated steady-state diagnostic; not a task-accuracy campaign",
        }


def main():
    """Execute the requested experiment and retain its evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["conv", "epilogue", "resnet"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=21)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--conv-case", choices=["all", "pointwise", "dense-3x3", "depthwise-3x3"], default="all")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")
    if args.output.exists():
        parser.error("output must not already exist")
    if args.task == "resnet" and (args.artifact is None or args.image is None):
        parser.error("resnet requires --artifact and --image")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    result = {"conv": conv, "epilogue": epilogue, "resnet": resnet}[args.task](args)
    result.update(
        {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
