"""
Compare the original Keras model with its TFLite and ONNX exports.

For each model, reports:
  - file size on disk
  - test-set accuracy (all 7,172 Sign MNIST test images)
  - agreement with the Keras model's top-1 prediction
  - single-image latency (batch 1, one CPU thread), median and p95

Batch 1 on one thread mirrors the real-time demo, which classifies one
crop per frame. Run export_models.py first.

Usage:
  python benchmark.py --data-dir "Sign Language MNIST"
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import onnxruntime as ort  # noqa: E402
import tensorflow as tf  # noqa: E402

MODEL_PATH = "sign_language_cnn.h5"
MODELS_DIR = "models"
WARMUP_RUNS = 50
TIMED_RUNS = 1000


def load_test_set(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "sign_mnist_test.csv"))
    x = df.drop("label", axis=1).values.astype(np.float32) / 255.0
    return x.reshape(-1, 28, 28, 1), df["label"].values


class KerasRunner:
    """Keras model called directly, without model.predict's per-call setup."""

    def __init__(self, path):
        self.model = tf.keras.models.load_model(path)
        self.fn = tf.function(lambda x: self.model(x, training=False))

    def predict_all(self, x):
        return self.model.predict(x, batch_size=256, verbose=0)

    def predict_one(self, x):
        return self.fn(x).numpy()


class KerasPredictRunner(KerasRunner):
    """model.predict(x) per image, which is what realtime_demo.py does."""

    def predict_one(self, x):
        return self.model.predict(x, verbose=0)


class TFLiteRunner:
    def __init__(self, path):
        self.interp = tf.lite.Interpreter(model_path=path, num_threads=1)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def _quantize(self, x):
        if self.inp["dtype"] == np.float32:
            return x
        scale, zero = self.inp["quantization"]
        q = np.round(x / scale + zero)
        info = np.iinfo(self.inp["dtype"])
        return np.clip(q, info.min, info.max).astype(self.inp["dtype"])

    def _dequantize(self, y):
        if self.out["dtype"] == np.float32:
            return y
        scale, zero = self.out["quantization"]
        return (y.astype(np.float32) - zero) * scale

    def predict_one(self, x):
        self.interp.set_tensor(self.inp["index"], self._quantize(x))
        self.interp.invoke()
        return self._dequantize(self.interp.get_tensor(self.out["index"]))

    def predict_all(self, x):
        return np.concatenate([self.predict_one(x[i:i + 1])
                               for i in range(len(x))])


class OnnxRunner:
    def __init__(self, path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, opts,
                                         providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def predict_one(self, x):
        return self.sess.run(None, {self.name: x})[0]

    def predict_all(self, x):
        return np.concatenate([self.predict_one(x[i:i + 256])
                               for i in range(0, len(x), 256)])


def latency_ms(runner, sample, runs):
    for _ in range(WARMUP_RUNS):
        runner.predict_one(sample)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        runner.predict_one(sample)
        times.append((time.perf_counter() - start) * 1000)
    return np.median(times), np.percentile(times, 95)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="Sign Language MNIST",
                        help="folder containing sign_mnist_test.csv")
    parser.add_argument("--runs", type=int, default=TIMED_RUNS)
    args = parser.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    x, y = load_test_set(args.data_dir)
    sample = x[:1]

    m = lambda name: os.path.join(MODELS_DIR, name)  # noqa: E731
    candidates = [
        ("Keras .h5, model.predict", MODEL_PATH, KerasPredictRunner,
         args.runs // 10),
        ("Keras .h5, tf.function", MODEL_PATH, KerasRunner, args.runs),
        ("TFLite fp32", m("sign_language_cnn_fp32.tflite"), TFLiteRunner,
         args.runs),
        ("TFLite int8", m("sign_language_cnn_int8.tflite"), TFLiteRunner,
         args.runs),
        ("ONNX fp32", m("sign_language_cnn_fp32.onnx"), OnnxRunner,
         args.runs),
        ("ONNX int8", m("sign_language_cnn_int8.onnx"), OnnxRunner,
         args.runs),
    ]

    reference = None
    rows = []
    for name, path, cls, runs in candidates:
        runner = cls(path)
        pred = runner.predict_all(x).argmax(axis=1)
        if reference is None:
            reference = pred
        median, p95 = latency_ms(runner, sample, runs)
        rows.append((name, os.path.getsize(path) / 1024, (pred == y).mean(),
                     (pred == reference).mean(), median, p95))

    print(f"\n{'Model':<26}{'Size (KB)':>10}{'Test acc':>10}"
          f"{'Agree':>9}{'Median ms':>11}{'p95 ms':>9}")
    for name, size, acc, agree, median, p95 in rows:
        print(f"{name:<26}{size:>10.0f}{acc:>10.2%}{agree:>9.2%}"
              f"{median:>11.3f}{p95:>9.3f}")


if __name__ == "__main__":
    main()
