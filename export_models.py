"""
Export the trained Keras CNN (sign_language_cnn.h5) to deployable formats.

Writes four files to models/:
  sign_language_cnn_fp32.tflite   TFLite, float32 (no quantization)
  sign_language_cnn_int8.tflite   TFLite, full-integer int8 (weights,
                                  activations, input and output)
  sign_language_cnn_fp32.onnx     ONNX, float32
  sign_language_cnn_int8.onnx     ONNX, static int8 quantization (QDQ)

The int8 models are calibrated on images from the TRAINING set, so the test
set stays untouched for benchmark.py.

Usage:
  python export_models.py --data-dir "Sign Language MNIST"
"""

import argparse
import os

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

MODEL_PATH = "sign_language_cnn.h5"
OUT_DIR = "models"
CALIBRATION_SAMPLES = 500
INPUT_NAME = "image"


def load_calibration_images(data_dir, n, seed=0):
    """Random training images, normalized and shaped like the notebook's."""
    df = pd.read_csv(os.path.join(data_dir, "sign_mnist_train.csv"))
    pixels = df.drop("label", axis=1).values.astype(np.float32) / 255.0
    idx = np.random.default_rng(seed).choice(len(pixels), n, replace=False)
    return pixels[idx].reshape(-1, 28, 28, 1)


def export_tflite(model, calib, out_dir):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32 = converter.convert()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = (
        lambda: ([calib[i:i + 1]] for i in range(len(calib))))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8 = converter.convert()

    for name, blob in [("fp32", fp32), ("int8", int8)]:
        path = os.path.join(out_dir, f"sign_language_cnn_{name}.tflite")
        with open(path, "wb") as f:
            f.write(blob)
        print(f"wrote {path}")


def export_onnx(model, calib, out_dir):
    import tf2onnx
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_path = os.path.join(out_dir, "sign_language_cnn_fp32.onnx")
    spec = (tf.TensorSpec([None, 28, 28, 1], tf.float32, name=INPUT_NAME),)
    fn = tf.function(lambda x: model(x, training=False))
    tf2onnx.convert.from_function(fn, input_signature=spec, opset=17,
                                  output_path=fp32_path)
    print(f"wrote {fp32_path}")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib[i:i + 1] for i in range(len(calib)))

        def get_next(self):
            batch = next(self.it, None)
            return None if batch is None else {INPUT_NAME: batch}

    prep_path = os.path.join(out_dir, "_prep.onnx")
    int8_path = os.path.join(out_dir, "sign_language_cnn_int8.onnx")
    quant_pre_process(fp32_path, prep_path)
    quantize_static(prep_path, int8_path, Reader(),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=True)
    os.remove(prep_path)
    print(f"wrote {int8_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="Sign Language MNIST",
                        help="folder containing sign_mnist_train.csv")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = tf.keras.models.load_model(args.model)
    calib = load_calibration_images(args.data_dir, CALIBRATION_SAMPLES)

    export_tflite(model, calib, args.out_dir)
    export_onnx(model, calib, args.out_dir)


if __name__ == "__main__":
    main()
