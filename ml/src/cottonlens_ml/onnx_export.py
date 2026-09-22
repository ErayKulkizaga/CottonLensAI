"""The same native Keras ONNX path is used by the smoke test and final release."""

from pathlib import Path

import numpy as np


def export_lstm(model, scaler, horizon: int, path: Path, raw_sequences: np.ndarray, target_scaler=None) -> dict:
    import onnx
    import onnxruntime as ort
    import tensorflow as tf

    if horizon not in (1, 5) or scaler is None:
        raise ValueError("LSTM export requires T+1/T+5 and the fitted train scaler")
    raw = np.asarray(raw_sequences, dtype=np.float32)
    expected_shape = (60, len(scaler.mean_))
    if raw.ndim != 3 or raw.shape[1:] != expected_shape or len(raw) == 0:
        raise ValueError(f"Expected nonempty raw sequences (N, {expected_shape}), got {raw.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ <= 0):
        raise ValueError("Non-finite inputs or invalid scaler")
    column = 0 if horizon == 1 else 1

    def clone_layer(layer):
        config = layer.get_config()
        if isinstance(layer, tf.keras.layers.LSTM):
            # Only the inference copy uses standard TF ops; fitted weights, training
            # architecture, GPU training and checkpoints are unchanged.
            config["use_cudnn"] = False
        return layer.__class__.from_config(config)

    with tf.device("/CPU:0"):
        inference_model = tf.keras.models.clone_model(model, clone_function=clone_layer)
        inference_model.set_weights(model.get_weights())
        inputs = tf.keras.Input(shape=expected_shape, dtype="float32", name="features")
        scaled = tf.keras.layers.Rescaling(
            scale=(1.0 / scaler.scale_).astype(np.float32),
            offset=(-scaler.mean_ / scaler.scale_).astype(np.float32),
            name="train_fitted_standard_scaler",
        )(inputs)
        outputs = inference_model(scaled, training=False)[:, column:column + 1]
        if target_scaler is not None:
            outputs = tf.keras.layers.Rescaling(
                scale=float(target_scaler.scale_[column]),
                offset=float(target_scaler.mean_[column]),
                name="train_fitted_target_inverse",
            )(outputs)
        wrapper = tf.keras.Model(inputs, outputs)
        expected = np.asarray(wrapper(raw, training=False))
        scaled_reference = scaler.transform(raw.reshape(-1, raw.shape[-1])).reshape(raw.shape).astype(np.float32)
        original = np.asarray(model(scaled_reference, training=False))[:, column:column + 1]
        if target_scaler is not None:
            original = original * target_scaler.scale_[column] + target_scaler.mean_[column]
        if not np.isfinite(expected).all() or not np.isfinite(original).all():
            raise ValueError("Non-finite TensorFlow prediction")
        wrapper_error = float(np.max(np.abs(original - expected)))
        if wrapper_error >= 1e-4:
            raise ValueError(f"Inference copy/scaler parity failed: {wrapper_error}")
        # Keras 3.10 calls tf2onnx.from_function internally; no from_keras call.
        wrapper.export(
            str(path), format="onnx", verbose=False,
            input_signature=[tf.TensorSpec((1, *expected_shape), tf.float32, name="features")],
        )

    onnx.checker.check_model(onnx.load(str(path)), full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    actual = np.concatenate([
        np.asarray(session.run(None, {session.get_inputs()[0].name: row[None, ...]})[0]).reshape(1, 1)
        for row in raw
    ])
    error = float(np.max(np.abs(expected - actual)))
    original_error = float(np.max(np.abs(original - actual)))
    if not np.isfinite(actual).all() or max(error, original_error) >= 1e-4:
        raise ValueError(f"T+{horizon} ONNX parity failed: wrapper={error}, original={original_error}")
    return {
        "converter": "keras.Model.export(onnx)", "onnx_max_abs_diff": error,
        "original_model_max_abs_diff": original_error, "wrapper_max_abs_diff": wrapper_error,
        "parity_sequences": len(raw),
    }
