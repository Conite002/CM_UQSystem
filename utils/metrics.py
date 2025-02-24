import numpy as np

def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)

def expected_calibration_error(probs, y_true, num_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    true_labels = y_true.numpy()

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if np.any(mask):
            bin_accuracy = np.mean(predictions[mask] == true_labels[mask])
            bin_confidence = np.mean(confidences[mask])
            ece += np.abs(bin_confidence - bin_accuracy) * np.sum(mask) / len(probs)

    return ece
