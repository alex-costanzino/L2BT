import numpy as np
from scipy.ndimage.measurements import label
from bisect import bisect


class GroundTruthComponent:
    """
    Stores sorted anomaly scores of a single ground truth component.
    Used to efficiently compute the region overlap for many increasing thresholds.
    """

    def __init__(self, anomaly_scores):
        """
        Initialize the module.

        Args:
            anomaly_scores: List of all anomaly scores within the ground truth
                            component as numpy array.
        """
        # Keep a sorted list of all anomaly scores within the component.
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()

        # Pointer to the anomaly score where the current threshold divides the component into OK / NOK pixels.
        self.index = 0

        # The last evaluated threshold.
        self.last_threshold = None

    def compute_overlap(self, threshold):
        """
        Compute the region overlap for a specific threshold.
        Thresholds must be passed in increasing order.

        Args:
            threshold: Threshold to compute the region overlap.

        Returns:
            Region overlap for the specified threshold.
        """
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold

        # Increase the index until it points to an anomaly score that is just above the specified threshold.
        while (self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold):
            self.index += 1

        # Compute the fraction of component pixels that are correctly segmented as anomalous.
        return 1.0 - self.index / len(self.anomaly_scores)


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by x- and corresponding y-values.
    In contrast to, e.g., 'numpy.trapz()', this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a warning.

    Args:
        x:     Samples from the domain of the function to integrate need to be sorted in ascending order. May contain
               the same value multiple times. In that case, the order of the corresponding y values will affect the
               integration with the trapezoidal rule.
        y:     Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be determined by interpolating between its
               neighbors. Must not lie outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            """WARNING: Not all x and y values passed to trapezoid are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between the last x[ins-1] and x_max. Since we do not
            # know the exact value of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    """
    Extract anomaly scores for each ground truth connected component as well as anomaly scores for each potential false
    positive pixel from anomaly maps.

    Args:
        anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
                           for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains
                           an anomaly.

    Returns:
        ground_truth_components: A list of all ground truth connected components that appear in the dataset. For each
                                 component, a sorted list of its anomaly scores is stored.

        anomaly_scores_ok_pixels: A sorted list of anomaly scores of all anomaly-free pixels of the dataset. This list
                                  can be used to quickly select thresholds that fix a certain false positive rate.
    """
    # Make sure an anomaly map is present for each ground truth map.
    assert len(anomaly_maps) == len(ground_truth_maps)

    # Initialize ground truth components and scores of potential fp pixels.
    ground_truth_components = []
    weights = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    # Collect anomaly scores within each ground truth region and for all potential fp pixels.
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):

        blob_size = []

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)

        # Store all potential fp scores.
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index:ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels

        # Fetch anomaly scores within each GT component.
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            blob_size.append(len(component_scores))
            ground_truth_components.append(GroundTruthComponent(component_scores))

        weights += blob_size

    # Sort all potential false positive scores.
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()

    return ground_truth_components, weights, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds, weighted, size_thr):
    """
    Compute the PRO curve at equidistant interpolation points for a set of anomaly maps with corresponding ground
    truth maps. The number of interpolation points can be set manually.

    Args:
        anomaly_maps:      List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
                           for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains
                           an anomaly.

        num_thresholds:    Number of thresholds to compute the PRO curve.
    Returns:
        fprs: List of false positive rates.
        pros: List of correspoding PRO values.
    """
    # Fetch sorted anomaly scores.
    ground_truth_components, weights, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)

    weights = np.array(weights)


    # Select equidistant thresholds.
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)

    fprs = [1.0]
    pros = [1.0]
    thrs = [0.0]

    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]

        # Compute the false positive rate for this threshold.
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        # Compute the PRO value for this threshold.
        pro = 0.0
        cnt = 0

        for (component, weight) in zip(ground_truth_components, weights):
            if weighted:
                pro += component.compute_overlap(threshold) * weights.min()/weight
                cnt += 1
            elif not weighted:
                if size_thr is None:
                    pro += component.compute_overlap(threshold)
                    cnt += 1
                elif size_thr is not None:
                    if weight >= size_thr[0] and weight <= size_thr[1]:
                        pro += component.compute_overlap(threshold)
                        cnt += 1
                        
        pro /= cnt

        fprs.append(fpr)
        pros.append(pro)
        thrs.append(threshold)

    # Return (FPR/PRO) pairs in increasing FPR order.
    fprs = fprs[::-1]
    pros = pros[::-1]
    thrs = thrs[::-1]

    return (fprs, pros, thrs), weights


def calculate_au_pro(gts, predictions, integration_limit = [0.3, 0.1, 0.05, 0.01], num_thresholds = 99, weighted = None, size_thr = None):
    """
    Compute the area under the PRO curve for a set of ground truth images and corresponding anomaly images.
    Args:
        gts:         List of tensors that contain the ground truth images for a single dataset object.
        predictions: List of tensors containing anomaly images for each ground truth image.
        integration_limit:    Integration limit to use when computing the area under the PRO curve.
        num_thresholds:       Number of thresholds to use to sample the area under the PRO curve.

    Returns:
        au_pro:    Area under the PRO curve computed up to the given integration limit.
        pro_curve: PRO curve values for localization (fpr,pro).
    """
    # Compute the PRO curve.
    pro_curve, weights = compute_pro(anomaly_maps = predictions, ground_truth_maps = gts, num_thresholds = num_thresholds, weighted = weighted, size_thr = size_thr)

    au_pros = []

    # Compute the area under the PRO curve.
    for int_lim in integration_limit:
        au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max = int_lim)
        au_pro /= int_lim

        au_pros.append(au_pro)

    # Return the evaluation metrics.
    return au_pros, pro_curve, weights
