LEARNING_RATE = 0.001



def mean_squared_error(y_true, y_pred):
    residuals_squared = []

    for y_value in range(len(y_true)):
        residuals_squared.append((y_true[y_value] - y_pred[y_value]) ** 2)

    residual_sum = sum(residuals_squared)

    return residual_sum / len(y_true)



def calculate_intercept_gradient(X, y_true, slopes, intercept):
    intercept_gradient_contributions = []
    weighted_feature_sums = []

    for sample_index in range(len(X)):
        weighted_features = []
        for feature_index in range(len(slopes)):
            weighted_features.append(slopes[feature_index] * X[sample_index][feature_index])
        weighted_feature_sums.append(sum(weighted_features))

    for sample_index in range(len(y_true)):
        intercept_gradient_contributions.append((-2*(y_true[sample_index] - weighted_feature_sums[sample_index] - intercept)))
    total_intercept_gradient = sum(intercept_gradient_contributions)

    return total_intercept_gradient / len(y_true)



def calculate_slope_gradient(X, y_true, slopes, intercept):
    sample_slope_gradients = []
    weighted_feature_sums = []
    average_slope_gradients = []

    for sample_index in range(len(X)):
        weighted_features = []
        for feature_index in range(len(slopes)):
            weighted_features.append(slopes[feature_index] * X[sample_index][feature_index])
        weighted_feature_sums.append(sum(weighted_features))

    for sample_index in range(len(y_true)):
        sample_gradients = []
        for feature_index in range(len(slopes)):
            sample_gradients.append((-2*X[sample_index][feature_index]*(y_true[sample_index] - weighted_feature_sums[sample_index] - intercept)))
        sample_slope_gradients.append(sample_gradients)

    for slope_index in range(len(slopes)):
        total_slope_gradient = 0
        for sample_index in range(len(sample_slope_gradients)):
            total_slope_gradient += (sample_slope_gradients[sample_index][slope_index])
        average_slope_gradients.append(total_slope_gradient / len(sample_slope_gradients))

    return average_slope_gradients



def update_intercept(X, y_true, m, b, learning_rate=LEARNING_RATE):
    intercept_gradient = calculate_intercept_gradient(X, y_true, m, b)
    b = b - learning_rate * intercept_gradient
    return b



def update_slope(X, y_true, m, b, learning_rate=LEARNING_RATE):
    slope_gradient = calculate_slope_gradient(X, y_true, m, b)
    updated_slopes = []
    for slope in range(len(m)):
        slope = m[slope] - learning_rate * slope_gradient[slope]
        updated_slopes.append(slope)

    return updated_slopes

