def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    num_features = len(vectors)
    num_obs = len(vectors[0])
    
    # Check if all features have the same number of observations
    for feature in vectors:
        if len(feature) != num_obs:
            raise ValueError("All features must have the same number of observations")
    
    # Calculate means of each feature
    means = [sum(feature) / num_obs for feature in vectors]
    
    # Initialize covariance matrix
    covariance_matrix = [[0.0] * num_features for _ in range(num_features)]
    
    # Calculate covariance for each pair (i, j)
    for i in range(num_features):
        for j in range(num_features):
            numerator = sum(
                (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
                for k in range(num_obs)
            )
            covariance_matrix[i][j] = numerator / (num_obs - 1)
    
    return covariance_matrix
