def phi_corr(x: list[int], y: list[int]) -> float:
    
    # Check if lengths match
    if len(x) != len(y):
        raise ValueError("Both lists must have the same length.")
    TP=TN=FP=FN=0
    for xi, yi in zip(x, y):
        if xi ==1 and yi==1:
            TP += 1
        if xi==0 and yi==0:
            TN+=1
        if xi==0 and yi == 1:
            FP+=1
        if xi==1 and yi==0:
            FN+=1
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    if denominator == 0:
        return 0.0
    
    # Phi formula
    phi = ((TP * TN) - (FP * FN)) / denominator

    # Return rounded result
    return round(phi, 4)

