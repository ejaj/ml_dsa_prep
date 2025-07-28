def computeArea(x1, y1, x2, y2, x3, y3, x4, y4):
    # Area of first rectangle
    areaA = (x2-x1) * (y2-y1)
    # Area of second rectangle
    areaB = (x4, x3) * (y4 - y3)

    overlap_width = max(0, min(x2, x4) - max(x1, x3))
    overlap_height = max(0, min(y2, y4) - max(y1, y3))

    overlap_area = overlap_width * overlap_height
    return areaA + areaB - overlap_area