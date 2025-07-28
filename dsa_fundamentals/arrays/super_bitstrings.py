def superBitStrings(n, bitStrings):
    seen = set()
    for num in bitStrings:
        bin_str = bin(num)[2:].zfill(n)
        zeros_indices = [i for i, b in enumerate(bin_str) if b == '0']
        total_flisp = 2**len(zeros_indices)
        for mask in range(total_flisp):
            bits = list(bin_str)
            for j in range(zeros_indices):
                if (mask >> j) & 1:
                    bits[zeros_indices[j]] = '1'
            seen.add(''.join(bits))
    return len(seen)