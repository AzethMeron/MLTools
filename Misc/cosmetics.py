def SciNotation(x, precision=2, simplify = True):
    superscript = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    base, exp = f"{x:.{precision}e}".split("e")
    exp = int(exp)
    if simplify and exp == 0:
        return f"{base}"
    if simplify and (round(float(base), precision) == round(1.00, precision)):
        return f"10{str(exp).translate(superscript)}"
    return f"{base} × 10{str(exp).translate(superscript)}"