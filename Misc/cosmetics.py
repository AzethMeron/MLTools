def SciNotation(x, precision=2):
    superscript = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    base, exp = f"{x:.{precision}e}".split("e")
    exp = int(exp)
    return f"{base} × 10{str(exp).translate(superscript)}"