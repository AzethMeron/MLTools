def SciNotation(x, precision=2, simplify = True):
    superscript = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    base, exp = f"{x:.{precision}e}".split("e")
    exp = int(exp)
    if simplify and exp == 0:
        return f"{base}"
    if simplify and (round(float(base), precision) == round(1.00, precision)):
        return f"10{str(exp).translate(superscript)}"
    return f"{base} × 10{str(exp).translate(superscript)}"

def CountParameters(model, return_sci_notation = False, trainable_only = True):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
    if return_sci_notation: return SciNotation(num)
    return num
