"""
Лабораторна робота №1, задача 2, варіант 9.
Автономне рівняння: y' = g(y) = y - y^2 = y(1 - y).

Запуск:  python3 t2.py          (показати графік)
         python3 t2.py --save   (зберегти у t2.png)
"""
import sys
import numpy as np
import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp

Y1, Y2 = 0.0, 1.0          # нулі g(y) у порядку зростання
B = Y2                     # старший коефіцієнт < 0  =>  b = y2 = 1


def g(y):
    return y - y**2


def g_prime(y):
    return 1 - 2 * y


def x_star(y0):
    """Вертикальна асимптота: 1 - y0 + y0*e^x = 0  =>  x* = ln(1 - 1/y0)."""
    return np.log(1 - 1 / y0)


def y_exact(x, y0):
    """Явний розв'язок задачі Коші y(0) = y0 на інтервалі I_{y0}; поза ним NaN."""
    x = np.asarray(x, dtype=float)
    den = 1 - y0 + y0 * np.exp(x)
    y = y0 * np.exp(x) / den
    if y0 > 1:
        y = np.where(x > x_star(y0), y, np.nan)
    elif y0 < 0:
        y = np.where(x < x_star(y0), y, np.nan)
    return y


def interval(y0):
    if y0 in (Y1, Y2):
        return "(-∞, +∞)  (сталий розв'язок)"
    if Y1 < y0 < Y2:
        return "(-∞, +∞)"
    if y0 > Y2:
        return f"(x*, +∞) = ({x_star(y0):.6f}, +∞)"
    return f"(-∞, x*) = (-∞, {x_star(y0):.6f})"


# ---------- аналітика в консоль ----------
def print_analysis():
    print("=" * 72)
    print("ЗАДАЧА 2, варіант 9:   y' = g(y) = y - y^2 = y(1 - y)")
    print("=" * 72)
    print("Нулі g: y1 = 0, y2 = 1 (положення рівноваги, сталі розв'язки y ≡ 0, y ≡ 1).\n")

    print("1) Векторне поле на прямій y: вектор від y0 до y0 + g(y0):")
    for y0 in range(-4, 5):
        gv = g(y0)
        if gv == 0:
            print(f"   y0 = {y0:+d}: g = 0 — стаціонарна точка, вектора немає")
        else:
            print(f"   y0 = {y0:+d}: g = {gv:+d}, вектор [{y0}, {y0 + gv}], "
                  f"довжина {abs(gv)}, напрям {'вгору' if gv > 0 else 'вниз'}")
    print()

    print("2) Монотонність (знак g = y(1 - y)):")
    print("   y < 0:     g < 0  — спадання")
    print("   0 < y < 1: g > 0  — зростання")
    print("   y > 1:     g < 0  — спадання")
    print("   y'' = g'(y)g(y) = (1 - 2y)·y(1 - y) = 0  <=>  y = 0, y = 1/2, y = 1.")
    print("   y = 0, y = 1 — сталі розв'язки; точки перегину кривих — на прямій y = 1/2")
    print("   (її перетинають лише криві з 0 < y0 < 1).")
    print("   Опуклість: y'' > 0 (вниз) при 0 < y < 1/2 та y > 1;")
    print("              y'' < 0 (вгору) при y < 0 та 1/2 < y < 1.\n")

    print("3) Явний розв'язок (відокремлення змінних):")
    print("   dy / (y(1 - y)) = dx,   1/(y(1-y)) = 1/y + 1/(1 - y)")
    print("   ln|y| - ln|1 - y| = x + C   <=>   ln|(y - 1)/y| = -x + C")
    print("   y/(1 - y) = y0/(1 - y0) · e^x   =>")
    print("   y(x, y0) = y0·e^x / (1 - y0 + y0·e^x)")
    print("   Асимптота: 1 - y0 + y0·e^x = 0  =>  x* = ln((y0 - 1)/y0) = ln(1 - 1/y0)")
    print("   • y0 = 0 або y0 = 1:  y ≡ y0,  I = (-∞, +∞)")
    print("   • 0 < y0 < 1: (y0-1)/y0 < 0, асимптот немає, I = (-∞, +∞), y → 1 при x → +∞, y → 0 при x → -∞")
    print("   • y0 > 1: 0 < (y0-1)/y0 < 1, x* < 0, I = (x*, +∞), y → +∞ при x → x*+, y → 1 при x → +∞")
    print("   • y0 < 0: (y0-1)/y0 > 1,  x* > 0, I = (-∞, x*), y → -∞ при x → x*-, y → 0 при x → -∞")

    # перевірка підстановкою
    print("\n   Перевірка підстановкою (max |y' - g(y)| на сітці, y' — центральна різниця):")
    for y0 in (-0.5, 0.3, 2.0):
        lo, hi = (-5, 5)
        if y0 > 1:
            lo = x_star(y0) + 0.05
        if y0 < 0:
            hi = x_star(y0) - 0.05
        xs = np.linspace(lo, hi, 4001)
        h = 1e-6
        dy = (y_exact(xs + h, y0) - y_exact(xs - h, y0)) / (2 * h)
        res = np.nanmax(np.abs(dy - g(y_exact(xs, y0))) / (1 + np.abs(g(y_exact(xs, y0)))))
        sol = solve_ivp(lambda x, y: [g(y[0])], (0, hi if y0 < 0 else 3), [y0],
                        rtol=1e-11, atol=1e-13, dense_output=True)
        xt = np.linspace(0, (hi if y0 < 0 else 3), 200)
        err = np.max(np.abs(sol.sol(xt)[0] - y_exact(xt, y0)))
        print(f"   y0 = {y0:+.1f}: відн. нев'язка {res:.1e}; відхилення від solve_ivp {err:.1e}")
    print()

    print("4) Графіки y(x, y0) для y0 з [0, 1], (-∞, 0), (1, +∞):")
    for y0 in (0.0, 0.3, 1.0, -0.5, 2.0):
        print(f"   y0 = {y0:+.1f}: I = {interval(y0)}")
    print(f"   Для y0 = -0.5: x* = ln 3 ≈ {x_star(-0.5):.6f};   для y0 = 2: x* = ln(1/2) ≈ {x_star(2.0):.6f}\n")

    print("5) Старший коефіцієнт від'ємний: спадання (-∞, 0), зростання (0, 1), спадання (1, +∞).")
    print("   I = (a, b) = (0, 1) — зростання, J = (b, c) = (1, +∞) — спадання,  b = 1.")
    print("   y(x, 3/2) = 3e^x / (3e^x - 1),   y(x, 1/2) = e^x / (e^x + 1).")
    print("   Нехай u = e^x ≥ 1. Різниця:")
    print("   y(x,3/2) - y(x,1/2) = 3u/(3u-1) - u/(u+1) = 4u / ((3u - 1)(u + 1)) > 0,")
    print("   спадає по u (чисельник похідної -4(3u²+1) < 0), при x = 0 дорівнює 1.")
    print("   4u / ((3u-1)(u+1)) < 10^-3  <=>  3u² - 3998u - 1 > 0")
    u0 = (3998 + np.sqrt(3998**2 + 12)) / 6
    x0 = np.log(u0)
    print(f"   u > (3998 + √(3998² + 12))/6 ≈ {u0:.6f}")
    print(f"   x > x0 = ln((3998 + √15984016)/6) ≈ {x0:.6f}")
    d = lambda x: y_exact(x, 1.5) - y_exact(x, 0.5)
    print(f"   Перевірка: різниця при x0 = {d(x0):.10f}, при x0 - 0.01: {d(x0 - 0.01):.10f}, "
          f"при x0 + 0.01: {d(x0 + 0.01):.10f}")
    xs = np.linspace(0, 12, 1_200_001)
    first = xs[np.argmax(np.abs(d(xs)) < 1e-3)]
    print(f"   Чисельний пошук на сітці з кроком 1e-5: перше x з |різниця| < 1e-3 ≈ {first:.5f}")
    print(f"   Відповідь: |y(x, 3/2) - y(x, 1/2)| < 10^-3 для всіх x > {x0:.4f} (наприклад, x = 7.2).\n")
    return x0


def main():
    x0 = print_analysis()
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))

    # ---------- (1) векторне поле на прямій ----------
    ax = axs[0, 0]
    ax.axvline(0, color="black", lw=1.2)
    pts = list(range(-4, 5))
    for i, y0 in enumerate(pts):
        gv = g(y0)
        off = 0.35 + 0.3 * i
        if gv == 0:
            ax.plot(0, y0, "o", color="#2ca02c", ms=10, zorder=5)
            continue
        c = "#1f77b4" if gv > 0 else "#d62728"
        ax.annotate("", xy=(off, y0 + gv), xytext=(off, y0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=16))
        ax.plot([0, off], [y0, y0], ":", color="gray", lw=0.8)
        ax.plot(0, y0, "o", color="black", ms=5)
        ax.text(off + 0.05, y0 + gv, f"{y0 + gv}", fontsize=8, va="center", color=c)
    for y0 in pts:
        ax.text(-0.12, y0, f"{y0}", ha="right", va="center", fontsize=10)
    ax.set_xlim(-0.6, 3.2)
    ax.set_ylim(-26, 5)
    ax.set_xticks([])
    ax.set_ylabel("y")
    ax.set_title("1) Векторне поле на прямій y: вектор від y₀ до y₀ + g(y₀)\n"
                 "(вектори зсунуто вбік від осі, щоб не накладались)")
    ax.legend(handles=[
        Patch(color="#d62728", label="g(y₀) < 0 — вниз"),
        plt.Line2D([], [], marker="o", color="#2ca02c", ls="", ms=9,
                   label="g(y₀) = 0: y₀ = 0, 1 — рівновага"),
    ], loc="lower right", fontsize=9)

    # ---------- (2) монотонність і перегин ----------
    ax = axs[0, 1]
    XL = (-4, 4)
    ax.axhspan(-2, 0, color="#f6c9c9", alpha=0.6)
    ax.axhspan(0, 1, color="#cdeccd", alpha=0.6)
    ax.axhspan(1, 3, color="#f6c9c9", alpha=0.6)
    xs = np.linspace(*XL, 4000)
    for y0 in (-0.05, -0.2, -0.6):
        ax.plot(xs, y_exact(xs, y0), color="#444444", lw=1.3)
    for y0 in (0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98):
        ax.plot(xs, y_exact(xs, y0), color="#444444", lw=1.3)
    for y0 in (1.05, 1.3, 2.0):
        ax.plot(xs, y_exact(xs, y0), color="#444444", lw=1.3)
    ax.axhline(0, color="#2ca02c", lw=2.5)
    ax.axhline(1, color="#2ca02c", lw=2.5)
    ax.axhline(0.5, color="#9b30ff", lw=2, ls="--")
    xi = np.log(1 / np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98]) - 1)  # y = 1/2 при e^x = (1-y0)/y0
    ax.plot(xi, 0.5 * np.ones_like(xi), "o", color="#9b30ff", ms=6)
    ax.set_xlim(*XL)
    ax.set_ylim(-2, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2) Області монотонності та точки перегину")
    ax.legend(handles=[
        Patch(color="#cdeccd", label="зростання: 0 < y < 1"),
        Patch(color="#f6c9c9", label="спадання: y < 0, y > 1"),
        plt.Line2D([], [], color="#2ca02c", lw=2.5, label="сталі розв'язки y = 0, y = 1"),
        plt.Line2D([], [], color="#9b30ff", lw=2, ls="--", marker="o",
                   label="перегин: y = 1/2"),
        plt.Line2D([], [], color="#444444", lw=1.3, label="інтегральні криві"),
    ], loc="lower right", fontsize=9)

    # ---------- (4) явні розв'язки з трьох інтервалів ----------
    ax = axs[1, 0]
    xs = np.linspace(-4, 5, 20001)
    cases = [(0.3, "#1f77b4", "y₀ = 0.3 ∈ [0, 1], I = ℝ"),
             (-0.5, "#d62728", f"y₀ = −0.5 < 0, I = (−∞, ln 3), x* ≈ {x_star(-0.5):.3f}"),
             (2.0, "#ff7f0e", f"y₀ = 2 > 1, I = (ln ½, +∞), x* ≈ {x_star(2.0):.3f}")]
    for y0, c, lab in cases:
        ax.plot(xs, y_exact(xs, y0), color=c, lw=2.5, label=lab)
        ax.plot(0, y0, "o", color=c, mec="black", ms=8)
        if y0 < 0 or y0 > 1:
            ax.axvline(x_star(y0), color=c, ls=":", lw=1.8)
    ax.axhline(0, color="#2ca02c", lw=2, label="y₀ = 0 та y₀ = 1: сталі розв'язки")
    ax.axhline(1, color="#2ca02c", lw=2)
    ax.set_xlim(-4, 5)
    ax.set_ylim(-4, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("4) y(x, y₀) = y₀eˣ / (1 − y₀ + y₀eˣ); пунктир — вертикальні асимптоти")
    ax.legend(loc="upper left", fontsize=9)

    # ---------- (5) збіжність до b = 1 ----------
    ax = axs[1, 1]
    xs = np.linspace(0, 10, 4001)
    ya = y_exact(xs, 1.5)
    yb = y_exact(xs, 0.5)
    ax.plot(xs, ya, color="#ff7f0e", lw=2.5, label="y(x, b + ½) = 3eˣ/(3eˣ − 1)")
    ax.plot(xs, yb, color="#1f77b4", lw=2.5, label="y(x, b − ½) = eˣ/(eˣ + 1)")
    ax.axhline(1, color="#2ca02c", lw=1.5, ls="--", label="y = b = 1")
    ax.axvline(x0, color="black", lw=1.2, ls=":", label=f"x₀ ≈ {x0:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-0.1, 1.6)
    ax.set_title("5) b = 1: обидві криві прямують до b; вставка — різниця (лог. шкала)")
    ax.legend(loc="upper right", fontsize=9)
    ins = ax.inset_axes([0.45, 0.07, 0.52, 0.36])
    ins.semilogy(xs, ya - yb, color="#9b30ff", lw=2)
    ins.axhline(1e-3, color="red", ls="--", lw=1)
    ins.axvline(x0, color="black", ls=":", lw=1)
    ins.text(0.3, 1.5e-3, "10⁻³", color="red", fontsize=8)
    ins.text(x0 + 0.15, 0.02, f"x₀ ≈ {x0:.3f}", fontsize=8)
    ins.text(0.98, 0.95, "|y(x, 3/2) − y(x, 1/2)| = 4eˣ/((3eˣ−1)(eˣ+1))", transform=ins.transAxes,
             ha="right", va="top", fontsize=8)
    ins.tick_params(labelsize=7)

    fig.suptitle("Задача 2, варіант 9:  y' = y − y²", fontsize=15)
    fig.tight_layout()
    if "--save" in sys.argv:
        fig.savefig("t2.png", dpi=150, bbox_inches="tight")
        print("Графік збережено у t2.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()