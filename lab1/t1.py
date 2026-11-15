"""
Лабораторна робота №1, задача 1, варіант 9.
Рівняння: y' = f(x, y) = y^2 + x - 1.

Запуск:  python3 t1.py          (показати графік)
         python3 t1.py --save   (зберегти у t1.png)
"""
import sys
import numpy as np
import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp


# ---------- рівняння та похідні ----------
def f(x, y):
    return y**2 + x - 1


def f_x(x, y):
    return np.ones_like(np.asarray(x + 0.0 * y, dtype=float))


def f_y(x, y):
    return 2 * y


def ypp(x, y):
    """y'' = f_x + f_y * f = 1 + 2y(y^2 + x - 1)."""
    return f_x(x, y) + f_y(x, y) * f(x, y)


# ---------- аналітика в консоль ----------
def print_analysis():
    print("=" * 72)
    print("ЗАДАЧА 1, варіант 9:   y' = f(x, y) = y^2 + x - 1")
    print("=" * 72)
    print("Похідні: f_x = 1,  f_y = 2y")
    print("y'' = f_x + f_y*f = 1 + 2y(y^2 + x - 1)\n")

    print("1) Поле напрямків: у точках (m, n/2), m, n = -6..6, відрізок з нахилом f(m, n/2).\n")

    print("2) Точки екстремуму: y' = 0  <=>  y^2 + x - 1 = 0  <=>  x = 1 - y^2.")
    print("   На цій кривій y'' = f_x = 1 > 0, тобто всі екстремуми є МІНІМУМАМИ.")
    print("   Множина точок максимуму: ПОРОЖНЯ (∅).")
    print("   (Множина точок мінімуму: парабола x = 1 - y^2.)\n")

    print("3) Ізокліни f(x, y) = k:  y^2 + x - 1 = k  <=>  x = 1 + k - y^2")
    for k in (-1.5, -1, -0.5, 0.5, 1, 1.5):
        print(f"   k = {k:+.1f}:  x = {1 + k:+.1f} - y^2")
    print("   (нуль-ізокліна k = 0: x = 1 - y^2 — крива мінімумів)\n")

    print("4) Зростання: f > 0  <=>  x > 1 - y^2  (праворуч від параболи x = 1 - y^2)")
    print("   Спадання:  f < 0  <=>  x < 1 - y^2  (ліворуч, усередині параболи)\n")

    print("5) Точки перегину: y'' = 0  <=>  1 + 2y(y^2 + x - 1) = 0.")
    print("   При y = 0 маємо 1 = 0 — розв'язків немає, отже y ≠ 0 і")
    print("   x = 1 - y^2 - 1/(2y)   (дві гілки: y > 0 та y < 0).\n")

    print("6) Опуклість вгору: y'' < 0  <=>  1 + 2y(y^2 + x - 1) < 0:")
    print("   при y > 0:  x < 1 - y^2 - 1/(2y);")
    print("   при y < 0:  x > 1 - y^2 - 1/(2y).")
    print("   (Опуклість вниз — протилежні нерівності; на y = 0 маємо y'' = 1 > 0.)\n")

    print("7) Інтегральні криві через (0, -2), (0, 0), (0, 1) — чисельно (solve_ivp, RK45),")
    print("   в обидва боки від x = 0, з зупинкою при |y| > 10 (можливий «вибух» розв'язку).")
    for y0 in (-2, 0, 1):
        print(f"   у точці (0, {y0}):  y' = f = {f(0, y0):+}, y'' = {ypp(0, y0):+}")
    print()


# ---------- інтегральні криві ----------
def blow_up(x, y):
    return abs(y[0]) - 10
blow_up.terminal = True


def integral_curve(y0, x0=0.0, xmin=-6.5, xmax=6.5):
    rhs = lambda x, y: [f(x, y[0])]
    fwd = solve_ivp(rhs, (x0, xmax), [y0], events=blow_up,
                    max_step=0.01, rtol=1e-9, atol=1e-11)
    bwd = solve_ivp(rhs, (x0, xmin), [y0], events=blow_up,
                    max_step=0.01, rtol=1e-9, atol=1e-11)
    xs = np.concatenate([bwd.t[::-1], fwd.t[1:]])
    ys = np.concatenate([bwd.y[0][::-1], fwd.y[0][1:]])
    return xs, ys, bwd, fwd


def main():
    print_analysis()

    XMIN, XMAX, YMIN, YMAX = -6.5, 6.5, -3.5, 3.5
    fig, ax = plt.subplots(figsize=(15, 9))

    # ---- 4) області зростання/спадання (фон) ----
    X, Y = np.meshgrid(np.linspace(XMIN, XMAX, 700), np.linspace(YMIN, YMAX, 700))
    F = f(X, Y)
    ax.contourf(X, Y, F, levels=[-1e9, 0, 1e9],
                colors=["#f6c9c9", "#cdeccd"], alpha=0.55, zorder=0)

    # ---- 6) опуклість вгору (штрихування) ----
    YPP = ypp(X, Y)
    cs = ax.contourf(X, Y, YPP, levels=[-1e12, 0], colors="none",
                     hatches=["////"], zorder=1)
    for coll in getattr(cs, "collections", [cs]):
        try:
            coll.set_edgecolor("#7a5c99")
            coll.set_linewidth(0)
        except Exception:
            pass

    yy = np.linspace(YMIN, YMAX, 800)

    # ---- 3) ізокліни ----
    ks = [-1.5, -1, -0.5, 0.5, 1, 1.5]
    iso_colors = ["#08306b", "#2171b5", "#6baed6", "#fd8d3c", "#e6550d", "#a63603"]
    for k, c in zip(ks, iso_colors):
        ax.plot(1 + k - yy**2, yy, "--", color=c, lw=1.3, zorder=3)

    # ---- 2) нуль-ізокліна = крива мінімумів ----
    ax.plot(1 - yy**2, yy, color="black", lw=2.6, zorder=4)

    # ---- 5) крива перегину (дві гілки) ----
    for sgn in (+1, -1):
        yb = sgn * np.linspace(0.02, YMAX, 600)
        ax.plot(1 - yb**2 - 1 / (2 * yb), yb, color="#9b30ff", lw=2.4, zorder=4)

    # ---- 1) поле напрямків у точках (m, n/2) ----
    L = 0.42
    for m in range(-6, 7):
        for n in range(-6, 7):
            x0, y0 = m, n / 2
            s = f(x0, y0)
            dx = L / 2 / np.sqrt(1 + s**2)
            dy = s * dx
            ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy],
                    color="#333333", lw=1.2, zorder=5)
            ax.plot(x0, y0, ".", color="#333333", ms=2.5, zorder=5)

    # ---- 7) інтегральні криві ----
    curve_colors = {-2: "#d62728", 0: "#1f77b4", 1: "#ff7f0e"}
    for y0, c in curve_colors.items():
        xs, ys, bwd, fwd = integral_curve(y0)
        ax.plot(xs, ys, color=c, lw=3, zorder=6)
        ax.plot(0, y0, "o", color=c, mec="black", ms=8, zorder=7)
        for side, sol in (("x<0", bwd), ("x>0", fwd)):
            if sol.t_events[0].size:
                print(f"   крива через (0, {y0}): |y| досягає 10 при x ≈ {sol.t_events[0][0]:.4f} ({side})")

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.6, zorder=2)
    ax.axvline(0, color="gray", lw=0.6, zorder=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Задача 1, варіант 9:  y' = y² + x − 1", fontsize=14)

    handles = [
        Line2D([], [], color="#333333", lw=1.2, label="поле напрямків у (m, n/2)"),
        Line2D([], [], color="black", lw=2.6,
               label="y' = 0: x = 1 − y² (лише мінімуми; максимумів ∅)"),
    ]
    for k, c in zip(ks, iso_colors):
        handles.append(Line2D([], [], color=c, ls="--", lw=1.3,
                              label=f"ізокліна k = {k:+g}: x = {1 + k:g} − y²"))
    handles += [
        Line2D([], [], color="#9b30ff", lw=2.4, label="перегин: x = 1 − y² − 1/(2y)"),
        Patch(facecolor="#cdeccd", label="зростання: x > 1 − y²"),
        Patch(facecolor="#f6c9c9", label="спадання: x < 1 − y²"),
        Patch(facecolor="white", edgecolor="#7a5c99", hatch="////",
              label="опуклі вгору: y'' < 0"),
    ]
    for y0, c in curve_colors.items():
        handles.append(Line2D([], [], color=c, lw=3, label=f"інтегральна крива через (0, {y0})"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9.5)
    fig.tight_layout()

    if "--save" in sys.argv:
        fig.savefig("t1.png", dpi=150, bbox_inches="tight")
        print("\nГрафік збережено у t1.png")
    else:
        plt.show()


if __name__ == "__main__":
    main()