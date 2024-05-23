
### 1. Aitken's Scheme (Finite Differences)

Aitken's scheme, also known as Aitken's delta-squared process, is a method for calculating finite differences to approximate derivatives and interpolate values. The scheme is particularly useful for polynomial interpolation and numerical analysis. Here's a breakdown of how it works:

- **Objective**: Given a set of data points `(x0, y0)`, `(x1, y1)`, ..., `(xn, yn)`, where `xi` are the x-coordinates and `yi` are the corresponding y-coordinates, Aitken's scheme calculates a table of finite differences.

- **Finite Difference Table**:
  - Construct a table `F` where `F[i, j]` represents the finite difference of `j`th order at point `xi`.
  - The first column `F[:, 0]` is initialized with the y-coordinates (`yi`).
  - Subsequent columns are computed using the formula:
    \[ F[i, j] = \frac{F[i + 1, j - 1] - F[i, j - 1]}{xi + j - xi} \]
    where `i` ranges from `0` to `n - j` and `j` ranges from `1` to `n`.

- **Interpolation**:
  - Once the finite difference table `F` is computed, it can be used to interpolate values at specific points using methods like the progressive Newton formula.

### 2. Progressive Newton Formula (Newton's Divided Differences)

The progressive Newton formula is a method of polynomial interpolation based on Newton's divided differences. It allows for the construction of an interpolating polynomial without explicitly computing the polynomial coefficients. Here's how it works:

- **Objective**: Given a set of data points `(x0, y0)`, `(x1, y1)`, ..., `(xn, yn)`, the progressive Newton formula computes the value of the interpolating polynomial at a specific point `target_x`.

- **Process**:
  - Calculate the divided differences `F[i, j]` using the finite differences computed by Aitken's scheme.
  - Use the formula to compute the interpolated value at `target_x`:
    \[ P(target_x) = y0 + F[0, 1](target_x - x0) + F[0, 2](target_x - x0)(target_x - x1) + \ldots \]

- **Efficiency**: The progressive Newton formula avoids explicit computation of the polynomial coefficients, making it efficient for interpolation.

### 3. Least Squares Method (Polynomial Interpolation)

The least squares method is a mathematical technique used to approximate the solution of overdetermined systems (more equations than unknowns) by minimizing the sum of the squares of the residuals (the differences between observed and predicted values). Here's how it applies to polynomial interpolation:

- **Objective**: Given a set of data points `(x0, y0)`, `(x1, y1)`, ..., `(xn, yn)`, find a polynomial of degree `m` that best fits the data.

- **Process**:
  - Formulate a system of linear equations using the Vandermonde matrix `A`:
    \[ A = \begin{bmatrix} 1 & x0 & x0^2 & \ldots & x0^m \\ 1 & x1 & x1^2 & \ldots & x1^m \\ \vdots & \vdots & \vdots & & \vdots \\ 1 & xn & xn^2 & \ldots & xn^m \end{bmatrix} \]
  - Solve the linear system `A * c = y` using the least squares method, where `c` is the vector of coefficients of the polynomial.
  - The solution `c` minimizes the sum of squared errors:
    \[ \| A * c - y \|_2^2 \]

- **Interpolation**:
  - Once the coefficients `c` are obtained, the polynomial `P(x)` of degree `m` can be evaluated at any `target_x` using `np.polyval(c[::-1], target_x)`.

### Summary

- **Aitken's Scheme**: Computes finite differences to aid in interpolation and approximation of values.
- **Progressive Newton Formula**: Computes interpolating polynomials efficiently using Newton's divided differences.
- **Least Squares Method**: Determines the best-fit polynomial by minimizing the sum of squared residuals, allowing for robust interpolation and curve fitting.