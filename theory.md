# Introduction: Solving the 2D Poisson Equation

## 1. Physical and Mathematical Context

The **Poisson equation** is a fundamental partial differential equation (PDE) used across physics and engineering to describe how a field (like electric potential or temperature) behaves in the presence of a source (like electric charge or heat).

In two dimensions, the equation for electric potential $V$ is:

$$\nabla^2 V = \frac{\partial^{2}V}{\partial x^{2}} + \frac{\partial^{2}V}{\partial y^{2}} = -\frac{\rho(x,y)}{\epsilon}$$

Where:

*   **$\nabla^2$ (The Laplacian):** Represents the "curvature" or spatial second derivative of the potential.
*   **$\rho(x,y)$:** The charge density (the source). In this project, we use a specific Gaussian distribution: 
    $$\rho(x,y) = \rho_0 \exp\left( -\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2} \right)$$
*   **$\epsilon$:** The permittivity of the medium (assumed to be 1.0 for simplicity).

If the source term $\rho$ is zero, the equation is known as the **Laplace equation**, which describes the potential in a region "empty" of charges:
$$\frac{\partial^{2}V}{\partial x^{2}} + \frac{\partial^{2}V}{\partial y^{2}} = 0$$

## 2. Boundary Conditions (BCs)

To find a unique solution within a square domain $\Omega$, we must define what happens at the edges $\partial \Omega$:

1.  **Dirichlet Boundary Conditions:** We explicitly set the value of the potential $V$ at the edge. Here, we use sinusoidal functions (controlled by parameters $A$ through $D$) to simulate varying potentials along the four walls:
    $$V(x, y) = f(x, y) \quad \text{for} \quad (x, y) \in \partial \Omega$$
2.  **Neumann Boundary Conditions:** We set the *gradient* (slope) of the potential. For example, setting $\frac{\partial V}{\partial n} = 0$ on the right boundary simulates an insulating or reflective boundary:
    $$\nabla V \cdot \mathbf{n} = \frac{\partial V}{\partial n} = g(x, y)$$

## 3. The Numerical Approach: Successive Over-Relaxation (SOR)

To solve the Poisson equation numerically, we transform the continuous partial differential equation into a discrete system of algebraic equations. This process is known as the **Finite Difference Method (FDM)**.

### 1) Discretization of the Domain

We represent the continuous $xy$-plane as a grid (or mesh) of points. If the square domain has length $L$, we divide it into $N$ intervals along each axis, resulting in a grid spacing $h$:

$$h = \Delta x = \Delta y = \frac{L}{N}$$

A point in the domain is now identified by integer indices $(i, j)$, where the coordinates are:
$$x_i = i \cdot h, \quad y_j = j \cdot h$$

The continuous potential $V(x,y)$ is transformed into a discrete matrix $V_{i,j}$, where each entry represents the potential value at a specific grid node.

### 2) Finite Difference Approximation

To handle the second-order derivatives $\frac{\partial^2 V}{\partial x^2}$ and $\frac{\partial^2 V}{\partial y^2}$, we use the **Taylor Series expansion**. By combining the forward and backward expansions, we derive the **Central Difference Formula**:

$$\frac{\partial^2 V}{\partial x^2} \approx \frac{V_{i+1,j} - 2V_{i,j} + V_{i-1,j}}{h^2}$$
$$\frac{\partial^2 V}{\partial y^2} \approx \frac{V_{i,j+1} - 2V_{i,j} + V_{i,j-1}}{h^2}$$

Substituting these into the Poisson equation ($\nabla^2 V = -\rho/\epsilon$) gives the **Five-Point Stencil**:

$$\frac{V_{i+1,j} + V_{i-1,j} + V_{i,j+1} + V_{i,j-1} - 4V_{i,j}}{h^2} = -\frac{\rho_{i,j}}{\epsilon}$$

### 3) From Algebra to Iterative Schemes

If we solve the stencil equation for the central point $V_{i,j}$, we find that the potential at any point is the average of its four neighbors plus a correction for the local charge density:

$$V_{i,j} = \frac{1}{4} \left( V_{i+1,j} + V_{i-1,j} + V_{i,j+1} + V_{i,j-1} + \frac{h^2 \rho_{i,j}}{\epsilon} \right)$$

Because every point depends on its neighbors, we cannot solve for $V_{i,j}$ directly in one step. Instead, we use the **Gauss-Seidel** approach, where we sweep through the grid and update values using the most recent data available:

$$V_{i,j}^{\text{new}} = \frac{1}{4} \left( V_{i+1,j}^{\text{old}} + V_{i-1,j}^{\text{new}} + V_{i,j+1}^{\text{old}} + V_{i,j-1}^{\text{new}} + \frac{h^2 \rho_{i,j}}{\epsilon} \right)$$

### 4) Successive Over-Relaxation (SOR) Logic

The **Successive Over-Relaxation (SOR)** method accelerates convergence by looking at the "residual" (the difference between the current value and the calculated target) and amplifying the update.

We define the update as a weighted average between the old value and the Gauss-Seidel target using a relaxation factor $\omega$:

$$V_{i,j}^{(k+1)} = (1 - \omega)V_{i,j}^{(k)} + \omega V_{i,j}^{\text{GS target}}$$

*   **If $\omega = 1$:** The method is identical to Gauss-Seidel.
*   **If $1 < \omega < 2$ (Over-relaxation):** The algorithm pushes the value further than the local average suggests, which effectively "propagates" information across the grid much faster.
*   **Optimal $\omega$:** For a square grid of size $N$, the optimal value is approximately:
    $$\omega_{\text{opt}} \approx \frac{2}{1 + \sin(\pi/N)}$$

### 5) Convergence Criteria and Energy Functional

The numerical solver iterates until the system reaches a steady state. We monitor the **Energy Functional ($J$)**, which represents the total energy stored in the electrostatic field:

$$J[V] = \int_{\Omega} \left[ \frac{1}{2} |\nabla V|^2 - \frac{\rho}{\epsilon}V \right] dx dy$$

In the discrete version, the algorithm stops when the change in $V$ between two iterations (the $L_\infty$ norm of the residual) falls below a predefined tolerance:
$$\max |V_{i,j}^{(k+1)} - V_{i,j}^{(k)}| < \text{tol}$$

## 4. Physics-Informed Neural Networks (PINNs) vs. Numerical Methods

This project compares the traditional **SOR method** against a modern machine learning approach: **Physics-Informed Neural Networks (PINNs)**.

| Feature | Numerical (SOR) | PINNs (Deep Learning) |
| :--- | :--- | :--- |
| **Space** | Discrete grid (Nodes) | Continuous domain |
| **Mechanism** | Iterative local updates | Loss function minimization via Backpropagation |
| **Physics** | Hard-coded via finite differences | Embedded in the Neural Network's Loss Function |
| **Efficiency** | Very fast for simple 2D grids | Slower training, but mesh-free and scalable |

The PINN minimizes a composite loss function $\mathcal{L}$:
$$\mathcal{L} = \mathcal{L}_{residual} + \mathcal{L}_{BC}$$
Where the residual loss enforces the physics:
$$\mathcal{L}_{residual} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\partial^2 \hat{V}}{\partial x^2} + \frac{\partial^2 \hat{V}}{\partial y^2} + \frac{\rho}{\epsilon} \right|^2$$

By comparing the **Potential $V$** and the **Residual $R$** (how much the solution fails to satisfy the original equation), we can determine the accuracy and reliability of neural networks in solving classical electrostatic problems compared to time-tested numerical solvers.