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

Since PDEs are difficult to solve analytically for complex sources, we use numerical methods. We "discretize" the space into a grid of $N \times N$ points with spacing $h$. Using finite differences, the equation becomes:
$$\frac{V_{i+1,j} + V_{i-1,j} + V_{i,j+1} + V_{i,j-1} - 4V_{i,j}}{h^2} = -\frac{\rho_{i,j}}{\epsilon}$$

The **Successive Over-Relaxation (SOR)** method is an iterative algorithm used to solve these grid-based equations. It improves upon the basic Gauss-Seidel method by using a relaxation factor, $\omega$:

*   **Update Rule:** Each node $(i, j)$ is updated by looking at its four neighbors:
    $$V_{i,j}^{(k+1)} = (1-\omega)V_{i,j}^{(k)} + \frac{\omega}{4} \left( V_{i+1,j}^{(k)} + V_{i-1,j}^{(k+1)} + V_{i,j+1}^{(k)} + V_{i,j-1}^{(k+1)} + \frac{h^2 \rho_{i,j}}{\epsilon} \right)$$
*   **The $\omega$ Parameter:** 
    *   If $\omega = 1$, the method is standard Gauss-Seidel.
    *   If $1 < \omega < 2$, we "over-correct" the values to reach the solution faster. Choosing the optimal $\omega$ is key to computational efficiency.

The algorithm stops when the total **Energy Functional ($J$)** of the system stabilizes, indicating that the solution has converged:
$$J[V] = \int_{\Omega} \left[ \frac{1}{2} |\nabla V|^2 - \frac{\rho}{\epsilon}V \right] dx dy$$

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