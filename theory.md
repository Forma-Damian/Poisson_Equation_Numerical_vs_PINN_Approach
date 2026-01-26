

# Introduction: Solving the 2D Poisson Equation

## 1. Physical and Mathematical Context

The **Poisson equation** is a fundamental partial differential equation (PDE) used across physics and engineering to describe how a field (like electric potential or temperature) behaves in the presence of a source (like electric charge or heat).

In two dimensions, the equation for electric potential  is:

$$\frac{\partial^{2}V}{\partial x^{2}} + \frac{\partial^{2}V}{\partial y^{2}} = -\frac{\rho(x,y)}{\epsilon}$$

Where:

* **(The Laplacian):** Represents the "curvature" or spatial second derivative of the potential.
* **:** The charge density (the source). In this project, we use a specific distribution: .
* **:** The permittivity of the medium (assumed to be 1.0 for simplicity).

If the source term  is zero, the equation is known as the **Laplace equation**, which describes the potential in a region "empty" of charges.

## 2. Boundary Conditions (BCs)

To find a unique solution within a square domain , we must define what happens at the edges:

1. **Dirichlet Boundary Conditions:** We explicitly set the value of the potential  at the edge. Here, we use sinusoidal functions (controlled by parameters  through ) to simulate varying potentials along the four walls.
2. **Neumann Boundary Conditions:** We set the *gradient* (slope) of the potential. For example, setting  on the right boundary simulates an insulating or reflective boundary.

## 3. The Numerical Approach: Successive Over-Relaxation (SOR)

Since PDEs are difficult to solve analytically for complex sources, we use numerical methods. We "discretize" the space into a grid of  points.

The **Successive Over-Relaxation (SOR)** method is an iterative algorithm used to solve these grid-based equations. It improves upon the basic Gauss-Seidel method by using a relaxation factor, :

* **Update Rule:** Each node  is updated by looking at its four neighbors.
* **The  Parameter:** * If , the method is standard.
* If , we "over-correct" the values to reach the solution faster. Choosing the optimal  is key to computational efficiency.



The algorithm stops when the total **Energy Functional ()** of the system stabilizes, indicating that the solution has converged.

## 4. Physics-Informed Neural Networks (PINNs) vs. Numerical Methods

This project compares the traditional **SOR method** against a modern machine learning approach: **Physics-Informed Neural Networks (PINNs)**.

| Feature | Numerical (SOR) | PINNs (Deep Learning) |
| --- | --- | --- |
| **Space** | Discrete grid (Nodes) | Continuous domain |
| **Mechanism** | Iterative local updates | Loss function minimization via Backpropagation |
| **Physics** | Hard-coded via finite differences | Embedded in the Neural Network's Loss Function |
| **Efficiency** | Very fast for simple 2D grids | Slower training, but mesh-free and scalable |

By comparing the **Potential ** and the **Residual ** (how much the solution fails to satisfy the original equation), we can determine the accuracy and reliability of neural networks in solving classical electrostatic problems compared to time-tested numerical solvers.
