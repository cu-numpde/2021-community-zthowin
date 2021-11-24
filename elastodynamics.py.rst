Elastodynamics verification
===========================

Here we solve an elastodynamics equation using an implicit and explicit
timestepping schemes. This example demonstrates the use of pointwise 
operations on Functions and a time varying Neumann boundary condition. 
The strong form of the equation we set out to solve is:

.. math::

   \rho\mathbf{a} - \nabla\cdot\mathbf{\sigma} = 0

   \mathbf{\sigma} \cdot \mathbf{n} = \mathbf{t}^\sigma = -f(t) \ \textrm{on}\ \Gamma_N

   \mathbf{u}\cdot\mathbf{n} = 0 \ \textrm{on}\ \Gamma_D

   f(t) = \frac{f_0}{2}\left(1-\cos(\omega t)\right)

where we have assumed linear elasticity such that:

.. math::

  \mathbf{\sigma} = \lambda\mathrm{tr}(\mathbf{u}) + 2\mu\epsilon(\mathbf{u})

  \epsilon(\mathbf{u}) = \frac{1}{2}\left(\nabla\mathbf{u} + \left(\nabla\mathbf{u}\right)^T\right)

For the implicit integration scheme, we use the Newmark method and define the predictor:

.. math::

   \mathbf{\tilde{u}}_{n+1} = \mathbf{u}_n + \Delta t\mathbf{v}_n + \frac{\Delta t^2}{2}\left(1 - 2\beta)\mathbf{a}_n

such that the acceleration update may be written in terms of the displacement update as:

.. math::

    \mathbf{a}_{n+1} = \frac{1}{\beta\Delta t^2}\left(\mathbf{u}_{n+1} - \mathbf{\tilde{u}}_n\right)

and the corresponding velocity update as:

.. math::

    \mathbf{v}_{n+1} = \mathbf{v}_n + \Delta t\left[\left(1-\gamma\right)\mathbf{a}_n + \gamma\mathbf{a}_{n+1}\right]

We then write the weak form of the equation for :math:`u`. Find
:math:`u \in V` such that:

.. math::

   \int_\Omega \frac{\rho}{\beta\Delta t^2}\left(\mathbf{u}_{n+1} - \mathbf{\tilde{u}}_n\right)\cdot\mathbf{v} d\Omega + \int_\Omega\mathbf{\sigma}_{n+1}:\epsilon(\mathbf{v}) d\Omega = \int_{\partial\Omega}\mathbf{t}\cdot\mathbf{v} dS
   \quad \forall v \in V

for a suitable function space V.

We may also employ an explicit, central-differencing scheme where :math:`\gamma = 0.5` and :math:`\beta = 0`. The acceleration update is written in terms of the displacements as:

.. math::

    \mathbf{a}_{n+1} = \frac{\mathbf{u}_{n+1} - 2\mathbf{u}_n + \mathbf{u}_{n-1}}{\Delta t^2}

We then write the weak form of the equation for :math:`u`. Find
:math:`u \in V` such that:

.. math::

   \int_\Omega \frac{\rho}{\Delta t^2}\left(\mathbf{u}_{n+1} - 2\mathbf{u}_n + \mathbf{u}_{n-1}\right)\cdot\mathbf{v} d\Omega + \int_\Omega\mathbf{\sigma}_{n+1}:\epsilon(\mathbf{v}) d\Omega = \int_{\partial\Omega}\mathbf{t}\cdot\mathbf{v} dS
   \quad \forall v \in V

for a suitable function space V.

This problem has an analytical solution that was developed in \cite{Eringen}:

.. math::

  u(z,t) = \frac{4}{\pi\rho c}\sum_{n=1}^\infty\frac{(-1)^n}{2n - 1}\left[\int_0^t f(\tau)\sin\left(\frac{(2n-1)\pi c(t-\tau)}{2H}\right)\dtau\right]\sin\left(\frac{(2n-1)\pi z}{2H}\right)

The Python script is started::

    from firedrake import *
    import finat
    import numpy as np
    import matplotlib.pyplot as plt

    EPSILON = 1.0e-14

Define the strain and stress tensors::

    def epsilon(u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)

    def sigma(u):
        return lmbda*div(u)*Identity(3) + 2*mu*epsilon(u)

Select integration type::

    implicit = False
    lump_mass = True

We define Young's modulus and Poisson's ratio::

    E = 50.0e6
    nu = 0.3

Next we calculate Lame's constants and define the density::

    lmbda = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))
    rho = 1986.

The loading parameters are::

    amplitude = 40.0e3
    omega = 50.

We create a rectangular column mesh::

    height = 20.
    width = 1.
    nx = 1
    ny = 1
    nz = 20
    mesh = BoxMesh(nx, ny, nz, width, width, height)

We first compute all surface areas of the mesh and then assign
the differential area element :math: `\gamma` at the top surface::

    ds = Measure('ds', mesh)
    gamma = ds(6)

We choose a degree 1 continuous vector function space, and set up
the function space and functions::

    V = VectorFunctionSpace(mesh, 'Lagrange', 1)
    u = TrialFunction(V)
    w = TestFunction(V)

u_n = Function(V, name='displacement')

if implicit:
    u_bar = Function(V)
    v_n = Function(V)
    a_n = Function(V)
    a_nm1 = Function(V)
else:
    u_np1 = Function(V)
    u_nm1 = Function(V)

Output the initial conditions::

if implicit:
    outfile = File("out-trapezoidal.pvd")
else:
    if lump_mass:
        outfile = File("out-lumped.pvd")
    else:
        outfile = File("out-consistent.pvd")

outfile.write(u_n)

Set time parameters::
T = 0.4
dt = 0.0001
t = 0.
step = 0
Nsteps = int(T/dt)

Initialize solution arrays::
time = np.linspace(0, T, Nsteps+2)
u_top = np.zeros((Nsteps+2,))

Next we define the traction vector which will be updated in the time loop::
traction = Constant((0., 0., 0.))

We next establish the Dirichlet boundary conditions::
fixedSides = Constant(0.)
fixedBottom = as_vector((0.,0.,0.))

fixedLeft_BC_x = DirichletBC(V.sub(0), fixedSides, 1)
fixedLeft_BC_y = DirichletBC(V.sub(1), fixedSides, 1)
fixedRight_BC_x = DirichletBC(V.sub(0), fixedSides, 2)
fixedRight_BC_y = DirichletBC(V.sub(1), fixedSides, 2)
fixedBack_BC_x = DirichletBC(V.sub(0), fixedSides, 3)
fixedBack_BC_y = DirichletBC(V.sub(1), fixedSides, 3)
fixedFront_BC_x = DirichletBC(V.sub(0), fixedSides, 4)
fixedFront_BC_y = DirichletBC(V.sub(1), fixedSides, 4)
fixedBottom_BC = DirichletBC(V, fixedBottom, 5)

bcSet = [fixedLeft_BC_x, fixedRight_BC_x, fixedBack_BC_x, fixedFront_BC_x, \
         fixedLeft_BC_y, fixedRight_BC_y, fixedBack_BC_y, fixedFront_BC_y, \
         fixedBottom_BC]

if implicit:
    Apply an implicit trapezoidal time-stepping scheme::
    F = inner(sigma(u), epsilon(w))*dx + 4*rho/(dt*dt)*dot(u - u_bar, w)*dx - dot(traction, w)*gamma

    a, L = lhs(F), rhs(F)

    Time-stepping loop::
    while t <= T + EPSILON:

        Update load and predictor::
        traction.assign((0., 0., -0.5*amplitude*(1 - cos(omega*t))))
        u_bar.assign(u_n + dt*v_n + 0.25*dt*dt*a_n)

        solve(a == L, u_n, bcSet)

        Update accelerations and velocity for next time step::
        a_nm1.assign(a_n)
        a_n.assign(4/(dt*dt)*(u_n - u_bar))
        v_n.assign(v_n + 0.5*dt*(a_n + a_nm1))

        if step % 10 == 0:
            outfile.write(u_n, time=t)
        u_tip[step+1] = u_n([0., 0., 20.])[2]

        print(t)
        t += dt
        step += 1

else:
    if lump_mass:
        --------------------
        Time loop
        --------------------
        while t <= T + EPSILON:
            traction.assign((0., 0., -0.5*amplitude*(1 - cos(omega*t))))

            Do it like the linear wave equation: 
            u += (assemble(inner(sigma(u), epsilon(w))*dx) - assemble(dot(traction, w)*gamma)) / (assemble(rho/(dt*dt)*w*dx))
            bcSet.apply(u)

            Do it like the higher-order example for mass lumping
            quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, V.ufl_element().degree(), scheme='default')
            dxlump = dx(rule=quad_rule)
            m = (u - 2.0 * u_n + u_nm1) / Constant(dt * dt) * w * dx Fails here

            Do it as suggested in FEniCS documentation: https://comet-fenics.readthedocs.io/en/latest/demo/tips_and_tricks/mass_lumping.html
            First method, second method no longer supported?
            mass_form = Constant(rho/(dt*dt))*w*u*dx Fails here
            mass_action_form = action(mass_form, Constant(1))
            M_lumped = assemble(mass_form)
            M_lumped.zero()
            M_lumped.set_diagonal(assemble(mass_action_form))
            u += assemble(inner(sigma(v), epsilon(w))*dx) / M_lumped
            
            Do it as suggested by wence on issues: https://github.com/firedrakeproject/firedrake/issues/539
            one = Function(V).interpolate(as_vector([1, 1, 1]))
            M_lumped = assemble(action(inner(w, u)*dx, one))
            print("Lumped mass matrix ('lumped' scheme):\n", np.array_str(M_lumped.array(), precision=3))
            inv_lumped = Function(V).assign(1)
            inv_lumped /= M_lumped
            u_n = assemble(inner(sigma(u), epsilon(w))*dx)*inv_lumped*Constant(dt*dt/rho)
            u_n *= inv_lumped Fails here
            
            input()

            if step % 10 == 0:
                outfile.write(u_n, time=t)

            print(t)
            t += dt
            step += 1

    else:
        Apply a central differencing scheme::
        F = inner(sigma(u), epsilon(w))*dx + rho/(dt*dt)*dot(u - 2*u_n- u_nm1, w)*dx - dot(traction, w)*gamma

        a, L = lhs(F), rhs(F)
        
        Time-stepping loop::
        while t <= T + EPSILON:
            Update load::
            traction.assign((0., 0., -0.5*amplitude*(1 - cos(omega*t))))

            solve(a == L, u_np1, bcSet)

            Update solutions for next time step::
            u_nm1.assign(u_n)
            u_n.assign(u_np1)

            if step % 10 == 0:
                outfile.write(u_n, time=t)

            print(t)
            t += dt
            step += 1

plt.plot(time, u_tip)
plt.show()
