###########################################################################################################################################################################################################################
# This FEniCS code implements the new phase-field model introduced in  Kumar, A., Bourdin, B., Ravi-Chandar, K. G.A. and Lopez-Pamies, O., 2022.
# This new phase-field model is used to solve the 'Cylindrical Indentation test' in Glass for different size of the indenter.
# 
#
# Input: Set the material properties in lines 35-40. 
#		 
#
# Output: Output files can be read with paraview to visualize crack nucleation and propagation
#
# Contact Aditya Kumar (akumar355@gatech.edu) for questions.
###########################################################################################################################################################################################################################


from dolfin import *
import numpy as np
import time


set_log_level(40)  #Error level=40, warning level=30
parameters["linear_algebra_backend"] = "PETSc"
parameters["ghost_mode"] = "shared_facet"


parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

comm = MPI.comm_world 
comm_rank = MPI.rank(comm)

# Elasticity parameters
E, nu = 80000, 0.22
mu, lmbda, kappa = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu))), Constant(E/(3*(1 - 2*nu)))
# Fracture parameters
k, eta1, eta2 = 0.009, 0.0, 0.0     
sts=60
scs=1000


boundary_layer=0.104#0.051
mesh=RectangleMesh(comm, Point(0.0,0.0), Point(25.0,-12.5), 130, 65, "crossed") 


domain1 = CompiledSubDomain("x[1]>-2.5 && x[0]<5.0")
ir=0
while ir<2:
	d_markers = MeshFunction("bool", mesh, 2, False)
	domain1.mark(d_markers, True)
	mesh = refine(mesh,d_markers, True)
	ir+=1

domain2 = CompiledSubDomain("x[1]>-1.25 && x[0]<3.5")
ir=0
while ir<2:
	d_markers = MeshFunction("bool", mesh, 2, False)
	domain2.mark(d_markers, True)
	mesh = refine(mesh,d_markers, True)
	ir+=1

domain3 = CompiledSubDomain("x[1]>-0.4 && x[0]<2.25")
ir=0
while ir<3:
	d_markers = MeshFunction("bool", mesh, 2, False)
	domain3.mark(d_markers, True)
	mesh = refine(mesh,d_markers, True)
	ir+=1

domain4 = CompiledSubDomain("x[1]>-1.0 && x[1]<-0.4+1e-4 && x[0]<3.25 && x[0]>1.5")
ir=0
while ir<3:
	d_markers = MeshFunction("bool", mesh, 2, False)
	domain4.mark(d_markers, True)
	mesh = refine(mesh,d_markers, True)
	ir+=1

h=FacetArea(mesh)          #area/length of a cell facet on a given mesh
h_avg = (h('+') + h('-'))/2

# Choose phase-field model
phase_model=1;  #1 for linear model, 2 for quadratic model



V = VectorFunctionSpace(mesh, "CG", 1)   #Function space for u
Y = FunctionSpace(mesh, "CG", 1)         #Function space for z

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = 0.0, tol=1e-4)
front =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = 25.0, tol=1e-4)
bottom =  CompiledSubDomain("near(x[1], side, tol) && on_boundary", side = -12.5, tol=1e-4)
top =  CompiledSubDomain("near(x[1], side, tol) && on_boundary", side = 0.0, tol=1e-4)

def loadset(x):
	return x[0]>0-1e-4 and x[0]<1.0+1e-4 and abs(x[1]-0)<1e-4 

def outer(x):
	return x[0]>3.5 or x[1]<-2.0

def loadsetz(x):
	return x[0]>0-1e-4 and x[0]<-0.55+1e-4 #and abs(x[1]-0)<1e-4
	
##################################################################################
# Define Dirichlet boundary (x = 0 or x = 1)
##################################################################################
c=Expression("t*0.0",degree=1,t=0)
r=Expression("-t*0.05",degree=1,t=0)
r0=Expression("-(t-tau)*0.05",degree=1,t=0,tau=0)
								
bcl = DirichletBC(V.sub(0), c, left )
bcb = DirichletBC(V.sub(1), c, bottom )
bct = DirichletBC(V.sub(1), r, loadset)
bcs = [bcl, bcb, bct]

cz=Constant(1.0)
bct_z = DirichletBC(Y, cz, outer)
bct_z2 = DirichletBC(Y, cz, loadsetz)
cz2=Constant(0.0)
bcs_z=[bct_z, bct_z2]

bct_du=DirichletBC(V.sub(1),Constant(0.0),loadset)
bct_du0=DirichletBC(V.sub(1), r0, loadset)
bcs_du = [bcl, bcb, bct_du]
bcs_du0 = [bcl, bcb, bct_du0]

d_dz=Constant(0.0)
bct_dz = DirichletBC(Y, d_dz, outer)
bct_dz2 = DirichletBC(Y, d_dz, loadsetz)
bcs_dz=[bct_dz, bct_dz2]


########################################################################
# Define functions
########################################################################
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
u_inc = Function(V)
dz = TrialFunction(Y)            # Incremental phase field
y  = TestFunction(Y)             # Test function
z  = Function(Y)                 # Phase field from previous iteration
z_inc = Function(Y)
d = u.geometric_dimension()
B  = Constant((0.0, 0.0))  # Body force per unit volume
T  = Constant((0.0,  0.0))  # Traction force on the boundary


##############################################################
#Initialisation of displacement field,u and the phase field,z
##############################################################
u_init = Constant((0.0,  0.0))
u.interpolate(u_init)
for bc in bcs:
	bc.apply(u.vector())

z_init = Constant(1.0)
z.interpolate(z_init)
for bc in bcs_z:
	bc.apply(z.vector())

z_ub = Function(Y)
z_ub.interpolate(Constant(1.0))	
z_lb = Function(Y)
z_lb.interpolate(Constant(-0.0))
	

u_prev = Function(V)
assign(u_prev,u)
z_prev = Function(Y)
assign(z_prev,z)
	
#################################################
###Label the dofs on boundary
#################################################
def extract_dofs_boundary(V, bsubd):	
	label = Function(V)
	label_bc_bsubd = DirichletBC(V, Constant((1,1)), bsubd)
	label_bc_bsubd.apply(label.vector())
	bsubd_dofs = np.where(label.vector()==1)[0]
	return bsubd_dofs

#Dofs on which reaction is calculated
top_dofs=extract_dofs_boundary(V,loadset)
y_dofs_top=top_dofs[1::d]

#subdomains
tol = 1E-7	
xm = SpatialCoordinate(mesh)

	
def eig_plus(A):
	return (((tr(A) + sqrt(abs(tr(A)**2-4*det(A))))/2))

def eig_minus(A):
	return (((tr(A) - sqrt(abs(tr(A)**2-4*det(A))))/2))




l_0, l_1=0.026, 0.0075
lch = Expression('x[1] >= -boundary_layer + tol ? l_0 : l_1', degree=1,
               tol=tol, l_0=l_0, l_1=l_1, boundary_layer=boundary_layer)


beta0_0, beta0_1=-1.85 , -0.34
beta0 = Expression('x[1] >= -boundary_layer + tol ? beta0_0 : beta0_1', degree=1,
               tol=tol, beta0_0=beta0_0, beta0_1=beta0_1, boundary_layer=boundary_layer)

beta1_0, beta1_1=0.0151335, 0.00130083
beta1 = Expression('x[1] >= -boundary_layer + tol ? beta1_0 : beta1_1', degree=1,
               tol=tol, beta1_0=beta1_0, beta1_1=beta1_1, boundary_layer=boundary_layer)

beta2_0, beta2_1=0.0171133, 0.00209083
beta2 = Expression('x[1] >= -boundary_layer + tol ? beta2_0 : beta2_1', degree=1,
               tol=tol, beta2_0=beta2_0, beta2_1=beta2_1, boundary_layer=boundary_layer)


pen=1000*(1.0)

def eps(v):
	return sym(as_tensor([[v[0].dx(0), v[0].dx(1),0],
                          [v[1].dx(0), v[1].dx(1), 0],
                          [0, 0, v[0]/xm[0]]]))


def ll1(v):
	return eig_plus(sym(grad(v)))

def ll2(v):
	return eig_minus(sym(grad(v)))

def ll3(v):
	return v[0]/xm[0]

def energy(v):
	return mu*(inner(eps(v),eps(v)))+  0.5*(lmbda)*(tr(eps(v)))**2 


def sigma(v):
	return 2.0*mu*eps(v) + (lmbda)*tr(eps(v))*Identity(3)	
	

def sigmavm(sig,v):
	return sqrt(3/2*(inner(sig-1/3*(tr(sig))*Identity(3), sig-1/3*(tr(sig))*Identity(3)) ))
	
	
# Stored strain energy density (compressible L-P model)
psi1 =(z**2)*energy(u)

psi11=energy(u)


ce= (beta1*(z**2)*(tr(sigma(u))) + beta2*(z**2)*(sigmavm(sigma(u),u)) +beta0) - (1- abs(tr(sigma(u)))/tr(sigma(u)))*z*((sigmavm(sigma(u),u)**2)/(2*mu) + (tr(sigma(u))**2)/(18*kappa))

 
# Total potential energy
Pi = psi1*2*np.pi*xm[0]*dx

# Compute first variation of Pi (directional derivative about u in the direction of v)
R = derivative(Pi, u, v) 


# Compute Jacobian of R
Jac = derivative(R1, u, du)

A=PETScMatrix()
b=PETScVector()

#Balance of configurational forces PDE
Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*2*np.pi*xm[0]*dx
Wv2=conditional(le(z, 0.1), 1, 0)*40*pen/2*( 1/4*( abs(z_prev-z)-(z_prev-z) )**2 )*dx
if phase_model==1:
	R_z = y*(psi11)*2*np.pi*xm[0]*dx+ y*(ce)*2*np.pi*xm[0]*dx+  3*k/8*(y*(-1)/lch + 2*lch*inner(grad(z),grad(y)))*2*np.pi*xm[0]*dx + derivative(Wv,z,y) # #linear model y*(ce)*2*np.pi*xm[0]*dx+
else:
	R_z = y*2*z*(psi11+ce)*dx(0)+ k*(y*(z-1)/lch + lch*inner(grad(z),grad(y)))*dx(0) #+ derivative(Wv2,z,y)  #quadratic model
	
# Compute Jacobian of R_z
Jac_z = derivative(R_z, z, dz)



# Define the solver parameters
solver_u = KrylovSolver('cg', 'amg')
max_iterations=450
solver_u.parameters["maximum_iterations"] = max_iterations
solver_u.parameters["error_on_nonconvergence"] = False

solver_u2 = KrylovSolver('cg', 'petsc_amg')
max_iterations=450
solver_u2.parameters["maximum_iterations"] = max_iterations
solver_u2.parameters["error_on_nonconvergence"] = False


solver_u3 = KrylovSolver('cg', 'jacobi')
max_iterations=2000
solver_u3.parameters["maximum_iterations"] = max_iterations
solver_u3.parameters["error_on_nonconvergence"] = False

solver_z = KrylovSolver('cg', 'amg') #'hypre_amg'
max_iterations=450
solver_z.parameters["maximum_iterations"] = max_iterations						
solver_z.parameters["error_on_nonconvergence"] = False
							
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "cg",   #lu or gmres or cg 'preconditioner: ilu, amg, jacobi'
                                          "preconditioner": "petsc_amg",						  
                                          "maximum_iterations": 10,
                                          "report": True,
                                          "error_on_nonconvergence": False}}			


#time-stepping parameters
T=1
Totalsteps=10000
minstepsize=1/Totalsteps/10000
maxstepsize=1/Totalsteps*20
startstepsize=1/Totalsteps
stepsize=startstepsize
t=stepsize
step=1
samesizecount=1
#other time stepping parameters
terminate=0
terminate2=0
gfcount=0
nrcount=0
printsteps=10  #Number of incremental steps after which solution will be stored
rtol=1e-9


u_inc1 = Function(V)
tau=0
# u_proj = Function(V)


start_time=time.time()
# Solve variational problem
while t-stepsize < T:	
	
	
	if comm_rank==0:
		print('Step= %d' %step, 'Omega= %e' %omega, 'Stepsize= %e' %stepsize)
	c.t=t; r.t=t; r0.t=t; r0.tau=tau
	
	stag_iter=1
	rnorm_stag=1
	while stag_iter<200 and rnorm_stag > 1e-9:
		##############################################################
		#First PDE
		##############################################################		
		Problem_u = NonlinearVariationalProblem(R, u, bcs, J=Jac)
		solver_u  = NonlinearVariationalSolver(Problem_u)
		solver_u.parameters.update(snes_solver_parameters)
		(iter, converged) = solver_u.solve()
			
		##############################################################
		#Second PDE
		##############################################################
		Problem_z = NonlinearVariationalProblem(R_z, z, bcs_z, J=Jac_z)
		solver_z  = NonlinearVariationalSolver(Problem_z)
		solver_z.parameters.update(snes_solver_parameters)
		(iter, converged) = solver_z.solve()
			
		min_z = z.vector().min();
		zmin = MPI.min(comm, min_z)
		if comm_rank==0:
			print(zmin)
		
		if comm_rank==0:
			print("--- %s seconds ---" % (time.time() - start_time))

	  
		###############################################################
		#Residual check for stag loop
		###############################################################
		b=assemble(-R, tensor=b)
		fint=b.copy() #assign(fint,b) 
		for bc in bcs_du:
			bc.apply(b)
		rnorm_stag=b.norm('l2')	
		stag_iter+=1  

		
	######################################################################
	#Post-Processing
	if terminate==1:
		assign(u,u_prev)
		assign(z,z_prev)
	else:
		assign(u_prev,u)
		assign(z_prev,z)
		
		
		tau+=stepsize
		####Calculate Reaction
		Fx=MPI.sum(comm,sum(fint[y_dofs_top]))
		if comm_rank==0:
			print(Fx)
			with open('Indent.txt', 'a') as rfile:
				rfile.write("%s %s %s\n" % (str(t), str(Fx), str(zmin)))
	
		####Plot solution on incremental steps
		if step % printsteps==0:
			file_results = XDMFFile( "/fenics/Indent_" + str(step) + ".xdmf" )
			file_results.parameters["flush_output"] = True
			file_results.parameters["functions_share_mesh"] = True
			u.rename("u", "displacement field")
			z.rename("z", "phase field")
			file_results.write(u,step)
			file_results.write(z,step)
			
	
	#time stepping
	if terminate==1:
		if stepsize>minstepsize:
			t-=stepsize
			stepsize/=2
			t+=stepsize
			samesizecount=1
		else:
			break
	else:
		if samesizecount<2:
			step+=1
			if t+stepsize<=T:
				samesizecount+=1
				t+=stepsize
			else:
				samesizecount=1
				stepsize=T-t
				t+=stepsize
				
		else:
			step+=1
			if stepsize*2<=maxstepsize and t+stepsize*2<=T:	
				stepsize*=2
				t+=stepsize
			elif stepsize*2>maxstepsize and t+maxstepsize<=T:
				stepsize=maxstepsize
				t+=stepsize
			else:
				stepsize=T-t
				t+=stepsize
			samesizecount=1

			
		
	 
#######################################################end of all loops

