###########################################################################################################################################################################################################################
# This FEniCS code implements the classical AT1 phase-field model to solve the 'Surfing problem' to show that fracture propagation follows the Griffith principle.
#
# Input: Set the material properties in lines 17-23. Create a geometry with a crack substantially bigger than Irwin's characteristic length scale. 
#		 Set appropriate surfing boundary conditions in line 68.
#
# Output: Energy release rate, G, is calculated by computing J-integral in a loop around the crack. It is printed in the file 'Surfing.txt'. G is expected to stabilize around Gc.
#		  The crack growth rate can be calculated from visualizing the results in paraview.
#
# Contact Aditya Kumar (akumar355@gatech.edu) for questions.
###########################################################################################################################################################################################################################


from dolfin import *
import numpy as np
import time

# Material properties
E, nu = 9800, 0.13	#Young's modulus and Poisson's ratio
Gc= 0.091125	#Critical energy release rate
sts, scs= 27, 77	#Tensile strength and compressive strength
#Irwin characteristic length
lch=3*Gc*E/8/(sts**2)
#The regularization length
eps=0.35  #epsilon should not be chosen to be too large compared to lch. Typically eps<4*lch should work

delta=1.16 #not needed for the classical propagation model

comm = MPI.comm_world 
comm_rank = MPI.rank(comm)

# Create mesh and define function space
mesh=Mesh("Ss_eps35_h_35e-3.xml") 
h=FacetArea(mesh)          #area/length of a cell facet on a given mesh
h_avg = (h('+') + h('-'))/2
n=FacetNormal(mesh)

# Choose phase-field model
phase_model=1;  #1 for linear model, 2 for quadratic model

V = VectorFunctionSpace(mesh, "CG", 1)   #Function space for u
Y = FunctionSpace(mesh, "CG", 1)         #Function space for z

# Mark boundary subdomians
left =  CompiledSubDomain("near(x[0], side, tol) && abs(x[1]) > tol && on_boundary", side = 0.0, tol=1e-4)
right =  CompiledSubDomain("near(x[0], side, tol) && on_boundary", side = 30.0, tol=1e-4)  #30.0
bottom =  CompiledSubDomain("near(x[1], side, tol) && on_boundary", side = -5.0, tol=1e-4)  #5.0
top =  CompiledSubDomain("near(x[1], side, tol) && on_boundary", side = 5.0, tol=1e-4)
righttop = CompiledSubDomain("abs(x[0]-30.0)<1e-4 && abs(x[1]-5.0)<1e-4 ")
corner = CompiledSubDomain("abs(x[1]-0.0)<1e-4 && x[0]<5+1e-4 && x[0]>5.5")
	

#############################################################	
set_log_level(40)  #Error level=40, warning level=30
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}	
############################################################
	
##################################################################################
# Define Dirichlet boundary conditions
##################################################################################

c=Expression("K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*cos(atan2(x[1],(x[0]-V*(t+0.1)))/2)",degree=4,t=0, V=20, K1=30, mu=4336.28, kap=2.54)
r=Expression("K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*sin(atan2(x[1],(x[0]-V*(t+0.1)))/2)",degree=4,t=0, V=20, K1=30, mu=4336.28, kap=2.54)

c0=Expression("(K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*cos(atan2(x[1],(x[0]-V*(t+0.1)))/2))-(K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(tau+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(tau+0.1)))))*cos(atan2(x[1],(x[0]-V*(tau+0.1)))/2))",degree=4,t=0, tau=0, V=20, K1=30, mu=4336.28, kap=2.54)
r0=Expression("(K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(t+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(t+0.1)))))*sin(atan2(x[1],(x[0]-V*(t+0.1)))/2))-(K1/(2*mu)*sqrt(sqrt(pow(x[0]-V*(tau+0.1),2)+pow(x[1],2))/(2*pi))*(kap-cos(atan2(x[1],(x[0]-V*(tau+0.1)))))*sin(atan2(x[1],(x[0]-V*(tau+0.1)))/2))",degree=4,t=0, tau=0, V=20, K1=30, mu=4336.28, kap=2.54)
								
bcl= DirichletBC(V.sub(0), Constant(0.0), righttop, method='pointwise'  )
bcb2 = DirichletBC(V.sub(1), r, bottom )
bct2 = DirichletBC(V.sub(1), r, top)
bcs = [ bcl, bcb2,  bct2]

cz=Constant(1.0)
bcb_z = DirichletBC(Y, cz, bottom)
bct_z = DirichletBC(Y, cz, top)
cz2=Constant(0.0)
bcc_z = DirichletBC(Y, cz2, corner)
bcs_z=[ bcb_z, bct_z, bcc_z]


bcb_du=DirichletBC(V.sub(1),Constant(0.0), bottom )
bct_du=DirichletBC(V.sub(1),Constant(0.0),top)
bcs_du = [bcl, bcb_du, bct_du]


bcb2_du0 = DirichletBC(V.sub(1), r0, bottom )
bct2_du0 = DirichletBC(V.sub(1), r0, top)
bcs_du0 = [ bcl, bcb2_du0, bct2_du0]


d_dz=Constant(0.0)
bcb_dz = DirichletBC(Y, d_dz, bottom)
bct_dz = DirichletBC(Y, d_dz, top)
bcc_dz = DirichletBC(Y, d_dz, corner)
bcs_dz=[ bcb_dz, bct_dz, bcc_dz] 



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
Tf  = Expression(("t*0.0", "t*0"),degree=1,t=0)  # Traction force on the boundary


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
top_dofs=extract_dofs_boundary(V,top)
y_dofs_top=top_dofs[1::d]


boundary_subdomains = MeshFunction("size_t", mesh, 1)
boundary_subdomains.set_all(0)
left.mark(boundary_subdomains,1)
right.mark(boundary_subdomains,1)
bottom.mark(boundary_subdomains,2)
top.mark(boundary_subdomains,2)

# Define new measures associated with the interior domains 	
ds = ds(subdomain_data=boundary_subdomains)


# Elasticity parameters
mu, lmbda, kappa = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu))), Constant(E/(3*(1 - 2*nu)))

def energy(v):
	return mu*(inner(sym(grad(v)),sym(grad(v))) + ((nu/(1-nu))**2)*(tr(sym(grad(v))))**2 )+  0.5*(lmbda)*(tr(sym(grad(v)))*(1-2*nu)/(1-nu))**2 
	
def epsilon(v):
	return sym(grad(v))

def sigma(v):
	return 2.0*mu*sym(grad(v)) + (lmbda)*tr(sym(grad(v)))*(1-2*nu)/(1-nu)*Identity(len(v))

def sigmavm(sig,v):
	return sqrt(1/2*(inner(sig-1/3*tr(sig)*Identity(len(v)), sig-1/3*tr(sig)*Identity(len(v))) + (1/9)*tr(sig)**2 ))

############################################################################################################################
##configurational external force to model strength - not included in the classical phase-field model for crack propagation
beta0=-3*Gc/8/eps*delta	
beta3=0
beta1= ((9*E*Gc)/2. - (9*E*Gc*sts)/(2.*scs) - (9*beta3*E*Gc*scs*sts)/2. + (9*beta3*E*Gc*sts**2)/2.)/(24.*E*eps*sts) +  (-12*beta0*E + (12*beta0*E*sts)/scs + 12*scs*sts + 12*beta3*scs**3*sts - 12*sts**2 - 12*beta3*sts**4)/(24.*E*sts)
beta2= (9*E*Gc*scs + 9*E*Gc*sts + 9*beta3*E*Gc*scs**2*sts + 9*beta3*E*Gc*scs*sts**2)/(16.*sqrt(3)*E*eps*scs*sts) +  (-24*beta0*E*scs - 24*beta0*E*sts - 24*scs**2*sts - 24*beta3*scs**4*sts - 24*scs*sts**2 - 24*beta3*scs*sts**4)/(16.*sqrt(3)*E*scs*sts)

ce= (beta1*(z**2)*(tr(sigma(u))) + beta2*(z**2)*(sigmavm(sigma(u),u)) +beta0)/(1+beta3*(z**4)*(tr(sigma(u)))**2)
############################################################################################################################

pen=1000*conditional(lt(-beta0,Gc/eps),Gc/eps, -beta0)
eta=1e-8	
# Stored strain energy density (compressible L-P model)
psi1 =(z**2+eta)*(energy(u))	
psi11=energy(u)
stress=(z**2+eta)*sigma(u)

# Total potential energy
Pi = psi1*dx

# Compute first variation of Pi (directional derivative about u in the direction of v)
R = derivative(Pi, u, v) 

# Compute Jacobian of R
Jac = derivative(R, u, du) 

#To use later for memory allocation for these tensors
A=PETScMatrix()
b=PETScVector()

#Balance of configurational forces PDE
Wv=pen/2*((abs(z)-z)**2 + (abs(1-z) - (1-z))**2 )*dx
Wv2=conditional(le(z, 0.1), 1, 0)*40*pen/2*( 1/4*( abs(z_prev-z)-(z_prev-z) )**2 )*dx
if phase_model==1:
	R_z = y*2*z*(psi11)*dx +3*Gc/8*(y*(-1)/eps + 2*eps*inner(grad(z),grad(y)))*dx + derivative(Wv,z,y) + derivative(Wv2,z,y) #+ y*(ce)*dx #linear model
else:
	R_z = y*2*z*(psi11)*dx+ Gc*(y*(z-1)/eps + eps*inner(grad(z),grad(y)))*dx + derivative(Wv2,z,y) # + y*(ce)*dx  #quadratic model
	
# Compute Jacobian of R_z
Jac_z = derivative(R_z, z, dz)



## Define the solver parameters
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "cg",   
                                          "preconditioner": "amg",						  
                                          "maximum_iterations": 10,
                                          "report": True,
                                          "error_on_nonconvergence": False}}			


#time-stepping parameters
T=2

Totalsteps=100
startstepsize=1/Totalsteps
stepsize=startstepsize
t=stepsize
step=1
rtol=1e-9
printsteps=5 
tau=0

start_time=time.time()
while t-stepsize < T:

	if comm_rank==0:
		print('Step= %d' %step, 't= %f' %t, 'Stepsize= %e' %stepsize)
	
	c.t=t; c0.t=t; r.t=t; r0.t=t; c0.tau=tau; r0.tau=tau
	
	stag_iter=1
	rnorm_stag=1
	while stag_iter<200 and rnorm_stag > 1e-8:
		start_time=time.time()
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
		##############################################################
		
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
		if comm_rank==0:
			print('Stag Iteration no= %d' %stag_iter,  'Residual= %e' %rnorm_stag)
		stag_iter+=1  

	
	######################################################################
	#Post-Processing
	######################################################################
	assign(u_prev,u)
	assign(z_prev,z)
	
	tau+=stepsize
	
	####Calculate Reaction
	Fx=MPI.sum(comm,sum(fint[y_dofs_top]))
	surfenergy = 3/8*((1-z)/lch + lch*inner(grad(z),grad(z)))*dx(0)
	SE = assemble(surfenergy)
	
	JI1=(psi1-dot(dot(stress,n),u.dx(0)))*ds(1)
	JI2=(-dot(dot(stress,n),u.dx(0)))*ds(2)
	Jintegral=assemble(JI1)+assemble(JI2)

	if comm_rank==0:
		print(Fx)
		with open('SurfingAT1d.txt', 'a') as rfile:
			rfile.write("%s %s\n" % (str(t), str(Jintegral)))
		

	####Plot solution on incremental steps
	if step % printsteps==0:
		file_results = XDMFFile( "classical/SurfingAT1d_" + str(step) + ".xdmf" )
		file_results.parameters["flush_output"] = True
		file_results.parameters["functions_share_mesh"] = True
		u.rename("u", "displacement field")
		z.rename("z", "phase field")
		file_results.write(u,t)
		file_results.write(z,t)
	
	#time stepping
	step+=1
	t+=stepsize
 
#######################################################end of all loops
if comm_rank==0:	
	print("--- %s seconds ---" % (time.time() - start_time))

