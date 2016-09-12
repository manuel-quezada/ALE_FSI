#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <fstream>
#include <iostream>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/base/std_cxx1x/shared_ptr.h>
#include "LevelSetSolver.cc"

using namespace dealii;

#define MAX_NUM_ITER_TO_RECOMPUTE_PRECONDITIONER 10

/////////////////////////////////////////////////////////////////
///////////////////// NAVIER STOKES SOLVER //////////////////////
/////////////////////////////////////////////////////////////////
template<int dim>
class NavierStokesSolver {
public:
  // constructor for using LEVEL SET
  NavierStokesSolver(const unsigned int degree_LS, 
		     const unsigned int degree_U,
		     const double time_step, 
		     const double eps, 
		     const double rho_air, 
		     const double nu_air,
		     const double rho_fluid, 
		     const double nu_fluid, 
		     Function<dim> &force_function,
		     const bool verbose, 
		     parallel::distributed::Triangulation<dim> &triangulation, 
		     MappingQ<dim> &mapping_tnp1, 
		     MappingQ<dim> &mapping_tn, 
		     MappingQ<dim> &mapping_tnm1,
		     MPI_Comm &mpi_communicator);
  // constructor for NOT LEVEL SET
  NavierStokesSolver(const unsigned int degree_LS, 
		     const unsigned int degree_U,
		     const double time_step, 
		     Function<dim> &force_function, 
		     Function<dim> &rho_function,
		     Function<dim> &nu_function, 
		     const bool verbose,
		     parallel::distributed::Triangulation<dim> &triangulation, 
		     MappingQ<dim> &mapping_tnp1,
		     MappingQ<dim> &mapping_tn, 
		     MappingQ<dim> &mapping_tnm1,
		     MPI_Comm &mpi_communicator);

  void set_coord(Vector<double> xn_coord, Vector<double> yn_coord,
		 Vector<double> xnp1_coord, Vector<double> ynp1_coord);		 
  void set_mesh_velocity(PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_un,
			 PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_vn);
  void set_mesh_velocity(PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_un,
			 PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_vn,
			 PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_wn);
  // rho and nu functions
  void set_rho_and_nu_functions(const Function<dim> &rho_function,
				const Function<dim> &nu_function);
  //initial conditions
  void initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_rho,
			 PETScWrappers::MPI::Vector locally_relevant_solution_u,
			 PETScWrappers::MPI::Vector locally_relevant_solution_v,
			 PETScWrappers::MPI::Vector locally_relevant_solution_p);
  void initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_rho,
			 PETScWrappers::MPI::Vector locally_relevant_solution_u,
			 PETScWrappers::MPI::Vector locally_relevant_solution_v,
			 PETScWrappers::MPI::Vector locally_relevant_solution_w,
			 PETScWrappers::MPI::Vector locally_relevant_solution_p);
  //boundary conditions
  void set_boundary_conditions(std::vector<unsigned int> boundary_values_id_u,
			       std::vector<unsigned int> boundary_values_id_v, std::vector<double> boundary_values_u,
			       std::vector<double> boundary_values_v);
  void set_boundary_conditions(std::vector<unsigned int> boundary_values_id_u,
			       std::vector<unsigned int> boundary_values_id_v,
			       std::vector<unsigned int> boundary_values_id_w, std::vector<double> boundary_values_u,
			       std::vector<double> boundary_values_v, std::vector<double> boundary_values_w);
  void set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_u,
		    PETScWrappers::MPI::Vector locally_relevant_solution_v);
  void set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_u,
		    PETScWrappers::MPI::Vector locally_relevant_solution_v,
		    PETScWrappers::MPI::Vector locally_relevant_solution_w);
  void set_phi(PETScWrappers::MPI::Vector locally_relevant_solution_phi);
  void get_pressure(PETScWrappers::MPI::Vector &locally_relevant_solution_p);
  void get_velocity(PETScWrappers::MPI::Vector &locally_relevant_solution_u,
		    PETScWrappers::MPI::Vector &locally_relevant_solution_v);
  void get_velocity(PETScWrappers::MPI::Vector &locally_relevant_solution_u,
		    PETScWrappers::MPI::Vector &locally_relevant_solution_v,
		    PETScWrappers::MPI::Vector &locally_relevant_solution_w);
  // DO STEPS //
  void nth_time_step(double time_step);
  // SETUP //
  void setup();

  ~NavierStokesSolver();

private:
  // SETUP AND INITIAL CONDITION //
  void setup_DOF();
  void setup_VECTORS();
  void init_constraints();
  // ASSEMBLE SYSTEMS //
  // NON-INCREMENTAL BDF 1
  void assemble_system_U_BDF1();
  void assemble_system_dpsi_q_BDF1();
  void assemble_extend_pressure_TMP();
  void assemble_system_U_BDF2();
  void assemble_system_dpsi_q_BDF2();
  void get_differentiable_mesh_velocity_at_tnp1(double dt, double t);
  void get_differentiable_mesh_velocity_at_tnp1_v2(double dt, double t);
  void get_non_differentiable_mesh_velocity_at_tnp1(double dt, double t);
  // SOLVERS //
  void solve_U(const ConstraintMatrix &constraints, PETScWrappers::MPI::SparseMatrix &Matrix,
	       std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
	       PETScWrappers::MPI::Vector &completely_distributed_solution,
	       const PETScWrappers::MPI::Vector &rhs);
  void solve_P(const ConstraintMatrix &constraints, PETScWrappers::MPI::SparseMatrix &Matrix,
	       std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
	       PETScWrappers::MPI::Vector &completely_distributed_solution,
	       const PETScWrappers::MPI::Vector &rhs);
  // GET DIFFERENT FIELDS //
  void get_rho_and_nu(double phi);
  void get_rho(double &rho, double phi);
  void get_rho_and_nu(double &rho, double &mu, double phi);
  void get_grad_and_value_rho(Tensor<1,dim> &grad_rho, double &rho, 
			      Tensor<1,dim> grad_phi, double phi);
  void get_velocity();
  void get_pressure();
  // OTHERS //
  void save_old_solution();

  MPI_Comm &mpi_communicator;

  // MAPPINGS //
  MappingQ<dim> &mapping_tnp1;
  MappingQ<dim> &mapping_tn;
  MappingQ<dim> &mapping_tnm1;

  parallel::distributed::Triangulation<dim> &triangulation;

  int degree_LS;
  DoFHandler<dim> dof_handler_LS;
  FE_Q<dim> fe_LS;
  IndexSet locally_owned_dofs_LS;
  IndexSet locally_relevant_dofs_LS;

  int degree_U;
  DoFHandler<dim> dof_handler_U;
  FE_Q<dim> fe_U;
  IndexSet locally_owned_dofs_U;
  IndexSet locally_relevant_dofs_U;

  DoFHandler<dim> dof_handler_P;
  FE_Q<dim> fe_P;
  IndexSet locally_owned_dofs_P;
  IndexSet locally_relevant_dofs_P;

  DoFHandler<dim>      dof_handler_U_mesh;
  FE_Q<dim>            fe_U_mesh;
  IndexSet             locally_owned_dofs_U_mesh;
  IndexSet             locally_relevant_dofs_U_mesh;

  Function<dim> &force_function;
  Function<dim> &rho_function;
  Function<dim> &nu_function;

  double rho_air;
  double nu_air;
  double rho_fluid;
  double nu_fluid;

  double time_tnm1;
  double time_tn;
  double time_tnp1;
  double time_step;
  double eps;

  bool verbose;
  unsigned int LEVEL_SET;

  ConditionalOStream pcout;

  LevelSetSolver<dim> transport_pressure;

  double rho_min;
  double rho_value;
  double nu_value;

  double h;
  double umax;

  int degree_MAX;

  ConstraintMatrix constraints;
  ConstraintMatrix constraints_q;
  ConstraintMatrix constraints_psi;

  std::vector<unsigned int> boundary_values_id_u;
  std::vector<unsigned int> boundary_values_id_v;
  std::vector<unsigned int> boundary_values_id_w;
  std::vector<double> boundary_values_u;
  std::vector<double> boundary_values_v;
  std::vector<double> boundary_values_w;

  PETScWrappers::MPI::SparseMatrix system_Matrix_u;
  PETScWrappers::MPI::SparseMatrix system_Matrix_v;
  PETScWrappers::MPI::SparseMatrix system_Matrix_w;
  bool rebuild_Matrix_U;
  std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_Matrix_u;
  std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_Matrix_v;
  std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_Matrix_w;
  PETScWrappers::MPI::SparseMatrix system_S;
  std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_S;
  PETScWrappers::MPI::SparseMatrix system_M;
  std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner_M;
  bool rebuild_S_M;
  bool rebuild_Matrix_U_preconditioners;
  bool rebuild_S_M_preconditioners;
  // MESH VELOCITY VECTORS
  // coord
  Vector<double> xn_coord;
  Vector<double> yn_coord;  
  Vector<double> xnp1_coord;
  Vector<double> ynp1_coord;  
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_udotn;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_vdotn;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_un;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_vn;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_wn;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_unm1;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_vnm1;
  PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_wnm1;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_u;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_v;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_udot;
  PETScWrappers::MPI::Vector completely_distributed_mesh_velocity_vdot;
  
  // OTHER VECTORS 
  PETScWrappers::MPI::Vector system_rhs_u;
  PETScWrappers::MPI::Vector system_rhs_v;
  PETScWrappers::MPI::Vector system_rhs_w;
  PETScWrappers::MPI::Vector system_rhs_psi;
  PETScWrappers::MPI::Vector system_rhs_q;
  PETScWrappers::MPI::Vector locally_relevant_solution_phi;
  PETScWrappers::MPI::Vector locally_relevant_solution_phi_old;
  PETScWrappers::MPI::Vector locally_relevant_solution_u;
  PETScWrappers::MPI::Vector locally_relevant_solution_v;
  PETScWrappers::MPI::Vector locally_relevant_solution_w;
  PETScWrappers::MPI::Vector locally_relevant_solution_u_old;
  PETScWrappers::MPI::Vector locally_relevant_solution_v_old;
  PETScWrappers::MPI::Vector locally_relevant_solution_w_old;

  PETScWrappers::MPI::Vector locally_relevant_solution_psi;
  PETScWrappers::MPI::Vector locally_relevant_solution_psi_old;
  PETScWrappers::MPI::Vector locally_relevant_solution_p;

  PETScWrappers::MPI::Vector completely_distributed_solution_u;
  PETScWrappers::MPI::Vector completely_distributed_solution_v;
  PETScWrappers::MPI::Vector completely_distributed_solution_w;
  PETScWrappers::MPI::Vector completely_distributed_solution_psi;
  PETScWrappers::MPI::Vector completely_distributed_solution_q;
  PETScWrappers::MPI::Vector completely_distributed_solution_p;
};

// CONSTRUCTOR FOR LEVEL SET
template<int dim>
NavierStokesSolver<dim>::NavierStokesSolver(const unsigned int degree_LS,
					    const unsigned int degree_U, 
					    const double time_step, 
					    const double eps, 
					    const double rho_air,
					    const double nu_air, 
					    const double rho_fluid, 
					    const double nu_fluid,
					    Function<dim> &force_function, 
					    const bool verbose, 
					    parallel::distributed::Triangulation<dim> &triangulation, 
					    MappingQ<dim> &mapping_tnp1, 
					    MappingQ<dim> &mapping_tn, 
					    MappingQ<dim> &mapping_tnm1,
					    MPI_Comm &mpi_communicator) 
  :
  mpi_communicator(mpi_communicator), 
  mapping_tnp1(mapping_tnp1),
  mapping_tn(mapping_tn),
  mapping_tnm1(mapping_tnm1),
  triangulation(triangulation), 
  degree_LS(degree_LS), 
  dof_handler_LS(triangulation), 
  fe_LS(degree_LS), 
  degree_U(degree_U), 
  dof_handler_U(triangulation), 
  fe_U(degree_U), 
  dof_handler_P(triangulation), 
  fe_P(degree_U-1), 
  dof_handler_U_mesh(triangulation),
  fe_U_mesh(1),
  force_function(force_function),
  //This is dummy since rho and nu functions won't be used
  rho_function(force_function), 
  nu_function(force_function), 
  rho_air(rho_air), 
  nu_air(nu_air), 
  rho_fluid(rho_fluid), 
  nu_fluid(nu_fluid), 
  time_step(time_step), 
  eps(eps), 
  verbose(verbose), 
  LEVEL_SET(1), 
  pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator)==0)), 
  transport_pressure(1,1, //degree_LS and velocity
		     time_step,
		     0,1.0, // cK and cE
		     verbose,
		     "NMPP_uH",FORWARD_EULER, // ALGORITHMS in SPACE and TIME
		     triangulation,
		     mapping_tn,mapping_tnm1,
		     mpi_communicator),
  rebuild_Matrix_U(true), 
  rebuild_S_M(true),
  rebuild_Matrix_U_preconditioners(true),
  rebuild_S_M_preconditioners(true)
{setup();}

// CONSTRUCTOR NOT FOR LEVEL SET
template<int dim>
NavierStokesSolver<dim>::NavierStokesSolver(const unsigned int degree_LS,
					    const unsigned int degree_U, 
					    const double time_step, 
					    Function<dim> &force_function,
					    Function<dim> &rho_function, 
					    Function<dim> &nu_function, 
					    const bool verbose,
					    parallel::distributed::Triangulation<dim> &triangulation, 
					    MappingQ<dim> &mapping_tnp1, 
					    MappingQ<dim> &mapping_tn, 
					    MappingQ<dim> &mapping_tnm1,
					    MPI_Comm &mpi_communicator) :
  mpi_communicator(mpi_communicator), 
  mapping_tnp1(mapping_tnp1),
  mapping_tn(mapping_tn),
  mapping_tnm1(mapping_tnm1),
  triangulation(triangulation), 
  degree_LS(degree_LS), 
  dof_handler_LS(triangulation), 
  fe_LS(degree_LS), 
  degree_U(degree_U), 
  dof_handler_U(triangulation), 
  fe_U(degree_U), 
  dof_handler_P(triangulation), 
  fe_P(degree_U-1), 
  dof_handler_U_mesh(triangulation),
  fe_U_mesh(1),
  force_function(force_function), 
  rho_function(rho_function), 
  nu_function(nu_function), 
  time_step(time_step), 
  verbose(verbose), 
  LEVEL_SET(0), 
  pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator)==0)), 
  transport_pressure(1,1, //degree_LS and velocity
		     time_step,
		     0,1.0, // cK and cE
		     verbose,
		     "NMPP_uH",FORWARD_EULER, // ALGORITHMS in SPACE and TIME
		     triangulation,
		     mapping_tn,mapping_tnm1,
		     mpi_communicator),
  rebuild_Matrix_U(true), 
  rebuild_S_M(true),
  rebuild_Matrix_U_preconditioners(true),
  rebuild_S_M_preconditioners(true)
{setup();}

template<int dim>
NavierStokesSolver<dim>::~NavierStokesSolver() {
  dof_handler_LS.clear();
  dof_handler_U.clear();
  dof_handler_P.clear();
  dof_handler_U_mesh.clear();
}

/////////////////////////////////////////////////////////////
//////////////////// SETTERS AND GETTERS ////////////////////
/////////////////////////////////////////////////////////////
template <int dim>
void NavierStokesSolver<dim>::set_coord(Vector<double> xn_coord, Vector<double> yn_coord,
					Vector<double> xnp1_coord, Vector<double> ynp1_coord)
{
  this->xn_coord = xn_coord;
  this->yn_coord = yn_coord;
  this->xnp1_coord = xnp1_coord;
  this->ynp1_coord = ynp1_coord;
}

template <int dim>
void NavierStokesSolver<dim>::set_mesh_velocity(PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_un,
						PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_vn)
{
  // set mesh velocity JUST FOR FIRST TIME STEP
  this->locally_relevant_mesh_velocity_un=locally_relevant_mesh_velocity_un;
  this->locally_relevant_mesh_velocity_vn=locally_relevant_mesh_velocity_vn;
  // old velocity
  locally_relevant_mesh_velocity_unm1=locally_relevant_mesh_velocity_un;
  locally_relevant_mesh_velocity_vnm1=locally_relevant_mesh_velocity_vn;
}

template <int dim>
void NavierStokesSolver<dim>::set_mesh_velocity(PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_un,
						PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_v,
						PETScWrappers::MPI::Vector locally_relevant_mesh_velocity_w)
{
  // set mesh velocity JUST FOR FIRST TIME STEP
  this->locally_relevant_mesh_velocity_un=locally_relevant_mesh_velocity_un;
  this->locally_relevant_mesh_velocity_vn=locally_relevant_mesh_velocity_vn;
  this->locally_relevant_mesh_velocity_wn=locally_relevant_mesh_velocity_wn;
  // old velocity
  locally_relevant_mesh_velocity_unm1=locally_relevant_mesh_velocity_un;
  locally_relevant_mesh_velocity_vnm1=locally_relevant_mesh_velocity_vn;
  locally_relevant_mesh_velocity_wnm1=locally_relevant_mesh_velocity_wn;
}

template<int dim>
void NavierStokesSolver<dim>::set_rho_and_nu_functions(const Function<dim> &rho_function,
						       const Function<dim> &nu_function) {
  this->rho_function=rho_function;
  this->nu_function=nu_function;
}

template<int dim>
void NavierStokesSolver<dim>::initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_phi,
						PETScWrappers::MPI::Vector locally_relevant_solution_u,
						PETScWrappers::MPI::Vector locally_relevant_solution_v,
						PETScWrappers::MPI::Vector locally_relevant_solution_p) {
  this->locally_relevant_solution_phi=locally_relevant_solution_phi;
  this->locally_relevant_solution_u=locally_relevant_solution_u;
  this->locally_relevant_solution_v=locally_relevant_solution_v; 
  this->locally_relevant_solution_p=locally_relevant_solution_p;
  completely_distributed_solution_p=locally_relevant_solution_p;
  // set old vectors to the initial condition (just for first time step)
  save_old_solution();
  locally_relevant_solution_phi_old = locally_relevant_solution_phi;
}

template<int dim>
void NavierStokesSolver<dim>::initial_condition(PETScWrappers::MPI::Vector locally_relevant_solution_phi,
						PETScWrappers::MPI::Vector locally_relevant_solution_u,
						PETScWrappers::MPI::Vector locally_relevant_solution_v,
						PETScWrappers::MPI::Vector locally_relevant_solution_w,
						PETScWrappers::MPI::Vector locally_relevant_solution_p) 
{
  this->locally_relevant_solution_phi=locally_relevant_solution_phi;
  this->locally_relevant_solution_u=locally_relevant_solution_u;
  this->locally_relevant_solution_v=locally_relevant_solution_v;
  this->locally_relevant_solution_w=locally_relevant_solution_w;
  this->locally_relevant_solution_p=locally_relevant_solution_p;
  completely_distributed_solution_p=locally_relevant_solution_p;
  // set old vectors to the initial condition (just for first time step)
  save_old_solution();
  locally_relevant_solution_phi_old=locally_relevant_solution_phi;
}

template<int dim>
void NavierStokesSolver<dim>::set_boundary_conditions(std::vector<unsigned int> boundary_values_id_u,
						      std::vector<unsigned int> boundary_values_id_v, 
						      std::vector<double> boundary_values_u,
						      std::vector<double> boundary_values_v) 
{
  this->boundary_values_id_u=boundary_values_id_u;
  this->boundary_values_id_v=boundary_values_id_v;
  this->boundary_values_u=boundary_values_u;
  this->boundary_values_v=boundary_values_v;
}

template<int dim>
void NavierStokesSolver<dim>::set_boundary_conditions(std::vector<unsigned int> boundary_values_id_u,
						      std::vector<unsigned int> boundary_values_id_v,
						      std::vector<unsigned int> boundary_values_id_w, 
						      std::vector<double> boundary_values_u,
						      std::vector<double> boundary_values_v, 
						      std::vector<double> boundary_values_w) 
{
  this->boundary_values_id_u=boundary_values_id_u;
  this->boundary_values_id_v=boundary_values_id_v;
  this->boundary_values_id_w=boundary_values_id_w;
  this->boundary_values_u=boundary_values_u;
  this->boundary_values_v=boundary_values_v;
  this->boundary_values_w=boundary_values_w;
}

template<int dim>
void NavierStokesSolver<dim>::set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_u,
					   PETScWrappers::MPI::Vector locally_relevant_solution_v) {
  this->locally_relevant_solution_u=locally_relevant_solution_u;
  this->locally_relevant_solution_v=locally_relevant_solution_v;
}

template<int dim>
void NavierStokesSolver<dim>::set_velocity(PETScWrappers::MPI::Vector locally_relevant_solution_u,
					   PETScWrappers::MPI::Vector locally_relevant_solution_v,
					   PETScWrappers::MPI::Vector locally_relevant_solution_w) {
  this->locally_relevant_solution_u=locally_relevant_solution_u;
  this->locally_relevant_solution_v=locally_relevant_solution_v;
  this->locally_relevant_solution_w=locally_relevant_solution_w;
}

template<int dim>
void NavierStokesSolver<dim>::set_phi(PETScWrappers::MPI::Vector locally_relevant_solution_phi) {
  locally_relevant_solution_phi_old=this->locally_relevant_solution_phi;
  this->locally_relevant_solution_phi=locally_relevant_solution_phi;
}

template<int dim>
void NavierStokesSolver<dim>::get_rho_and_nu(double phi) {
  double H=0;
  // get rho, nu
  if (phi>eps)
    H=1;
  else if (phi<-eps)
    H=-1;
  else
    H=phi/eps;
  rho_value=rho_fluid*(1+H)/2.+rho_air*(1-H)/2.;
  nu_value=nu_fluid*(1+H)/2.+nu_air*(1-H)/2.;
}

template<int dim>
void NavierStokesSolver<dim>::get_rho(double &rho, double phi) {
  double H=0;
  // get rho, nu
  if (phi>eps)
    H=1;
  else if (phi<-eps)
    H=-1;
  else
    H=phi/eps;
  rho=rho_fluid*(1+H)/2.+rho_air*(1-H)/2.;
}

template<int dim>
void NavierStokesSolver<dim>::get_rho_and_nu(double &rho, double &mu, double phi) {
  double H=0;
  // get rho, nu
  if (phi>eps)
    H=1;
  else if (phi<-eps)
    H=-1;
  else
    H=phi/eps;
  rho=rho_fluid*(1+H)/2.+rho_air*(1-H)/2.;
  mu=nu_fluid*(1+H)/2.+nu_air*(1-H)/2.;
}

template<int dim>
void NavierStokesSolver<dim>::get_grad_and_value_rho(Tensor<1,dim> &grad_rho, double &rho, 
						     Tensor<1,dim> grad_phi, double phi) {
  double H=0;
  double Hp=0;
  // get rho, nu
  if (phi>eps)
    {
      H=1;
      Hp=0;
    }
  else if (phi<-eps)
    {
      H=-1;
      Hp=0;
    }
  else
    {
      H=phi/eps;
      Hp=1./eps;
    }
  rho=rho_fluid*(1+H)/2.+rho_air*(1-H)/2.;
  grad_rho[0] = (rho_fluid-rho_air)/2.*Hp*grad_phi[0];
  grad_rho[1] = (rho_fluid-rho_air)/2.*Hp*grad_phi[1];
}


template<int dim>
void NavierStokesSolver<dim>::get_pressure(PETScWrappers::MPI::Vector &locally_relevant_solution_p) 
{
  locally_relevant_solution_p=this->locally_relevant_solution_p;
}

template<int dim>
void NavierStokesSolver<dim>::get_velocity(PETScWrappers::MPI::Vector &locally_relevant_solution_u,
					   PETScWrappers::MPI::Vector &locally_relevant_solution_v) {
  locally_relevant_solution_u=this->locally_relevant_solution_u;
  locally_relevant_solution_v=this->locally_relevant_solution_v;
}

template<int dim>
void NavierStokesSolver<dim>::get_velocity(PETScWrappers::MPI::Vector &locally_relevant_solution_u,
					   PETScWrappers::MPI::Vector &locally_relevant_solution_v,
					   PETScWrappers::MPI::Vector &locally_relevant_solution_w) {
  locally_relevant_solution_u=this->locally_relevant_solution_u;
  locally_relevant_solution_v=this->locally_relevant_solution_v;
  locally_relevant_solution_w=this->locally_relevant_solution_w;
}

///////////////////////////////////////////////////////
///////////// SETUP AND INITIAL CONDITION /////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::setup() {
  pcout<<"***** SETUP IN NAVIER STOKES SOLVER *****"<<std::endl;
  setup_DOF();
  init_constraints();
  setup_VECTORS();
}

template<int dim>
void NavierStokesSolver<dim>::setup_DOF() {
  time_tnm1=0;
  time_tn=0;
  time_tnp1=0;
  rho_min = 1.;
  degree_MAX=std::max(degree_LS,degree_U);
  // setup system LS
  dof_handler_LS.distribute_dofs(fe_LS);
  locally_owned_dofs_LS=dof_handler_LS.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_LS,locally_relevant_dofs_LS);
  // setup system U
  dof_handler_U.distribute_dofs(fe_U);
  locally_owned_dofs_U=dof_handler_U.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_U,locally_relevant_dofs_U);
  // setup system P //
  dof_handler_P.distribute_dofs(fe_P);
  locally_owned_dofs_P=dof_handler_P.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_P,locally_relevant_dofs_P);
  // setup system U mesh
  dof_handler_U_mesh.distribute_dofs (fe_U_mesh);
  locally_owned_dofs_U_mesh = dof_handler_U_mesh.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs (dof_handler_U_mesh,
					   locally_relevant_dofs_U_mesh);
}

template<int dim>
void NavierStokesSolver<dim>::setup_VECTORS() {
  xn_coord.reinit(dof_handler_U_mesh.n_dofs());
  yn_coord.reinit(dof_handler_U_mesh.n_dofs());
  xnp1_coord.reinit(dof_handler_U_mesh.n_dofs());
  ynp1_coord.reinit(dof_handler_U_mesh.n_dofs());
  // mesh velocity vectors
  locally_relevant_mesh_velocity_udotn.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator);
  locally_relevant_mesh_velocity_vdotn.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator);
  locally_relevant_mesh_velocity_udotn=0.;
  locally_relevant_mesh_velocity_vdotn=0.;
  locally_relevant_mesh_velocity_un.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator);
  locally_relevant_mesh_velocity_vn.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator);
  locally_relevant_mesh_velocity_wn.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator);
  locally_relevant_mesh_velocity_un=0.;
  locally_relevant_mesh_velocity_vn=0.;
  locally_relevant_mesh_velocity_wn=0.;  
  locally_relevant_mesh_velocity_unm1.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator);
  locally_relevant_mesh_velocity_vnm1.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator);
  locally_relevant_mesh_velocity_wnm1.reinit(locally_owned_dofs_U_mesh,locally_relevant_dofs_U_mesh,mpi_communicator);
  locally_relevant_mesh_velocity_unm1=0.;
  locally_relevant_mesh_velocity_vnm1=0.;
  locally_relevant_mesh_velocity_wnm1=0.;  
  completely_distributed_mesh_velocity_u.reinit(locally_owned_dofs_U_mesh,mpi_communicator);
  completely_distributed_mesh_velocity_v.reinit(locally_owned_dofs_U_mesh,mpi_communicator);
  completely_distributed_mesh_velocity_udot.reinit(locally_owned_dofs_U_mesh,mpi_communicator);
  completely_distributed_mesh_velocity_vdot.reinit(locally_owned_dofs_U_mesh,mpi_communicator);
  // init vectors for phi
  locally_relevant_solution_phi.reinit(locally_owned_dofs_LS,locally_relevant_dofs_LS,
				       mpi_communicator);
  locally_relevant_solution_phi_old.reinit(locally_owned_dofs_LS,locally_relevant_dofs_LS,
					   mpi_communicator);
  locally_relevant_solution_phi=0;
  locally_relevant_solution_phi_old=0;
  //init vectors for u
  locally_relevant_solution_u.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
				     mpi_communicator);
  locally_relevant_solution_u=0;
  completely_distributed_solution_u.reinit(locally_owned_dofs_U,mpi_communicator);
  system_rhs_u.reinit(locally_owned_dofs_U,mpi_communicator);
  //init vectors for u_old
  locally_relevant_solution_u_old.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
					 mpi_communicator);
  locally_relevant_solution_u_old=0;
  //init vectors for v
  locally_relevant_solution_v.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
				     mpi_communicator);
  locally_relevant_solution_v=0;
  completely_distributed_solution_v.reinit(locally_owned_dofs_U,mpi_communicator);
  system_rhs_v.reinit(locally_owned_dofs_U,mpi_communicator);
  //init vectors for v_old
  locally_relevant_solution_v_old.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
					 mpi_communicator);
  locally_relevant_solution_v_old=0;
  //init vectors for w
  locally_relevant_solution_w.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
				     mpi_communicator);
  locally_relevant_solution_w=0;
  completely_distributed_solution_w.reinit(locally_owned_dofs_U,mpi_communicator);
  system_rhs_w.reinit(locally_owned_dofs_U,mpi_communicator);
  //init vectors for w_old
  locally_relevant_solution_w_old.reinit(locally_owned_dofs_U,locally_relevant_dofs_U,
					 mpi_communicator);
  locally_relevant_solution_w_old=0;
  //init vectors for dpsi
  locally_relevant_solution_psi.reinit(locally_owned_dofs_P,locally_relevant_dofs_P,
				       mpi_communicator);
  locally_relevant_solution_psi=0;
  system_rhs_psi.reinit(locally_owned_dofs_P,mpi_communicator);
  //init vectors for dpsi old
  locally_relevant_solution_psi_old.reinit(locally_owned_dofs_P,locally_relevant_dofs_P,
					   mpi_communicator);
  locally_relevant_solution_psi_old=0;
  //init vectors for q
  completely_distributed_solution_q.reinit(locally_owned_dofs_P,mpi_communicator);
  system_rhs_q.reinit(locally_owned_dofs_P,mpi_communicator);
  //init vectors for psi
  completely_distributed_solution_psi.reinit(locally_owned_dofs_P,mpi_communicator);
  //init vectors for p
  locally_relevant_solution_p.reinit(locally_owned_dofs_P,locally_relevant_dofs_P,
				     mpi_communicator);
  locally_relevant_solution_p=0;
  completely_distributed_solution_p.reinit(locally_owned_dofs_P,mpi_communicator);
  completely_distributed_solution_p=0;
  ////////////////////////////
  // Initialize constraints //
  ////////////////////////////
  init_constraints();
  //////////////////////
  // Sparsity pattern //
  //////////////////////
  // sparsity pattern for A
  DynamicSparsityPattern dsp_Matrix(locally_relevant_dofs_U);
  DoFTools::make_sparsity_pattern(dof_handler_U,dsp_Matrix,constraints,false);
  SparsityTools::distribute_sparsity_pattern(dsp_Matrix,
					     dof_handler_U.n_locally_owned_dofs_per_processor(),mpi_communicator,
					     locally_relevant_dofs_U);
  system_Matrix_u.reinit(mpi_communicator,dsp_Matrix,
			 dof_handler_U.n_locally_owned_dofs_per_processor(),
			 dof_handler_U.n_locally_owned_dofs_per_processor(),
			 Utilities::MPI::this_mpi_process(mpi_communicator));
  system_Matrix_v.reinit(mpi_communicator,dsp_Matrix,
			 dof_handler_U.n_locally_owned_dofs_per_processor(),
			 dof_handler_U.n_locally_owned_dofs_per_processor(),
			 Utilities::MPI::this_mpi_process(mpi_communicator));
  system_Matrix_w.reinit(mpi_communicator,dsp_Matrix,
			 dof_handler_U.n_locally_owned_dofs_per_processor(),
			 dof_handler_U.n_locally_owned_dofs_per_processor(),
			 Utilities::MPI::this_mpi_process(mpi_communicator));
  rebuild_Matrix_U=true;
  // sparsity pattern for S
  DynamicSparsityPattern dsp_S(locally_relevant_dofs_P);
  DoFTools::make_sparsity_pattern(dof_handler_P,dsp_S,constraints_psi,false);
  SparsityTools::distribute_sparsity_pattern(dsp_S,
					     dof_handler_P.n_locally_owned_dofs_per_processor(),mpi_communicator,
					     locally_relevant_dofs_P);
  system_S.reinit(mpi_communicator,dsp_S,dof_handler_P.n_locally_owned_dofs_per_processor(),
		  dof_handler_P.n_locally_owned_dofs_per_processor(),
		  Utilities::MPI::this_mpi_process(mpi_communicator));
  // sparsity pattern for M
  DynamicSparsityPattern dsp_M(locally_relevant_dofs_P);
  DoFTools::make_sparsity_pattern(dof_handler_P,dsp_M,constraints_q,false);
  SparsityTools::distribute_sparsity_pattern(dsp_M,
					     dof_handler_P.n_locally_owned_dofs_per_processor(),mpi_communicator,
					     locally_relevant_dofs_P);
  system_M.reinit(mpi_communicator,dsp_M,dof_handler_P.n_locally_owned_dofs_per_processor(),
		  dof_handler_P.n_locally_owned_dofs_per_processor(),
		  Utilities::MPI::this_mpi_process(mpi_communicator));
  rebuild_S_M=true;
}

template<int dim>
void NavierStokesSolver<dim>::init_constraints() {
  //grl constraints
  constraints.clear();
  constraints.reinit(locally_relevant_dofs_U);
  DoFTools::make_hanging_node_constraints(dof_handler_U,constraints);
  constraints.close();
  //constraints for q
  constraints_q.clear();
  constraints_q.reinit(locally_relevant_dofs_P);
  DoFTools::make_hanging_node_constraints(dof_handler_P,constraints_q);
  constraints_q.close();
  //constraints for dpsi
  constraints_psi.clear();
  constraints_psi.reinit(locally_relevant_dofs_P);
  DoFTools::make_hanging_node_constraints(dof_handler_P,constraints_psi);
  if (constraints_psi.can_store_line(0))
    constraints_psi.add_line(0); //constraint u0 = 0
  constraints_psi.close();
}

///////////////////////////////////////////////////////
////////////////// ASSEMBLE SYSTEMS ///////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::assemble_system_U_BDF1() 
{
  system_Matrix_u=0;
  system_Matrix_v=0;
  system_rhs_u=0;
  system_rhs_v=0;
  
  const QGauss<dim> quadrature_formula(degree_MAX+1);
  FEValues<dim> fe_values_LS_tnp1(mapping_tnp1,fe_LS,quadrature_formula,
				  update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_LS_tn(mapping_tn,fe_LS,quadrature_formula,
				update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_P_tn(mapping_tn,fe_P,quadrature_formula,
			       update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U_tn(mapping_tn,fe_U,quadrature_formula, 
			       update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U_tnp1(mapping_tnp1,fe_U,quadrature_formula, 
				 update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U_mesh_tn(mapping_tn,fe_U_mesh,quadrature_formula, 
				    update_values|update_gradients|update_quadrature_points|update_JxW_values);
  
  const unsigned int dofs_per_cell=fe_U.dofs_per_cell;
  const unsigned int n_q_points=quadrature_formula.size();
  
  FullMatrix<double> cell_A_u(dofs_per_cell,dofs_per_cell);
  FullMatrix<double> cell_A_v(dofs_per_cell,dofs_per_cell);
  Vector<double> cell_rhs_u(dofs_per_cell);
  Vector<double> cell_rhs_v(dofs_per_cell);

  std::vector<double> phi_tn(n_q_points);
  std::vector<double> phi_tnp1(n_q_points);
  std::vector<Tensor<1, dim> > grad_phi_tn(n_q_points);

  std::vector<double> un(n_q_points);
  std::vector<double> vn(n_q_points);
  std::vector<Tensor<1, dim> > grad_un(n_q_points);
  std::vector<Tensor<1, dim> > grad_vn(n_q_points);
  std::vector<Tensor<1, dim> > grad_pn(n_q_points);
  std::vector<Tensor<1, dim> > grad_psin(n_q_points); //INCREMENTAL

  // mesh velocity
  std::vector<double> un_mesh(n_q_points);
  std::vector<double> vn_mesh(n_q_points);
  std::vector<Tensor<1, dim> > grad_un_mesh(n_q_points);
  std::vector<Tensor<1, dim> > grad_vn_mesh(n_q_points);

  std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  std::vector<Tensor<1, dim> > shape_grad_tn(dofs_per_cell);
  std::vector<Tensor<1, dim> > shape_grad_tnp1(dofs_per_cell);
  std::vector<double> shape_value_tnp1(dofs_per_cell);
  std::vector<double> shape_value_tn(dofs_per_cell);
  
  double force_u;
  double force_v;
  Vector<double> force_terms(dim);

  typename DoFHandler<dim>::active_cell_iterator 
    cell_U=dof_handler_U.begin_active(), endc_U=dof_handler_U.end();
  typename DoFHandler<dim>::active_cell_iterator cell_P=dof_handler_P.begin_active();
  typename DoFHandler<dim>::active_cell_iterator cell_LS=dof_handler_LS.begin_active();
  typename DoFHandler<dim>::active_cell_iterator cell_U_mesh=dof_handler_U_mesh.begin_active();

  // PRESSURE and FORCE at time=tn
  ExactSolution_p<dim> pressure(time_tn);
  force_function.set_time(time_tn); 
  double rhon, rhonp1, mu_value, mu_old_value;
  Tensor<1,dim> grad_rhon;
  
  for (; cell_U!=endc_U; ++cell_U,++cell_P,++cell_LS, ++cell_U_mesh)
    if (cell_U->is_locally_owned()) {
      cell_A_u=0;
      cell_A_v=0;
      cell_rhs_u=0;
      cell_rhs_v=0;

      fe_values_LS_tn.reinit(cell_LS);
      fe_values_LS_tnp1.reinit(cell_LS);
      fe_values_U_tn.reinit(cell_U);
      fe_values_U_tnp1.reinit(cell_U);
      fe_values_P_tn.reinit(cell_P);
      fe_values_U_mesh_tn.reinit(cell_U_mesh);

      // get function values for Phi at time=tn and time=tnp1
      fe_values_LS_tn.get_function_values(locally_relevant_solution_phi_old,phi_tn);
      fe_values_LS_tn.get_function_gradients(locally_relevant_solution_phi_old,grad_phi_tn);
      fe_values_LS_tnp1.get_function_values(locally_relevant_solution_phi,phi_tnp1);
      // get function values for U at time=tn
      fe_values_U_tn.get_function_values(locally_relevant_solution_u,un);
      fe_values_U_tn.get_function_values(locally_relevant_solution_v,vn);
      fe_values_U_tn.get_function_gradients(locally_relevant_solution_u,grad_un);
      fe_values_U_tn.get_function_gradients(locally_relevant_solution_v,grad_vn);
      // get mesh velocities at time=tn
      fe_values_U_mesh_tn.get_function_values(locally_relevant_mesh_velocity_un,un_mesh);
      fe_values_U_mesh_tn.get_function_values(locally_relevant_mesh_velocity_vn,vn_mesh);
      fe_values_U_mesh_tn.get_function_gradients(locally_relevant_mesh_velocity_un,grad_un_mesh);
      fe_values_U_mesh_tn.get_function_gradients(locally_relevant_mesh_velocity_vn,grad_vn_mesh);

      // get function values for pressure
      fe_values_P_tn.get_function_gradients(locally_relevant_solution_p,grad_pn);
      fe_values_P_tn.get_function_gradients(locally_relevant_solution_psi,grad_psin); // INCREMENTAL

      for (unsigned int q=0; q<n_q_points; ++q) 
	{
	  Point<dim> q_point_tn = fe_values_U_tn.quadrature_point(q);
	  Point<dim> q_point_tnp1 = fe_values_U_tnp1.quadrature_point(q);

	  const double JxW_tn=fe_values_U_tn.JxW(q);
	  const double JxW_tnp1=fe_values_U_tnp1.JxW(q);

	  for (unsigned int i=0; i<dofs_per_cell; ++i) 
	    {
	      shape_grad_tn[i]=fe_values_U_tn.shape_grad(i,q);
	      shape_grad_tnp1[i]=fe_values_U_tnp1.shape_grad(i,q);     
	      shape_value_tn[i]=fe_values_U_tn.shape_value(i,q);
	      shape_value_tnp1[i]=fe_values_U_tnp1.shape_value(i,q);
	    }

	  //double pressure_grad_u = pressure.gradient(q_point_tn)[0];
	  //double pressure_grad_v = pressure.gradient(q_point_tn)[1];
	  double pressure_grad_u = grad_pn[q][0]+grad_psin[q][0]; // INCREMENTAL
	  double pressure_grad_v = grad_pn[q][1]+grad_psin[q][1]; 

	  // FORCE TERMS at time=tn
	  force_function.vector_value(q_point_tn,force_terms);
	  force_u=force_terms[0];
	  force_v=force_terms[1];

	  // DENSITY AND VISCOSITY
	  if (LEVEL_SET==1)
	    {
	      // rho at tn
	      get_grad_and_value_rho(grad_rhon, rhon, grad_phi_tn[q], phi_tn[q]);
	      // rho and mu values at tnp1
	      get_rho_and_nu(rhonp1,mu_value,phi_tnp1[q]);
	      get_rho_and_nu(rhon,mu_old_value,phi_tn[q]);
	      force_u*=rhon;
	      force_v*=rhon;
	    }
	  else
	    {
	      // rho function at tn
	      rho_function.set_time(time_tn); 
	      rhon = rho_function.value(q_point_tn);
	      grad_rhon = rho_function.gradient(q_point_tn);
	      mu_old_value=nu_function.value(q_point_tn);
	      // rho and mu functions at tnp1
	      rho_function.set_time(time_tnp1);
	      nu_function.set_time(time_tnp1);
	      rhonp1 = rho_function.value(q_point_tnp1);
	      mu_value=nu_function.value(q_point_tnp1);
	    }	      

	  double divUn = grad_un[q][0]+grad_vn[q][1];
	  double divUn_mesh = grad_un_mesh[q][0] + grad_vn_mesh[q][1];

	  double nonlinearity1_u = rhon*un[q]*(divUn-divUn_mesh);
	  double nonlinearity1_v = rhon*vn[q]*(divUn-divUn_mesh);

	  double nonlinearity2_u = ( (un[q]-un_mesh[q])*(rhon*grad_un[q][0]+un[q]*grad_rhon[0]) 
				     + (vn[q]-vn_mesh[q])*(rhon*grad_un[q][1]+un[q]*grad_rhon[1]) );
	  double nonlinearity2_v = ( (un[q]-un_mesh[q])*(rhon*grad_vn[q][0]+vn[q]*grad_rhon[0]) 
				     + (vn[q]-vn_mesh[q])*(rhon*grad_vn[q][1]+vn[q]*grad_rhon[1]) );
	  
	  double nonlinearity_u = nonlinearity1_u + nonlinearity2_u;
	  double nonlinearity_v = nonlinearity1_v + nonlinearity2_v;

	  for (unsigned int i=0; i<dofs_per_cell; ++i) 
	    {
	      cell_rhs_u(i)+=((rhon*un[q]+time_step*(force_u-nonlinearity_u-pressure_grad_u))*shape_value_tn[i]
			      //-time_step*mu_old_value*(grad_vn[q][0]*shape_grad_tn[i][1])
			      )*JxW_tn;
	      cell_rhs_v(i)+=((rhon*vn[q]+time_step*(force_v-nonlinearity_v-pressure_grad_v))*shape_value_tn[i]
			      //-time_step*mu_old_value*(grad_un[q][1]*shape_grad_tn[i][0])
			      )*JxW_tn;
	      for (unsigned int j=0; j<dofs_per_cell; ++j) 
		{
		  cell_A_u(i,j)+=(rhonp1*shape_value_tnp1[i]*shape_value_tnp1[j]
				  +time_step*mu_value*(shape_grad_tnp1[i]*shape_grad_tnp1[j])
				  //+time_step*mu_value*(shape_grad_tnp1[i][0]*shape_grad_tnp1[j][0])
				  )*JxW_tnp1;
		  cell_A_v(i,j)+=(rhonp1*shape_value_tnp1[i]*shape_value_tnp1[j]
				  +time_step*mu_value*(shape_grad_tnp1[i]*shape_grad_tnp1[j])
				  //+time_step*mu_value*(shape_grad_tnp1[i][1]*shape_grad_tnp1[j][1])
				  )*JxW_tnp1;
		}
	    }
	}
      cell_U->get_dof_indices(local_dof_indices);
      // DISTRIBUTE
      constraints.distribute_local_to_global(cell_A_u,local_dof_indices,system_Matrix_u);
      constraints.distribute_local_to_global(cell_A_v,local_dof_indices,system_Matrix_v);
      constraints.distribute_local_to_global(cell_rhs_u,local_dof_indices,system_rhs_u);
      constraints.distribute_local_to_global(cell_rhs_v,local_dof_indices,system_rhs_v);
    }
  // COMPRESS
  system_rhs_u.compress(VectorOperation::add);
  system_rhs_v.compress(VectorOperation::add);
  system_Matrix_u.compress(VectorOperation::add);
  system_Matrix_v.compress(VectorOperation::add);
  // BOUNDARY CONDITIONS
  system_rhs_u.set(boundary_values_id_u,boundary_values_u);
  system_rhs_u.compress(VectorOperation::insert);
  system_rhs_v.set(boundary_values_id_v,boundary_values_v);
  system_rhs_v.compress(VectorOperation::insert);
  system_Matrix_u.clear_rows(boundary_values_id_u,1);
  system_Matrix_v.clear_rows(boundary_values_id_v,1);
  system_Matrix_u.compress(VectorOperation::insert);
  system_Matrix_v.compress(VectorOperation::insert);
  // PRECONDITIONERS
  if (rebuild_Matrix_U_preconditioners)
    {
      rebuild_Matrix_U_preconditioners=false;
      preconditioner_Matrix_u.reset(new PETScWrappers::PreconditionBoomerAMG
				    (system_Matrix_u,PETScWrappers::PreconditionBoomerAMG::AdditionalData(false)));
      preconditioner_Matrix_v.reset( new PETScWrappers::PreconditionBoomerAMG
				     (system_Matrix_v,PETScWrappers::PreconditionBoomerAMG::AdditionalData(false)));
    }
}

template<int dim>
void NavierStokesSolver<dim>::assemble_system_U_BDF2() 
{
  if (rebuild_Matrix_U==true) 
    {
      system_Matrix_u=0;
      system_Matrix_v=0;
    }
  system_rhs_u=0;
  system_rhs_v=0;
  
  const QGauss<dim> quadrature_formula(degree_MAX+1);
  //FEValues<dim> fe_values_LS_tnm1(mapping_tnm1,fe_LS,quadrature_formula,
  //			  update_values|update_gradients|update_quadrature_points|update_JxW_values);
  //FEValues<dim> fe_values_LS_tn(mapping_tn,fe_LS,quadrature_formula,
  //			update_values|update_gradients|update_quadrature_points|update_JxW_values);
  //FEValues<dim> fe_values_LS_tnp1(mapping_tnp1,fe_LS,quadrature_formula,
  //			  update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U_tnm1(mapping_tnm1,fe_U,quadrature_formula,
				 update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U_tn(mapping_tn,fe_U,quadrature_formula,
			       update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U_tnp1(mapping_tnp1,fe_U,quadrature_formula,
				 update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_P_tnm1(mapping_tnm1,fe_P,quadrature_formula,
				 update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_P_tn(mapping_tn,fe_P,quadrature_formula,
			       update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U_mesh_tnm1(mapping_tnm1,fe_U_mesh,quadrature_formula,
				      update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U_mesh_tn(mapping_tn,fe_U_mesh,quadrature_formula,
				    update_values|update_gradients|update_quadrature_points|update_JxW_values);

  const unsigned int dofs_per_cell=fe_U.dofs_per_cell;
  const unsigned int n_q_points=quadrature_formula.size();
  
  FullMatrix<double> cell_A_u(dofs_per_cell,dofs_per_cell);
  FullMatrix<double> cell_A_v(dofs_per_cell,dofs_per_cell);
  Vector<double> cell_rhs_u(dofs_per_cell);
  Vector<double> cell_rhs_v(dofs_per_cell);
  
  std::vector<double> phinm1(n_q_points);
  std::vector<double> phin(n_q_points);
  std::vector<double> phinp1(n_q_points);

  // velocity
  std::vector<double> un(n_q_points), unm1(n_q_points), vn(n_q_points), vnm1(n_q_points);
  std::vector<Tensor<1, dim> > grad_un(n_q_points), grad_vn(n_q_points), grad_unm1(n_q_points), grad_vnm1(n_q_points);
  // pressure
  std::vector<Tensor<1, dim> > grad_pn(n_q_points), grad_psin(n_q_points), grad_psinm1(n_q_points);
  // mesh velocity
  std::vector<double> un_mesh(n_q_points), vn_mesh(n_q_points), unm1_mesh(n_q_points), vnm1_mesh(n_q_points);
  std::vector<Tensor<1, dim> > grad_un_mesh(n_q_points), grad_vn_mesh(n_q_points), grad_unm1_mesh(n_q_points), grad_vnm1_mesh(n_q_points);
  
  std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  
  std::vector<Tensor<1, dim> > shape_grad_tnp1(dofs_per_cell);
  std::vector<double> shape_value_tnm1(dofs_per_cell), shape_value_tn(dofs_per_cell), shape_value_tnp1(dofs_per_cell);
  
  double force_unp1, force_vnp1;
  Vector<double> force_terms(dim);
  double rhonm1, rhon, rhonp1, munp1;
  Tensor<1,dim> grad_rhon, grad_rhonm1;
  force_function.set_time(time_tnp1);
  ExactSolution_p<dim> pressure(time_tnp1);

  typename DoFHandler<dim>::active_cell_iterator 
    cell_U=dof_handler_U.begin_active(), endc_U=dof_handler_U.end();
  typename DoFHandler<dim>::active_cell_iterator cell_P=dof_handler_P.begin_active();
  typename DoFHandler<dim>::active_cell_iterator cell_LS=dof_handler_LS.begin_active();
  typename DoFHandler<dim>::active_cell_iterator cell_U_mesh=dof_handler_U_mesh.begin_active();
  
  for (; cell_U!=endc_U; ++cell_U,++cell_P,++cell_LS,++cell_U_mesh)
    if (cell_U->is_locally_owned()) 
      {
	cell_A_u=0;
	cell_A_v=0;
	cell_rhs_u=0;
	cell_rhs_v=0;
      
	//fe_values_LS_tnm1.reinit(cell_LS);
	//fe_values_LS_tn.reinit(cell_LS);
	//fe_values_LS_tnp1.reinit(cell_LS);
	fe_values_U_tnm1.reinit(cell_U);
	fe_values_U_tn.reinit(cell_U);
	fe_values_U_tnp1.reinit(cell_U);
	fe_values_P_tnm1.reinit(cell_P);
	fe_values_P_tn.reinit(cell_P);
	fe_values_U_mesh_tnm1.reinit(cell_U_mesh);
	fe_values_U_mesh_tn.reinit(cell_U_mesh);
      
	// get LS solution at different times
	//fe_values_LS_tnm1.get_function_values(locally_relevant_solution_phi,phinm1);
	//fe_values_LS_tn.get_function_values(locally_relevant_solution_phi,phin);
	//fe_values_LS_tnp1.get_function_values(locally_relevant_solution_phi,phinp1);
	// get physical velocities at time=tnm1
	fe_values_U_tnm1.get_function_values(locally_relevant_solution_u_old,unm1);
	fe_values_U_tnm1.get_function_values(locally_relevant_solution_v_old,vnm1);
	fe_values_U_tnm1.get_function_gradients(locally_relevant_solution_u_old,grad_unm1);
	fe_values_U_tnm1.get_function_gradients(locally_relevant_solution_v_old,grad_vnm1);
	// get physical velocities at time=tn
	fe_values_U_tn.get_function_values(locally_relevant_solution_u,un);
	fe_values_U_tn.get_function_values(locally_relevant_solution_v,vn);
	fe_values_U_tn.get_function_gradients(locally_relevant_solution_u,grad_un);
	fe_values_U_tn.get_function_gradients(locally_relevant_solution_v,grad_vn);
	// get mesh velocities at time=tnm1
	fe_values_U_mesh_tnm1.get_function_values(locally_relevant_mesh_velocity_unm1,unm1_mesh);
	fe_values_U_mesh_tnm1.get_function_values(locally_relevant_mesh_velocity_vnm1,vnm1_mesh);
	fe_values_U_mesh_tnm1.get_function_gradients(locally_relevant_mesh_velocity_unm1,grad_unm1_mesh);
	fe_values_U_mesh_tnm1.get_function_gradients(locally_relevant_mesh_velocity_vnm1,grad_vnm1_mesh);
	// get mesh velocities at time=tn
	fe_values_U_mesh_tn.get_function_values(locally_relevant_mesh_velocity_un,un_mesh);
	fe_values_U_mesh_tn.get_function_values(locally_relevant_mesh_velocity_vn,vn_mesh);
	fe_values_U_mesh_tn.get_function_gradients(locally_relevant_mesh_velocity_un,grad_un_mesh);
	fe_values_U_mesh_tn.get_function_gradients(locally_relevant_mesh_velocity_vn,grad_vn_mesh);
	// get values and gradients for p and dpsi
	fe_values_P_tn.get_function_gradients(locally_relevant_solution_p,grad_pn);
	fe_values_P_tn.get_function_gradients(locally_relevant_solution_psi,grad_psin);
	fe_values_P_tnm1.get_function_gradients(locally_relevant_solution_psi_old,grad_psinm1);
      
	for (unsigned int q=0; q<n_q_points; ++q) 
	  {
	    const double JxW_tnm1=fe_values_U_tnm1.JxW(q);
	    const double JxW_tn=fe_values_U_tn.JxW(q);
	    const double JxW_tnp1=fe_values_U_tnp1.JxW(q);

	    for (unsigned int i=0; i<dofs_per_cell; ++i) 
	      {
		shape_value_tnm1[i]=fe_values_U_tnm1.shape_value(i,q);
		shape_value_tn[i]=fe_values_U_tn.shape_value(i,q);
		shape_value_tnp1[i]=fe_values_U_tnp1.shape_value(i,q);
		shape_grad_tnp1[i]=fe_values_U_tnp1.shape_grad(i,q);
	      }	

	if (LEVEL_SET==1) 
	  {
	    // not implemented yet
	  }
	else // rho and nu are defined through functions
	  {
	    //time=tnm1
	    rho_function.set_time(time_tnm1);
	    rhonm1=rho_function.value(fe_values_U_tnm1.quadrature_point(q));
	    grad_rhonm1 = rho_function.gradient(fe_values_U_tnm1.quadrature_point(q));
	    //time=tn
	    rho_function.set_time(time_tn);
	    rhon=rho_function.value(fe_values_U_tn.quadrature_point(q));
	    grad_rhon = rho_function.gradient(fe_values_U_tn.quadrature_point(q));
	    //time=tnp1
	    rho_function.set_time(time_tnp1);
	    rhonp1=rho_function.value(fe_values_U_tnp1.quadrature_point(q));
	    nu_function.set_time(time_tnp1);
	    munp1=nu_function.value(fe_values_U_tnp1.quadrature_point(q));
	  }
	
	// FORCE TERMS at time=tnp1
	force_function.vector_value(fe_values_U_tnp1.quadrature_point(q),force_terms);
	force_unp1=force_terms[0]; force_vnp1=force_terms[1]; 

	if (LEVEL_SET==1) 
	  {
	    force_unp1*=rhonp1;
	    force_vnp1*=rhonp1;
	  }
	
	// NONLINEARITY AT TIME tn
	double divUn = grad_un[q][0]+grad_vn[q][1];
	double divUn_mesh = grad_un_mesh[q][0] + grad_vn_mesh[q][1];
	
	double nonlinearity1_un = rhon*un[q]*(divUn-divUn_mesh);
	double nonlinearity1_vn = rhon*vn[q]*(divUn-divUn_mesh);
	
	double nonlinearity2_un = ( (un[q]-un_mesh[q])*(rhon*grad_un[q][0]+un[q]*grad_rhon[0]) 
				   + (vn[q]-vn_mesh[q])*(rhon*grad_un[q][1]+un[q]*grad_rhon[1]) );
	double nonlinearity2_vn = ( (un[q]-un_mesh[q])*(rhon*grad_vn[q][0]+vn[q]*grad_rhon[0]) 
				   + (vn[q]-vn_mesh[q])*(rhon*grad_vn[q][1]+vn[q]*grad_rhon[1]) );
	
	double nonlinearity_un = nonlinearity1_un + nonlinearity2_un;
	double nonlinearity_vn = nonlinearity1_vn + nonlinearity2_vn;

	// NONLINEARITY AT TIME tnm1
	double divUnm1 = grad_unm1[q][0]+grad_vnm1[q][1];
	double divUnm1_mesh = grad_unm1_mesh[q][0] + grad_vnm1_mesh[q][1];
	
	double nonlinearity1_unm1 = rhonm1*unm1[q]*(divUnm1-divUnm1_mesh);
	double nonlinearity1_vnm1 = rhonm1*vnm1[q]*(divUnm1-divUnm1_mesh);
	
	double nonlinearity2_unm1 = ( (unm1[q]-unm1_mesh[q])*(rhonm1*grad_unm1[q][0]+unm1[q]*grad_rhonm1[0]) 
				      + (vnm1[q]-vnm1_mesh[q])*(rhonm1*grad_unm1[q][1]+unm1[q]*grad_rhonm1[1]) );
	double nonlinearity2_vnm1 = ( (unm1[q]-unm1_mesh[q])*(rhonm1*grad_vnm1[q][0]+vnm1[q]*grad_rhonm1[0]) 
				      + (vnm1[q]-vnm1_mesh[q])*(rhonm1*grad_vnm1[q][1]+vnm1[q]*grad_rhonm1[1]) );
	
	double nonlinearity_unm1 = nonlinearity1_unm1 + nonlinearity2_unm1;
	double nonlinearity_vnm1 = nonlinearity1_vnm1 + nonlinearity2_vnm1;

	for (unsigned int i=0; i<dofs_per_cell; ++i) 
	  {
	    double pressure_grad_x_times_shape_value_times_JxW = 
	      pressure.gradient(fe_values_U_tnp1.quadrature_point(q))[0]*shape_value_tnp1[i]*JxW_tnp1;
	    //(grad_pn[q][0]+4./3*grad_psin[q][0])*shape_value_tn[i]*JxW_tn
	    //-1./3*grad_psinm1[q][0]*shape_value_tnm1[i]*JxW_tnm1;
	    double pressure_grad_y_times_shape_value_times_JxW = 
	      pressure.gradient(fe_values_U_tnp1.quadrature_point(q))[1]*shape_value_tnp1[i]*JxW_tnp1;
	    //(grad_pn[q][1]+4./3*grad_psin[q][1])*shape_value_tn[i]*JxW_tn
	    //-1./3*grad_psinm1[q][1]*shape_value_tnm1[i]*JxW_tnm1;
	    cell_rhs_u(i)+=
	      4./3*rhon*un[q]*shape_value_tn[i]*JxW_tn-1./3*rhonm1*unm1[q]*shape_value_tnm1[i]*JxW_tnm1
	      +2./3*time_step*force_unp1*shape_value_tnp1[i]*JxW_tnp1
	      -2./3*time_step*pressure_grad_x_times_shape_value_times_JxW
	      -2./3*time_step*(2*nonlinearity_un*shape_value_tn[i]*JxW_tn
			       -nonlinearity_unm1*shape_value_tnm1[i]*JxW_tnm1);
	    cell_rhs_v(i)+=
	      4./3*rhon*vn[q]*shape_value_tn[i]*JxW_tn-1./3*rhonm1*vnm1[q]*shape_value_tnm1[i]*JxW_tnm1
	      +2./3*time_step*force_vnp1*shape_value_tnp1[i]*JxW_tnp1
	      -2./3*time_step*pressure_grad_y_times_shape_value_times_JxW
	      -2./3*time_step*(2*nonlinearity_vn*shape_value_tn[i]*JxW_tn
			       -nonlinearity_vnm1*shape_value_tnm1[i]*JxW_tnm1);
	    if (rebuild_Matrix_U==true)
	      for (unsigned int j=0; j<dofs_per_cell; ++j) 
		{
		  cell_A_u(i,j)+=(rhonp1*shape_value_tnp1[i]*shape_value_tnp1[j]
				  +2./3*time_step*munp1*(shape_grad_tnp1[i]*shape_grad_tnp1[j])
				  )*JxW_tnp1;
		  cell_A_v(i,j)+=(rhonp1*shape_value_tnp1[i]*shape_value_tnp1[j]
				  +2./3*time_step*munp1*(shape_grad_tnp1[i]*shape_grad_tnp1[j])
				  )*JxW_tnp1;
		}
	  }
      }
      cell_U->get_dof_indices(local_dof_indices);
      // distribute
      if (rebuild_Matrix_U==true) 
	{
	  constraints.distribute_local_to_global(cell_A_u,local_dof_indices,system_Matrix_u);
	  constraints.distribute_local_to_global(cell_A_v,local_dof_indices,system_Matrix_v);
	}
      constraints.distribute_local_to_global(cell_rhs_u,local_dof_indices,system_rhs_u);
      constraints.distribute_local_to_global(cell_rhs_v,local_dof_indices,system_rhs_v);
    }
  system_rhs_u.compress(VectorOperation::add);
  system_rhs_v.compress(VectorOperation::add);
  if (rebuild_Matrix_U==true) 
    {
      system_Matrix_u.compress(VectorOperation::add);
      system_Matrix_v.compress(VectorOperation::add);
    }
  // BOUNDARY CONDITIONS
  system_rhs_u.set(boundary_values_id_u,boundary_values_u);
  system_rhs_u.compress(VectorOperation::insert);
  system_rhs_v.set(boundary_values_id_v,boundary_values_v);
  system_rhs_v.compress(VectorOperation::insert);
  if (rebuild_Matrix_U)
    {
      system_Matrix_u.clear_rows(boundary_values_id_u,1);
      system_Matrix_v.clear_rows(boundary_values_id_v,1);
      if (rebuild_Matrix_U_preconditioners)
	{
	  rebuild_Matrix_U_preconditioners=false;
	  // PRECONDITIONERS
	  preconditioner_Matrix_u.reset(new PETScWrappers::PreconditionBoomerAMG
					(system_Matrix_u,PETScWrappers::PreconditionBoomerAMG::AdditionalData(false)));
	  preconditioner_Matrix_v.reset( new PETScWrappers::PreconditionBoomerAMG
					 (system_Matrix_v,PETScWrappers::PreconditionBoomerAMG::AdditionalData(false)));
	}
    }
  rebuild_Matrix_U=true;
}

template<int dim>
void NavierStokesSolver<dim>::get_non_differentiable_mesh_velocity_at_tnp1(double dt, double t)
{
  // compute mesh velocity at tnp1 based on xnp1, xn and vn
  // loop locally owned dofs (in Q1)
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_U_mesh.begin();
  for (;idofs_iter!=locally_owned_dofs_U_mesh.end(); idofs_iter++)
   { 
     unsigned int gi = *idofs_iter;     
     // read position at tn and tnp1
     double xni = xn_coord(gi);
     double xnp1i = xnp1_coord(gi);
     double yni = yn_coord(gi);
     double ynp1i = ynp1_coord(gi);

     // get non-differentiable mesh velocity in x
     double unp1i = (xnp1i-xni)/dt;
     double vnp1i = (ynp1i-yni)/dt;

     //std::cout << uni << "\t" << unp1i << std::endl;
     completely_distributed_mesh_velocity_u(gi) = unp1i;
     completely_distributed_mesh_velocity_v(gi) = vnp1i;     
   }
  completely_distributed_mesh_velocity_u.compress(VectorOperation::insert);
  completely_distributed_mesh_velocity_v.compress(VectorOperation::insert);
  locally_relevant_mesh_velocity_un = completely_distributed_mesh_velocity_u;
  locally_relevant_mesh_velocity_vn = completely_distributed_mesh_velocity_v;
}

template<int dim>
void NavierStokesSolver<dim>::get_differentiable_mesh_velocity_at_tnp1(double dt, double t)
{
  // compute mesh velocity at tnp1 based on xnp1 and xn
  // loop locally owned dofs (in Q1)
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_U_mesh.begin();
  for (;idofs_iter!=locally_owned_dofs_U_mesh.end(); idofs_iter++)
   { 
     unsigned int gi = *idofs_iter;     
     // read velocity at tn and position at tn and tnp1
     double uni = locally_relevant_mesh_velocity_un(gi);
     double xni = xn_coord(gi);
     double xnp1i = xnp1_coord(gi);
     double vni = locally_relevant_mesh_velocity_vn(gi);
     double yni = yn_coord(gi);
     double ynp1i = ynp1_coord(gi);

     // get differentiable mesh velocity in x
     double unp1i = (2*(xnp1i-xni)-dt*uni)/dt;
     double vnp1i = (2*(ynp1i-yni)-dt*vni)/dt;

     //std::cout << uni << "\t" << unp1i << std::endl;
     completely_distributed_mesh_velocity_u(gi) = unp1i;
     completely_distributed_mesh_velocity_v(gi) = vnp1i;     
   }
  completely_distributed_mesh_velocity_u.compress(VectorOperation::insert);
  completely_distributed_mesh_velocity_v.compress(VectorOperation::insert);
  locally_relevant_mesh_velocity_un = completely_distributed_mesh_velocity_u;
  locally_relevant_mesh_velocity_vn = completely_distributed_mesh_velocity_v;
}

template<int dim>
void NavierStokesSolver<dim>::get_differentiable_mesh_velocity_at_tnp1_v2(double dt, double t)
{
  // compute mesh velocity at tnp1 based on xnp1 and xn
  // loop locally owned dofs (in Q1)
  IndexSet::ElementIterator idofs_iter = locally_owned_dofs_U_mesh.begin();
  for (;idofs_iter!=locally_owned_dofs_U_mesh.end(); idofs_iter++)
   { 
     unsigned int gi = *idofs_iter;     
     // read data at tn and tnp1
     double xni = xn_coord(gi);
     double xnp1i = xnp1_coord(gi);
     double uni = locally_relevant_mesh_velocity_un(gi);
     double udotni = locally_relevant_mesh_velocity_udotn(gi);

     double yni = yn_coord(gi);
     double ynp1i = ynp1_coord(gi);
     double vni = locally_relevant_mesh_velocity_vn(gi);
     double vdotni = locally_relevant_mesh_velocity_vdotn(gi);

     // get differentiable mesh velocity
     double unp1i = -(4*dt*uni+std::pow(dt,2)*udotni+6*xni-6*xnp1i)/(2.*dt);
     double vnp1i = -(4*dt*vni+std::pow(dt,2)*vdotni+6*yni-6*ynp1i)/(2.*dt);

     double udotnp1i = (-2*(3*dt*uni+std::pow(dt,2)*udotni+3*xni-3*xnp1i))/std::pow(dt,2);
     double vdotnp1i = (-2*(3*dt*vni+std::pow(dt,2)*vdotni+3*yni-3*ynp1i))/std::pow(dt,2);

     //std::cout << uni << "\t" << unp1i << std::endl;
     completely_distributed_mesh_velocity_u(gi) = unp1i;
     completely_distributed_mesh_velocity_v(gi) = vnp1i;
     completely_distributed_mesh_velocity_udot(gi) = udotnp1i;
     completely_distributed_mesh_velocity_vdot(gi) = vdotnp1i;
   }
  completely_distributed_mesh_velocity_u.compress(VectorOperation::insert);
  completely_distributed_mesh_velocity_v.compress(VectorOperation::insert);
  completely_distributed_mesh_velocity_udot.compress(VectorOperation::insert);
  completely_distributed_mesh_velocity_vdot.compress(VectorOperation::insert);
  locally_relevant_mesh_velocity_un = completely_distributed_mesh_velocity_u;
  locally_relevant_mesh_velocity_vn = completely_distributed_mesh_velocity_v;
  locally_relevant_mesh_velocity_udotn = completely_distributed_mesh_velocity_udot;
  locally_relevant_mesh_velocity_vdotn = completely_distributed_mesh_velocity_vdot;
}

template<int dim>
void NavierStokesSolver<dim>::assemble_system_dpsi_q_BDF1() {
  system_S=0;
  system_rhs_psi=0;
  
  const QGauss<dim> quadrature_formula(degree_MAX+1);
  FEValues<dim> fe_values_LS(mapping_tnp1,fe_LS,quadrature_formula,
			    update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_U(mapping_tnp1,fe_U,quadrature_formula,
			    update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_P(mapping_tnp1,fe_P,quadrature_formula,
			    update_values|update_gradients|update_quadrature_points|update_JxW_values);

  const unsigned int dofs_per_cell=fe_P.dofs_per_cell;
  const unsigned int n_q_points=quadrature_formula.size();

  FullMatrix<double> cell_S(dofs_per_cell,dofs_per_cell);
  Vector<double> cell_rhs_psi(dofs_per_cell);

  std::vector<double> phi_tnp1(n_q_points);
  std::vector<Tensor<1, dim> > grad_unp1(n_q_points);
  std::vector<Tensor<1, dim> > grad_vnp1(n_q_points);

  std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  std::vector<double> shape_value(dofs_per_cell);
  std::vector<Tensor<1, dim> > shape_grad(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator 
    cell_P=dof_handler_P.begin_active(), endc_P=dof_handler_P.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U=dof_handler_U.begin_active();
  typename DoFHandler<dim>::active_cell_iterator cell_LS=dof_handler_LS.begin_active();

  double rhonp1;
  for (; cell_P!=endc_P; ++cell_P,++cell_U,++cell_LS)
    if (cell_P->is_locally_owned()) {
      cell_S=0;
      cell_rhs_psi=0;

      fe_values_LS.reinit(cell_LS);
      fe_values_P.reinit(cell_P);
      fe_values_U.reinit(cell_U);
      // get function grads for u and v
      fe_values_LS.get_function_values(locally_relevant_solution_phi,phi_tnp1);
      fe_values_U.get_function_gradients(locally_relevant_solution_u,grad_unp1);
      fe_values_U.get_function_gradients(locally_relevant_solution_v,grad_vnp1);

      for (unsigned int q=0; q<n_q_points; ++q) {
	const double JxW=fe_values_P.JxW(q);
	double divU = grad_unp1[q][0]+grad_vnp1[q][1];

	for (unsigned int i=0; i<dofs_per_cell; ++i) 
	  shape_grad[i]=fe_values_P.shape_grad(i,q);

	if (LEVEL_SET==1)
	  get_rho(rhonp1,phi_tnp1[q]);
	else
	  {
	    rho_function.set_time(time_tnp1);
	    rhonp1=rho_function.value(fe_values_U.quadrature_point(q));
	  }

	for (unsigned int i=0; i<dofs_per_cell; ++i) 
	  {
	    cell_rhs_psi(i)+=-1./time_step*divU*fe_values_P.shape_value(i,q)*JxW;
	    for (unsigned int j=0; j<dofs_per_cell; ++j) 
	      cell_S(i,j)+=shape_grad[i]*shape_grad[j]/rhonp1*JxW;
	  }
      }
      cell_P->get_dof_indices(local_dof_indices);
      // Distribute
      constraints_psi.distribute_local_to_global(cell_S,local_dof_indices,system_S);
      constraints_psi.distribute_local_to_global(cell_rhs_psi,local_dof_indices,system_rhs_psi);
    }
  system_S.compress(VectorOperation::add);
  if (rebuild_S_M_preconditioners)
    {
      rebuild_S_M_preconditioners=false;
      preconditioner_S.reset(new PETScWrappers::PreconditionBoomerAMG
			     (system_S,PETScWrappers::PreconditionBoomerAMG::AdditionalData(true)));
    }
  system_rhs_psi.compress(VectorOperation::add);
}

template<int dim>
void NavierStokesSolver<dim>::assemble_extend_pressure_TMP() {
  system_rhs_q=0;  
  const QGauss<dim> quadrature_formula(degree_MAX+1);
  FEValues<dim> fe_values_P(mapping_tn,fe_P,quadrature_formula,
			    update_values|update_gradients|update_quadrature_points|update_JxW_values);

  const unsigned int dofs_per_cell=fe_P.dofs_per_cell;
  const unsigned int n_q_points=quadrature_formula.size();

  Vector<double> cell_rhs_q(dofs_per_cell);
  std::vector<double> p_tn(n_q_points);

  std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  std::vector<double> shape_value(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator 
    cell_P=dof_handler_P.begin_active(), endc_P=dof_handler_P.end();

  for (; cell_P!=endc_P; ++cell_P)
    if (cell_P->is_locally_owned()) 
      {
	cell_rhs_q=0;
	fe_values_P.reinit(cell_P);
	// get function grads for u and v
	fe_values_P.get_function_values(locally_relevant_solution_p,p_tn);     
	for (unsigned int q=0; q<n_q_points; ++q) 
	  for (unsigned int i=0; i<dofs_per_cell; ++i) 
	    cell_rhs_q(i)+=p_tn[q]*fe_values_P.shape_value(i,q)*fe_values_P.JxW(q);
	// Distribute
	cell_P->get_dof_indices(local_dof_indices);
	constraints_q.distribute_local_to_global(cell_rhs_q,local_dof_indices,system_rhs_q);
      }
  system_rhs_q.compress(VectorOperation::add);
}

template<int dim>
void NavierStokesSolver<dim>::assemble_system_dpsi_q_BDF2() {
  if (rebuild_S_M==true) 
    {
      system_S=0;
      system_M=0;
    }
  system_rhs_psi=0;
  system_rhs_q=0;
  
  const QGauss<dim> quadrature_formula(degree_MAX+1);

  FEValues<dim> fe_values_U(mapping_tnp1,fe_U,quadrature_formula,
			    update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_P(mapping_tnp1,fe_P,quadrature_formula,
			    update_values|update_gradients|update_quadrature_points|update_JxW_values);
  FEValues<dim> fe_values_LS(mapping_tnp1,fe_LS,quadrature_formula,
			     update_values|update_gradients|update_quadrature_points|update_JxW_values);

  const unsigned int dofs_per_cell=fe_P.dofs_per_cell;
  const unsigned int n_q_points=quadrature_formula.size();

  FullMatrix<double> cell_S(dofs_per_cell,dofs_per_cell);
  FullMatrix<double> cell_M(dofs_per_cell,dofs_per_cell);
  Vector<double> cell_rhs_psi(dofs_per_cell);
  Vector<double> cell_rhs_q(dofs_per_cell);

  std::vector<double> phiqnp1(n_q_points);
  std::vector<Tensor<1, dim> > gunp1(n_q_points);
  std::vector<Tensor<1, dim> > gvnp1(n_q_points);
  std::vector<Tensor<1, dim> > gwnp1(n_q_points);

  std::vector<unsigned int> local_dof_indices(dofs_per_cell);
  std::vector<double> shape_value(dofs_per_cell);
  std::vector<Tensor<1, dim> > shape_grad(dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator 
    cell_P=dof_handler_P.begin_active(), endc_P=dof_handler_P.end();
  typename DoFHandler<dim>::active_cell_iterator cell_U=dof_handler_U.begin_active();
  typename DoFHandler<dim>::active_cell_iterator cell_LS=dof_handler_LS.begin_active();
  
  double rhonp1;
  for (; cell_P!=endc_P; ++cell_P,++cell_U,++cell_LS)
    if (cell_P->is_locally_owned()) {
      cell_S=0;
      cell_M=0;
      cell_rhs_psi=0;
      cell_rhs_q=0;

      fe_values_P.reinit(cell_P);
      fe_values_U.reinit(cell_U);
      fe_values_LS.reinit(cell_LS);

      // get function values for LS
      fe_values_LS.get_function_values(locally_relevant_solution_phi,phiqnp1);

      // get function grads for u and v
      fe_values_U.get_function_gradients(locally_relevant_solution_u,gunp1);
      fe_values_U.get_function_gradients(locally_relevant_solution_v,gvnp1);
      if (dim==3)
	fe_values_U.get_function_gradients(locally_relevant_solution_w,gwnp1);

      for (unsigned int q_point=0; q_point<n_q_points; ++q_point) 
	{
	  const double JxW=fe_values_P.JxW(q_point);
	  double divU = gunp1[q_point][0]+gvnp1[q_point][1];
	  if (dim==3) divU += gwnp1[q_point][2]; 
	  for (unsigned int i=0; i<dofs_per_cell; ++i) {
	    shape_value[i]=fe_values_P.shape_value(i,q_point);
	    shape_grad[i]=fe_values_P.shape_grad(i,q_point);
	  }
	if (LEVEL_SET==1) // use level set to define rho and nu
	  get_rho_and_nu (phiqnp1[q_point]);
	else // rho and nu are defined through functions
	  {
	    rho_function.set_time(time_tnp1);
	    nu_function.set_time(time_tnp1);
	    rhonp1=rho_function.value(fe_values_U.quadrature_point(q_point));
	    nu_value=nu_function.value(fe_values_U.quadrature_point(q_point));
	  }
	for (unsigned int i=0; i<dofs_per_cell; ++i) 
	  {
	    cell_rhs_psi(i)-=3./2./time_step*divU*shape_value[i]*JxW;
	    cell_rhs_q(i)-=nu_value*divU*shape_value[i]*JxW;
	    if (rebuild_S_M==true)
	      for (unsigned int j=0; j<dofs_per_cell; ++j) 
		{
		  cell_S(i,j)+=shape_grad[i]*shape_grad[j]/rhonp1*JxW;
		  cell_M(i,j)+=shape_value[i]*shape_value[j]*JxW;
		}
	  }
      }
      cell_P->get_dof_indices(local_dof_indices);
      // Distribute
      if (rebuild_S_M==true) {
	constraints_psi.distribute_local_to_global(cell_S,local_dof_indices,system_S);
	constraints_q.distribute_local_to_global(cell_M,local_dof_indices,system_M);
      }
      constraints_psi.distribute_local_to_global(cell_rhs_psi,local_dof_indices,system_rhs_psi);
      constraints_q.distribute_local_to_global(cell_rhs_q,local_dof_indices,system_rhs_q);
    }
  if (rebuild_S_M==true) 
    {
      system_M.compress(VectorOperation::add);
      system_S.compress(VectorOperation::add);
      if (rebuild_S_M_preconditioners)
	{
	  rebuild_S_M_preconditioners=false;
	  preconditioner_S.reset(new PETScWrappers::PreconditionBoomerAMG
				 (system_S,PETScWrappers::PreconditionBoomerAMG::AdditionalData(true)));
	  preconditioner_M.reset(new PETScWrappers::PreconditionBoomerAMG
				 (system_M,PETScWrappers::PreconditionBoomerAMG::AdditionalData(true)));
	}
    }
  system_rhs_psi.compress(VectorOperation::add);
  system_rhs_q.compress(VectorOperation::add);
  rebuild_S_M=true;
}

///////////////////////////////////////////////////////
/////////////////////// SOLVERS ///////////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::solve_U(const ConstraintMatrix &constraints,
				      PETScWrappers::MPI::SparseMatrix &Matrix,
				      std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
				      PETScWrappers::MPI::Vector &completely_distributed_solution,
				      const PETScWrappers::MPI::Vector &rhs) {
  SolverControl solver_control(dof_handler_U.n_dofs(),1e-6); // * uv_L2_norm);
  //PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
  PETScWrappers::SolverBicgstab solver(solver_control,mpi_communicator);
  constraints.distribute(completely_distributed_solution);
  solver.solve(Matrix,completely_distributed_solution,rhs,*preconditioner);
  constraints.distribute(completely_distributed_solution);
  if (solver_control.last_step() > MAX_NUM_ITER_TO_RECOMPUTE_PRECONDITIONER)
    rebuild_Matrix_U_preconditioners=true;
  if (verbose==true)
    pcout<<"   Solved U in "<<solver_control.last_step()<<" iterations."<<std::endl;
}

template<int dim>
void NavierStokesSolver<dim>::solve_P(const ConstraintMatrix &constraints,
				      PETScWrappers::MPI::SparseMatrix &Matrix,
				      std_cxx1x::shared_ptr<PETScWrappers::PreconditionBoomerAMG> preconditioner,
				      PETScWrappers::MPI::Vector &completely_distributed_solution,
				      const PETScWrappers::MPI::Vector &rhs) {
  SolverControl solver_control(dof_handler_P.n_dofs(),1e-6);	// * uv_L2_norm);
  //PETScWrappers::SolverCG solver(solver_control,mpi_communicator);
  PETScWrappers::SolverBicgstab solver(solver_control,mpi_communicator);
  constraints.distribute(completely_distributed_solution);
  solver.solve(Matrix,completely_distributed_solution,rhs,*preconditioner);
  constraints.distribute(completely_distributed_solution);
  if (solver_control.last_step() > MAX_NUM_ITER_TO_RECOMPUTE_PRECONDITIONER)
    rebuild_S_M_preconditioners=true;
  if (verbose==true)
    pcout<<"   Solved P in "<<solver_control.last_step()<<" iterations."<<std::endl;
}

///////////////////////////////////////////////////////
//////////////// get different fields /////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::get_velocity() 
{
  //assemble_system_U_BDF1();
  assemble_system_U_BDF2();
  save_old_solution();
  solve_U(constraints,system_Matrix_u,preconditioner_Matrix_u,completely_distributed_solution_u,system_rhs_u);
  locally_relevant_solution_u=completely_distributed_solution_u;
  solve_U(constraints,system_Matrix_v,preconditioner_Matrix_v,completely_distributed_solution_v,system_rhs_v);
  locally_relevant_solution_v=completely_distributed_solution_v; 
}

template<int dim>
void NavierStokesSolver<dim>::get_pressure() 
{
  // GET DPSI via BDF1 
  //assemble_system_dpsi_q_BDF1();
  //solve_P(constraints_psi,system_S,preconditioner_S,completely_distributed_solution_psi,system_rhs_psi);
  //locally_relevant_solution_psi=completely_distributed_solution_psi;
  // UPDATE THE PRESSURE
  //completely_distributed_solution_p=0; // FOR NON-INCREMENTAL
  //completely_distributed_solution_p.add(1,completely_distributed_solution_psi);
  //locally_relevant_solution_p = completely_distributed_solution_p;

  // GET DPSI via BDF2 INCREMENTAL
  assemble_system_dpsi_q_BDF2();
  solve_P(constraints_psi,system_S,preconditioner_S,completely_distributed_solution_psi,system_rhs_psi);
  locally_relevant_solution_psi=completely_distributed_solution_psi;
  // Extended pressure 
  //assemble_extend_pressure_TMP();
  //solve_P(constraints_q,system_M,preconditioner_M,completely_distributed_solution_q,system_rhs_q);
  //completely_distributed_solution_p.equ(1,completely_distributed_solution_q);
  //std::cout << completely_distributed_solution_p.min() << ", " << completely_distributed_solution_p.max() << std::endl;
  // SOLVE Q (ROTATIONAL PART)
  //solve_P(constraints_q,system_M,preconditioner_M,completely_distributed_solution_q,system_rhs_q);
  // UPDATE THE PRESSURE
  completely_distributed_solution_p.add(1,completely_distributed_solution_psi);
  //completely_distributed_solution_p.add(1,completely_distributed_solution_q);
  locally_relevant_solution_p = completely_distributed_solution_p;

  // UPDATE PRESSURE VIA TRANSPORTING 
  //transport_pressure.set_mesh_velocity(locally_relevant_mesh_velocity_un,
  //			       locally_relevant_mesh_velocity_vn);
  //locally_relevant_mesh_velocity_un*=-1.;
  //locally_relevant_mesh_velocity_vn*=-1.;
  //transport_pressure.initial_condition(locally_relevant_solution_p,
  //			       locally_relevant_mesh_velocity_un,
  //			       locally_relevant_mesh_velocity_vn);
  //locally_relevant_mesh_velocity_un*=-1.;
  //locally_relevant_mesh_velocity_vn*=-1.;
  //transport_pressure.nth_time_step(0.5*time_step);
  //transport_pressure.nth_time_step(0.5*time_step);
  //transport_pressure.get_unp1(locally_relevant_solution_p);
  //completely_distributed_solution_p.equ(1.0,locally_relevant_solution_p);
  //completely_distributed_solution_p.add(1.0,locally_relevant_solution_psi);
  //locally_relevant_solution_p = completely_distributed_solution_p;

  // SUBSTRACT MEAN VALUE
  double mean_value = VectorTools::compute_mean_value(mapping_tnp1,dof_handler_P,QGauss<dim>(degree_MAX+1),
						      locally_relevant_solution_p,0);
  completely_distributed_solution_p.add(-mean_value);
  locally_relevant_solution_p = completely_distributed_solution_p;
}

///////////////////////////////////////////////////////
/////////////////////// DO STEPS //////////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::nth_time_step(double time_step) 
{  
  this->time_step = time_step;
  // UPDATE TIME 
  time_tnm1=time_tn;
  time_tn=time_tnp1;
  time_tnp1+=time_step;
  // GET VELOCITY AND PRESSURE
  get_velocity(); 
  //get_pressure();
  // get (non-)differentiable mesh velocity at next time step
  locally_relevant_mesh_velocity_unm1=locally_relevant_mesh_velocity_un;
  locally_relevant_mesh_velocity_vnm1=locally_relevant_mesh_velocity_vn;
  get_differentiable_mesh_velocity_at_tnp1(time_step,time_tn);  
}

///////////////////////////////////////////////////////
//////////////////////// OTHERS ///////////////////////
///////////////////////////////////////////////////////
template<int dim>
void NavierStokesSolver<dim>::save_old_solution() 
{
  locally_relevant_solution_u_old=locally_relevant_solution_u;
  locally_relevant_solution_v_old=locally_relevant_solution_v;
  locally_relevant_solution_w_old=locally_relevant_solution_w;
  locally_relevant_solution_psi_old=locally_relevant_solution_psi;
}

