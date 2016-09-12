///////////////////////////////////////////////////////
//////////////////// INITIAL PHI ////////////////////
///////////////////////////////////////////////////////
template <int dim>
class InitialPhi : public Function <dim>
{
public:
  InitialPhi (unsigned int PROBLEM, double sharpness=0.005) : Function<dim>(),
								sharpness(sharpness),
								PROBLEM(PROBLEM) {}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  double sharpness;
  unsigned int PROBLEM;
};
template <int dim>
double InitialPhi<dim>::value (const Point<dim> &p,
				 const unsigned int) const
{
  double x = p[0]; double y = p[1];
  double return_value = -1.;

  if (PROBLEM==CIRCULAR_ROTATION)
    {
      double x0=0.5; double y0=0.75;
      double r0=0.15;
      double r = std::sqrt(std::pow(x-x0,2)+std::pow(y-y0,2));
      return_value = -std::tanh((r-r0)/sharpness);
    }
  else // (PROBLEM==DIAGONAL_ADVECTION)
    {
      double x0=0.25; double y0=0.25;
      double r0=0.15;
      double r = std::sqrt(std::pow(x-x0,2)+std::pow(y-y0,2));
      return_value = -std::tanh((r-r0)/sharpness);
    }
  return return_value;
}

/////////////////////////////////////////////////////
//////////////////// BOUNDARY PHI ///////////////////
/////////////////////////////////////////////////////
template <int dim>
class BoundaryPhi : public Function <dim>
{
public:
  BoundaryPhi (double t=0) 
    : 
    Function<dim>() 
  {this->set_time(t);}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
};

template <int dim>
double BoundaryPhi<dim>::value (const Point<dim> &p, const unsigned int) const
{
  return -1.0;
}

///////////////////////////////////////////////////////
//////////////////// EXACT VELOCITY ///////////////////
///////////////////////////////////////////////////////
template <int dim>
class ExactU : public Function <dim>
{
public:
  ExactU (unsigned int PROBLEM, double time=0) : Function<dim>(), PROBLEM(PROBLEM), time(time) {}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  void set_time(double time){this->time=time;};
  unsigned PROBLEM;
  double time;
};

template <int dim>
double ExactU<dim>::value (const Point<dim> &p, const unsigned int) const
{
  if (PROBLEM==CIRCULAR_ROTATION)
    return -2*numbers::PI*(p[1]-0.5);
  else // (PROBLEM==DIAGONAL_ADVECTION)
    return 1.0;
}

template <int dim>
class ExactV : public Function <dim>
{
public:
  ExactV (unsigned int PROBLEM, double time=0) : Function<dim>(), PROBLEM(PROBLEM), time(time) {}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  void set_time(double time){this->time=time;};
  unsigned int PROBLEM;
  double time;
};

template <int dim>
double ExactV<dim>::value (const Point<dim> &p, const unsigned int) const
{
  if (PROBLEM==CIRCULAR_ROTATION)
    return 2*numbers::PI*(p[0]-0.5);
  else // (PROBLEM==DIAGONAL_ADVECTION)
    return 1.0;
}

//////////////////////////////////////////////////////
//////////////////// MESH VELOCITY ///////////////////
//////////////////////////////////////////////////////
template <int dim>
class MeshVelocityU : public Function<dim>
{
public:
  MeshVelocityU (unsigned int MESH_VELOCITY, double time=0) : Function<dim>(), 
							 MESH_VELOCITY(MESH_VELOCITY), time(time){}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  void set_time(double time){this->time=time;};
  unsigned MESH_VELOCITY;
  double time;
};

template <int dim>
double MeshVelocityU<dim>::value (const Point<dim> &p, const unsigned int) const
{
  double x = p[0]; double y = p[1];
  double pi = numbers::PI;

  if (MESH_VELOCITY==SINUSOIDAL_WITH_FIXED_BOUNDARY)
    return std::sin(pi*x)*std::cos(pi*y)*std::cos(2*pi*time);
  else if (MESH_VELOCITY==CIRCULAR)
    {
      return -2*pi*(y-0.5);
      // VIA IMAGE ROTATION
      //th=2*pi*dt;
      //return (std::cos(th)*x-std::sin(th)*y-x)/dt;
    }
  else if (MESH_VELOCITY==DIAGONAL)
    return 1.;
  else //ZERO VELOCITY
    return 0.;
}

template <int dim>
class MeshVelocityV : public Function<dim>
{
public:
  MeshVelocityV (unsigned int MESH_VELOCITY, double time=0) : Function<dim>(), 
							 MESH_VELOCITY(MESH_VELOCITY), time(time){}
  virtual double value (const Point<dim> &p, const unsigned int component=0) const;
  void set_time(double time){this->time=time;};
  unsigned MESH_VELOCITY;
  double time;
};

template <int dim>
double MeshVelocityV<dim>::value (const Point<dim> &p, const unsigned int) const
{
  double x = p[0]; double y = p[1];
  double pi = numbers::PI;

  if (MESH_VELOCITY==SINUSOIDAL_WITH_FIXED_BOUNDARY)
    return -std::cos(pi*x)*std::sin(pi*y)*std::cos(2*pi*time);      
  else if (MESH_VELOCITY==CIRCULAR)
    {
      return 2*pi*(x-0.5);
      // VIA IMAGE ROTATION
      //th=2*pi*dt;
      //return (std::sin(th)*x+std::cos(th)*y-y)/dt;
    }
  else if (MESH_VELOCITY==DIAGONAL)
    return 1.;
  else //ZERO VELOCITY
    return 0.;
}

///////////////////////////////////////////////////////
/////////////////// POST-PROCESSING ///////////////////
///////////////////////////////////////////////////////
template <int dim>
class Postprocessor : public DataPostprocessorScalar <dim>
{
public:
  Postprocessor(double eps, double rho_air, double rho_fluid)
    :
    DataPostprocessorScalar<dim>("Density",update_values)
  {
    this->eps=eps;
    this->rho_air=rho_air;
    this->rho_fluid=rho_fluid;
  }
  virtual void compute_derived_quantities_scalar (const std::vector< double > &uh,
						  const std::vector< Tensor< 1, dim > > &duh,
						  const std::vector< Tensor< 2, dim > > &dduh,
						  const std::vector< Point< dim > > &normals,
						  const std::vector< Point< dim > > &evaluation_points,
						  std::vector< Vector< double > > &computed_quantities 
						  ) const;
  double eps;
  double rho_air;
  double rho_fluid;
};

template <int dim>
void Postprocessor<dim>::compute_derived_quantities_scalar(const std::vector< double > &uh,
							   const std::vector< Tensor< 1, dim > > & /*duh*/,
							   const std::vector< Tensor< 2, dim > > & /*dduh*/,
							   const std::vector< Point< dim > > & /*normals*/,
							   const std::vector< Point< dim > > & /*evaluation_points*/,
							   std::vector< Vector< double > > &computed_quantities) const
{
  const unsigned int n_quadrature_points = uh.size();
  for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      double H;
      double rho_value;
      double phi_value=uh[q];
      if(phi_value > eps) 
	H=1;
      else if (phi_value < -eps) 
	H=-1;
      else 
	H=phi_value/eps;
      rho_value = rho_fluid*(1+H)/2. + rho_air*(1-H)/2.;
      computed_quantities[q] = rho_value;
    }
}
