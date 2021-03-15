//mhdtoy
//Greg Szypko

#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "Eigen/Core"

#define K_B 1.3807e-16 //boltzmann constant, erg K^-1
#define M_I 1.6726e-24 //ion mass, g
#define GRAV 2.748e4 //acceleration due to gravity at surface, cm sec^-2
#define BASE_GRAV 2.748e4
#define R_SUN 6.957e10 //radius of sun, cm
#define GAMMA 1.666667 //adiabatic index
#define KAPPA_0 1.0e-6 //thermal conductivity coefficient
#define TEMP_CHROMOSPHERE 3.0e4 //chromospheric temperature, K (for radiation purposes)
#define RADIATION_RAMP 1.0e3 //width of falloff for low-temperature radiation, K
#define B0 100.0 //strength of B field at base of domain, Gauss
#define HEATING_RATE 1.0e-4 //uniform volumetric heating rate, erg cm^-3 s^-1
#define PI 3.14159265358979323846

#define XDIM 100
#define YDIM 100
#define CHROMOSPHERE_DEPTH 10 //number of cells deep to maintain chromospheric temperature

// #define DX 0.1
// #define DY 0.1
#define DX 2.2649e9/XDIM
#define DY 2.2649e9/YDIM
#define NT 500
#define OUTPUT_INTERVAL 20 //time steps between file outputs

#define EPSILON 0.1 //dynamic time stepping safety factor
#define EPSILON_THERMAL 0.3 //safety factor for thermal conduction (<0.5)
#define EPSILON_VISCOUS 1.0 //controls strength of artificial viscosity
#define EPSILON_RADIATIVE 0.1 //max fraction of change in energy allowed for single radiative loss cycle
#define GRIDFLOOR 1.0e-25 //min value for non-negative parameters

//Boundary condition labels
#define PERIODIC 0
#define WALL 1
#define OPEN 2

#define XBOUND1 0
#define XBOUND2 0
#define YBOUND1 1
#define YBOUND2 2

//Output variables (0 to turn off output, 1 to turn on output)
#define RHO_OUT 1
#define TEMP_OUT 1
#define PRESS_OUT 1
#define RAD_OUT 1
#define ENERGY_OUT 0
#define VX_OUT 1
#define VY_OUT 1

using Grid = Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic>;

//Compute surface values from cell-centered values using Barton's method
//Meant to be used for transport terms only
//Result indexed s.t. element i,j indicates surface between i,j and i-1,j
//if "index"==0, or i,j and i,j-1 if "index"==1
Grid upwind_surface(const Grid &cell_center, const Grid &vel, const int index){
  int xdim = XDIM+1-index;
  int ydim = YDIM+index;
  Grid cell_surface = Grid::Zero(xdim,ydim);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < xdim; i++){
    for(int j = 0; j < ydim; j++){
      //Handle direction of cell_center being considered (i.e. index for differencing)
      int i2 = i, j2 = j;
      int i0, i1, i3, j0, j1, j3;
      if(index == 0){
        //Handle X boundary conditions
        j0 = j2; j1 = j2; j3 = j2;
        i0 = i2-2; i1 = i2-1; i3 = i2+1;
        //ENFORCES PERIODIC X-BOUNDARIES
        if(XBOUND1 == PERIODIC && XBOUND2 == PERIODIC){
          if(i2 == XDIM) continue;
          //Here, explicitly need macro XDIM instead of xdim
          i0 = (i0+XDIM)%XDIM;
          i1 = (i1+XDIM)%XDIM;
          i3 = (i3+XDIM)%XDIM;
        }
        if(XBOUND1 == WALL){
          if(i2 == 0){
            cell_surface(i2,j2) = 0.0;
            continue;
          } else if(i2 == 1){
            i0 = i1;
          }
        } else if(XBOUND1 == OPEN){
          if(i2 == 0){
            cell_surface(i2,j2) = 1.5*cell_center(i2,j2) - 0.5*cell_center(i3,j3); //lerp
            continue;
          } else if(i2 == 1){
            i0 = i1;
          }
        }
        if(XBOUND2 == WALL){
          if(i2 == XDIM){
            cell_surface(i2,j2) = 0.0;
            continue;
          } else if(i2 == XDIM - 1){
            i3 = i2;
          }
        } else if(XBOUND2 == OPEN){
          if(i2 == XDIM){
            cell_surface(i2,j2) = 1.5*cell_center(i1,j1) - 0.5*cell_center(i0,j0); //lerp
            continue;
          } else if(i2 == XDIM - 1){
            i3 = i2;
          }
        }
      }
      else{
        //Handle Y boundary conditions
        i0 = i2; i1 = i2; i3 = i2;
        j0 = j2-2; j1 = j2-1; j3 = j2+1;
        if(YBOUND1 == PERIODIC && YBOUND2 == PERIODIC){
          if(j2 == YDIM) continue;
          //Here, explicitly need macro YDIM instead of ydim
          j0 = (j0+YDIM)%YDIM;
          j1 = (j1+YDIM)%YDIM;
          j3 = (j3+YDIM)%YDIM;
        }
        if(YBOUND1 == WALL){
          if(j2 == 0){
            cell_surface(i2,j2) = 0.0;
            continue;
          } else if(j2 == 1){
            j0 = j1;
          }
        } else if(YBOUND1 == OPEN){
          if(j2 == 0){
            cell_surface(i2,j2) = 1.5*cell_center(i2,j2) - 0.5*cell_center(i3,j3); //lerp
            continue;
          } else if(j2 == 1){
            j0 = j1;
          }
        }
        if(YBOUND2 == WALL){
          if(j2 == YDIM){
            cell_surface(i2,j2) = 0.0;
            continue;
          } else if(j2 == YDIM - 1){
            j3 = j2;
          }
        } else if(YBOUND2 == OPEN){
          if(j2 == YDIM){
            cell_surface(i2,j2) = 1.5*cell_center(i1,j1) - 0.5*cell_center(i0,j0); //lerp
            continue;
          } else if(j2 == YDIM - 1){
            j3 = j2;
          }
        }
      }
      //Apply Barton's method
      double d1, d2, d3;
      d2 = 0.5*(cell_center(i1,j1)+cell_center(i2,j2));
      if(0.5*(vel(i1,j1)+vel(i2,j2))>=0.0){
        d3 = cell_center(i1,j1);
        d1 = 1.5*cell_center(i1,j1) - 0.5*cell_center(i0,j0);
        if(cell_center(i2,j2) <= cell_center(i1,j1)){
          cell_surface(i2,j2) = std::min(d3,std::max(d1,d2));
        } else { //cell_center(i2,j2) > cell_center(i1,j1)
          cell_surface(i2,j2) = std::max(d3,std::min(d1,d2));
        }
      } else { //vel(i1,j1)<0.0
        d3 = cell_center(i2,j2);
        d1 = 1.5*cell_center(i2,j2) - 0.5*cell_center(i3,j3);
        if(cell_center(i2,j2) <= cell_center(i1,j1)){
          cell_surface(i2,j2) = std::max(d3,std::min(d1,d2));
        } else { //cell_center(i2,j2) > cell_center(i1,j1)
          cell_surface(i2,j2) = std::min(d3,std::max(d1,d2));
        }
      }
    }
  }
  return cell_surface;
}

Grid transport_derivative1D(const Grid &quantity, const Grid &vel, const int index){
  Grid surf_quantity = upwind_surface(quantity, vel, index);
  int xdim = quantity.rows();
  int ydim = quantity.cols();
  double denom = DX*(1-index) + DY*(index);
  Grid div = Grid::Ones(xdim,ydim);
  #pragma omp parallel for collapse(2)
  for(int i=0; i<xdim; i++){
    for(int j=0; j<ydim; j++){
      int i0, i1, i2, i2surf, j0, j1, j2, j2surf; //Need separate indices for surface and vel
      i1 = i; j1 = j;
      if(index == 0){
        //Handle X boundary conditions
        j0 = j1; j2 = j1; j2surf = j1;
        i0 = i1-1; i2 = i1+1; i2surf = i1+1;
        if(XBOUND1 == PERIODIC && XBOUND2 == PERIODIC){
          i0 = (i0+XDIM)%XDIM;
          i2 = (i2+XDIM)%XDIM;
          i2surf = (i2surf+XDIM)%XDIM;
        }
        if(XBOUND1 == WALL){
          // if(i1 == 0 || i1 == 1){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(i1 == 0){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(XBOUND1 == OPEN){
          if(i1 == 0) i0 = i1;
        }
        if(XBOUND2 == WALL){
          // if(i1 == XDIM-1 || i1 == XDIM-2){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(i1 == XDIM-1){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(XBOUND2 == OPEN){
          if(i1 == XDIM-1) i2 = i1;
        }
        // if(XBOUND1 == WALL || XBOUND1 == OPEN){
        //   if(i1 == 0) i0 = i1;
        // }
        // if(XBOUND2 == WALL || XBOUND2 == OPEN){
        //   if(i1 == XDIM-1) i2 = i1;
        // }
      }
      else{
        //Handle Y boundary conditions
        i0 = i1; i2 = i1; i2surf = i1;
        j0 = j1-1; j2 = j1+1; j2surf = j1+1;
        if(YBOUND1 == PERIODIC && YBOUND2 == PERIODIC){
          j0 = (j0+YDIM)%YDIM;
          j2 = (j2+YDIM)%YDIM;
          j2surf = (j2surf+YDIM)%YDIM;
        }
        if(YBOUND1 == WALL){
          // if(j1 == 0 || j1 == 1){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(j1 == 0){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(YBOUND1 == OPEN){
          if(j1 == 0) j0 = j1;
        }
        if(YBOUND2 == WALL){
          // if(j1 == YDIM-1 || j1 == YDIM-2){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(j1 == YDIM-1){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(YBOUND2 == OPEN){
          if(j1 == YDIM-1) j2 = j1;
        }
        // if(YBOUND1 == WALL || YBOUND1 == OPEN){
        //   if(j1 == 0) j0 = j1;
        // }
        // if(YBOUND2 == WALL || YBOUND2 == OPEN){
        //   if(j1 == YDIM-1) j2 = j1;
        // }
      }
      div(i1,j1) = (surf_quantity(i2surf,j2surf)*0.5*(vel(i1,j1)+vel(i2,j2))
                - surf_quantity(i1,j1)*0.5*(vel(i1,j1)+vel(i0,j0)))/denom;
    }
  }
  return div;
}

//Compute single-direction divergence term for non-transport term (central differencing)
Grid derivative1D(const Grid &quantity, const int index){
  int xdim = quantity.rows();
  int ydim = quantity.cols();
  Grid div = Grid::Zero(xdim,ydim);
  double denom = 2.0*(DX*(1-index) + DY*(index));
  #pragma omp parallel for collapse(2)
  for(int i=0; i<xdim; i++){
    for(int j=0; j<ydim; j++){
      int i0, i1, i2, j0, j1, j2;
      i1 = i; j1 = j;
      if(index == 0){
        //Handle X boundary conditions
        j0 = j1; j2 = j1;
        i0 = i1-1; i2 = i1+1;
        //ENFORCES PERIODIC X-BOUNDARIES
        if(XBOUND1 == PERIODIC && XBOUND2 == PERIODIC){
          i0 = (i0+xdim)%xdim;
          i2 = (i2+xdim)%xdim;
        }
        if(XBOUND1 == WALL){
          // if(i1 == 0 || i1 == 1){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(i1 == 0){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(XBOUND1 == OPEN){
          if(i1 == 0) i0 = i1;
        }
        if(XBOUND2 == WALL){
          // if(i1 == XDIM-1 || i1 == XDIM-2){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(i1 == XDIM-1){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(XBOUND2 == OPEN){
          if(i1 == XDIM-1) i2 = i1;
        }
      }
      else{
        //Handle Y boundary conditions
        i0 = i1; i2 = i1;
        j0 = j1-1; j2 = j1+1;
        if(YBOUND1 == PERIODIC && YBOUND2 == PERIODIC){
          j0 = (j0+ydim)%ydim;
          j2 = (j2+ydim)%ydim;
        }
        if(YBOUND1 == WALL){
          // if(j1 == 0 || j1 == 1){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(j1 == 0){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(YBOUND1 == OPEN){
          if(j1 == 0) j0 = j1;
        }
        if(YBOUND2 == WALL){
          // if(j1 == YDIM-1 || j1 == YDIM-2){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(j1 == YDIM-1){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(YBOUND2 == OPEN){
          if(j1 == YDIM-1) j2 = j1;
        }
      }
      div(i1,j1) = (quantity(i2,j2) - quantity(i0,j0))/denom;
    }
  }
  return div;
}

//Compute divergence term for simulation parameter "quantity"
//"quantity","vx","vy" used for transport term
//Non-transport terms contained in "nontransp_x", "nontransp_y"
Grid divergence(const Grid &quantity, const Grid &nontransp_x, const Grid &nontransp_y, const Grid &vx, const Grid &vy){
  // Eigen::IOFormat one_line_format(4, Eigen::DontAlignCols, ",", ";", "", "", "", "\n");
  Grid result = transport_derivative1D(quantity, vx, 0) + transport_derivative1D(quantity, vy, 1);
  if(nontransp_x.size() > 1) result += derivative1D(nontransp_x, 0);
  if(nontransp_y.size() > 1) result += derivative1D(nontransp_y, 1);
  // std::cout << result.matrix().format(one_line_format);
  return result;
}

//Enforce dynamic time stepping
double recompute_dt(const Grid &press, const Grid &rho, const Grid &vx, const Grid &vy){
  double running_min_dt = std::numeric_limits<double>::max();
  #pragma omp parallel for collapse(2)
  for(int i=0; i<XDIM; i++){
    for(int j=0; j<YDIM; j++){
      double c_s = std::sqrt(GAMMA*press(i,j)/rho(i,j));
      double abs_vx = DX/(c_s+std::abs(vx(i,j)));
      double abs_vy = DY/(c_s+std::abs(vy(i,j)));
      running_min_dt = std::min(running_min_dt, abs_vx);
      running_min_dt = std::min(running_min_dt, abs_vy);
    }
  }
  return running_min_dt;
}

//Enforce dynamic time stepping for thermal conduction
double recompute_dt_thermal(const Grid &rho, const Grid &temp){
  double running_min_dt = std::numeric_limits<double>::max();
  #pragma omp parallel for collapse(2)
  for(int i=0; i<XDIM; i++){
    for(int j=0; j<YDIM; j++){
      double this_dt = K_B/KAPPA_0*(rho(i,j)/M_I)*DX*DY/std::pow(temp(i,j),2.5);
      running_min_dt = std::min(running_min_dt, this_dt);
    }
  }
  return running_min_dt;
}

//Enforce minimum dynamic time step for radiation
double recompute_dt_radiative(const Grid &energy, const Grid &rad_loss_rate){
  double running_min_dt = std::numeric_limits<double>::max();
  #pragma omp parallel for collapse(2)
  for(int i=0; i<XDIM; i++){
    for(int j=0; j<YDIM; j++){
      if(rad_loss_rate(i,j) > 0.0){
        double this_dt = std::abs(energy(i,j)/rad_loss_rate(i,j));
        running_min_dt = std::min(running_min_dt, this_dt);
      }
    }
  }
  return running_min_dt;
}


//Generates gaussian initial condition for a variable, centered at middle of grid
Grid GaussianGrid(int xdim, int ydim, double min, double max){
  Eigen::VectorXd gauss_x(xdim), gauss_y(ydim);
  double sigmax = 0.05*xdim;
  double sigmay = 0.05*ydim;
  for(int i=0; i<xdim; i++){
    gauss_x(i) = std::exp(-0.5*std::pow(((double)i-0.5*(double)(xdim-1))/sigmax,2.0));
  }
  for(int j=0; j<ydim; j++){
    gauss_y(j) = std::exp(-0.5*std::pow(((double)j-0.5*(double)(ydim-1))/sigmay,2.0));
  }
  Eigen::MatrixXd gauss_2d = gauss_x * gauss_y.transpose();
  gauss_2d = gauss_2d*(max-min) + min*Eigen::MatrixXd::Ones(xdim,ydim);
  return gauss_2d.array();
}

//Generates potential bipolar field for component corresponding to index "index"
//Centered s.t. origin lies at bottom middle of domain
//Pressure scale height h, field poles at +/- l, field strength at poles b0
Grid BipolarField(const int xdim, const int ydim, const double b0, const double h, const int index){
  Grid result = Grid::Zero(xdim, ydim);
  for(int i=0; i<xdim; i++){
    for(int j=0; j<ydim; j++){
      double x = (i - (double)(xdim-1)*0.5)*DX;
      double y = j*DY;
      if(index == 0) result(i,j) = b0*std::exp(-0.5*y/h)*std::cos(0.5*x/h);
      else result(i,j) = -b0*std::exp(-0.5*y/h)*std::sin(0.5*x/h);
    }
  }
  return result;
}

//Generates grid with exponential falloff in the y-direction, with the quantity
//"base_value" at y=0. Assumes isothermal atmosphere with temperature "iso_temp".
Grid HydrostaticFalloff(const double base_value, const double scale_height, const int xdim, const int ydim){
  Grid result = Grid::Zero(xdim, ydim);
  for(int i=0; i<xdim; i++){
    for(int j=0; j<ydim; j++){
      double y = j*DY;
      result(i,j) = base_value*std::exp(-y/scale_height);
    }
  }
  return result;
}

//Generates Grid containing magnitude of gravitational
//acceleration (in y-direction) at each grid cell
Grid Gravity(const double base_grav, const double r_sun, const int xdim, const int ydim){
  Grid result = Grid::Zero(xdim,ydim);
  for(int j=0; j<ydim; j++){
    double y = j*DY;
    for(int i=0; i<xdim; i++){
      result(i,j) = base_grav*std::pow(r_sun/(r_sun+y),2.0);
    }
  }
  return result;
}

//Compute single-direction second derivative
Grid second_derivative1D(const Grid &quantity, const int index){
  int xdim = quantity.rows();
  int ydim = quantity.cols();
  Grid div = Grid::Zero(xdim,ydim);
  double denom = std::pow((DX*(1-index) + DY*(index)),2.0);
  #pragma omp parallel for collapse(2)
  for(int i=0; i<xdim; i++){
    for(int j=0; j<ydim; j++){
      int i0, i1, i2, j0, j1, j2;
      i1 = i; j1 = j;
      if(index == 0){
        //Handle X boundary conditions
        j0 = j1; j2 = j1;
        i0 = i1-1; i2 = i1+1;
        //ENFORCES PERIODIC X-BOUNDARIES
        if(XBOUND1 == PERIODIC && XBOUND2 == PERIODIC){
          i0 = (i0+xdim)%xdim;
          i2 = (i2+xdim)%xdim;
        }
        if(XBOUND1 == WALL){
          // if(i1 == 0 || i1 == 1){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(i1 == 0){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(XBOUND1 == OPEN){
          if(i1 == 0) i0 = i1;
        }
        if(XBOUND2 == WALL){
          // if(i1 == XDIM-1 || i1 == XDIM-2){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(i1 == XDIM-1){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(XBOUND2 == OPEN){
          if(i1 == XDIM-1) i2 = i1;
        }
      }
      else{
        //Handle Y boundary conditions
        i0 = i1; i2 = i1;
        j0 = j1-1; j2 = j1+1;
        if(YBOUND1 == PERIODIC && YBOUND2 == PERIODIC){
          j0 = (j0+ydim)%ydim;
          j2 = (j2+ydim)%ydim;
        }
        if(YBOUND1 == WALL){
          // if(j1 == 0 || j1 == 1){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(j1 == 0){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(YBOUND1 == OPEN){
          if(j1 == 0) j0 = j1;
        }
        if(YBOUND2 == WALL){
          // if(j1 == YDIM-1 || j1 == YDIM-2){
          //   div(i1,j1) = 0.0;
          //   continue;
          // }
          if(j1 == YDIM-1){
            div(i1,j1) = 0.0;
            continue;
          }
        }
        if(YBOUND2 == OPEN){
          if(j1 == YDIM-1) j2 = j1;
        }
      }
      div(i1,j1) = (quantity(i2,j2) - 2.0*quantity(i1,j1) + quantity(i0,j0))/denom;
    }
  }
  return div;
}

//Computes Laplacian (del squared) of "quantity"
Grid laplacian(const Grid &quantity){
  Grid result_x = second_derivative1D(quantity,0);
  Grid result_y = second_derivative1D(quantity,1);
  return result_x+result_y;
}

//Computes cell-centered conductive flux from temperature "temp"
//Flux computed in direction indicated by "index": 0 for x, 1 for y
//k0 is conductive coefficient
Grid conductive_flux(const Grid &temp, const double k0, const int index){
  int xdim = temp.rows();
  int ydim = temp.cols();
  Grid flux = Grid::Zero(xdim,ydim);
  #pragma omp parallel for collapse(2)
  for(int i=0; i<xdim; i++){
    for(int j=0; j<ydim; j++){
      flux(i,j) = std::pow(temp(i,j),7.0/2.0);
    }
  }
  return -2.0/7.0*k0*derivative1D(flux,index);
}

Grid radiative_losses(const Grid &rho, const Grid &temp, const int xdim, const int ydim){
  Grid result = Grid::Zero(xdim,ydim);
  #pragma omp parallel for collapse(2)
  for(int i=0; i<xdim; i++){
    for(int j=0; j<ydim; j++){
      if(temp(i,j) < TEMP_CHROMOSPHERE){
        result(i,j) = 0.0;
        continue;
      }
      double logtemp = std::log10(temp(i,j));
      double n = rho(i,j)/M_I;
      double chi, alpha;
      if(logtemp <= 4.97){
        chi = 1.09e-31;
        alpha = 2.0;
      } else if(logtemp <= 5.67){
        chi = 8.87e-17;
        alpha = -1.0;
      } else if(logtemp <= 6.18){
        chi = 1.90e-22;
        alpha = 0.0;
      } else if(logtemp <= 6.55){
        chi = 3.53e-13;
        alpha = -1.5;
      } else if(logtemp <= 6.90){
        chi = 3.46e-25;
        alpha = 1.0/3.0;
      } else if(logtemp <= 7.63){
        chi = 5.49e-16;
        alpha = -1.0;
      } else{
        chi = 1.96e-27;
        alpha = 0.5;
      }
      result(i,j) = n*n*chi*std::pow(temp(i,j),alpha);
      if(temp(i,j) < TEMP_CHROMOSPHERE + RADIATION_RAMP){
        double ramp = 0.5*(1.0 - std::cos((temp(i,j) - TEMP_CHROMOSPHERE)*PI/RADIATION_RAMP));
        result(i,j) *= ramp;
      }
      if(result(i,j) < 0.0) std::cout << "oh no! " << result(i,j) << std::endl;
     }
  }
  return result;
}

int main(int argc,char* argv[]){
  auto start_time = std::chrono::steady_clock::now();

  Eigen::IOFormat one_line_format(4, Eigen::DontAlignCols, ",", ";", "", "", "", "\n");
  std::ofstream out_file;
  out_file.open ("output.txt");
  out_file << XDIM << "," << YDIM << std::endl;

  Grid rho, mom_x, mom_y, temp, press, energy, bx, by, bz, grav;

  grav = Gravity(BASE_GRAV, R_SUN, XDIM, YDIM);

  double b0 = B0; //magnetic field strength at base in G
  double iso_temp = TEMP_CHROMOSPHERE; //isothermal initial temp in K
  double base_rho = M_I*1.0e12; //initial mass density at base, g cm^-3
  double scale_height = 2.0*K_B*iso_temp/(M_I*BASE_GRAV);
  bx = BipolarField(XDIM, YDIM, b0, scale_height, 0);
  by = BipolarField(XDIM, YDIM, b0, scale_height, 1);
  bz = Grid::Zero(XDIM,YDIM);

  Grid mag_press = (bx*bx + by*by + bz*bz)/(8.0*PI);
  Grid mag_pxx = (-bx*bx + by*by + bz*bz)/(8.0*PI);
  Grid mag_pyy = (bx*bx - by*by + bz*bz)/(8.0*PI);
  Grid mag_pzz = (bx*bx + by*by - bz*bz)/(8.0*PI);
  Grid mag_pxy = -bx*by/(4.0*PI);
  Grid mag_pxz = -bx*bz/(4.0*PI);
  Grid mag_pyz = -by*bz/(4.0*PI);

  // //Simple Gaussian test case
  // rho = GaussianGrid(XDIM, YDIM, 1.0, 5.0);
  // mom_x = Grid::Zero(XDIM,YDIM); //x momentum density
  // mom_y = Grid::Zero(XDIM,YDIM); //y momentum density
  // temp = GaussianGrid(XDIM, YDIM, 1.0, 2.0); //temperature
  // double b0 = 0.1;
  // double h = (XDIM*DX)/(4.0*PI);
  // bx = BipolarField(XDIM, YDIM, b0, h, 0);
  // by = BipolarField(XDIM, YDIM, b0, h, 1);
  // bz = Grid::Zero(XDIM,YDIM);

  //Isothermal hydrostatic initial condition
  rho = HydrostaticFalloff(base_rho,scale_height,XDIM,YDIM);
  mom_x = Grid::Zero(XDIM,YDIM); //x momentum density
  mom_y = Grid::Zero(XDIM,YDIM); //y momentum density
  temp = iso_temp*Grid::Ones(XDIM,YDIM); //temperature

  press = 2.0*K_B*rho*temp/M_I;
  energy = press/(GAMMA - 1.0) + 0.5*(mom_x*mom_x + mom_y*mom_y)/rho + mag_press;

  out_file << bx.matrix().format(one_line_format);
  out_file << by.matrix().format(one_line_format);

  //Output intial state
  double t=0.0;
  double dt=1.0;
  double dt_thermal=1.0;
  double dt_radiative=1.0;
  double e_viscous=1.0;
  out_file << "t=" << t << std::endl;
  if(RHO_OUT){
    out_file << "rho\n";
    out_file << rho.matrix().format(one_line_format);
  }
  if(TEMP_OUT){
    out_file << "temp\n";
    out_file << temp.matrix().format(one_line_format);
  }
  if(PRESS_OUT){
    out_file << "press\n";
    out_file << press.matrix().format(one_line_format);
  }
  if(RAD_OUT){
    out_file << "rad\n";
    out_file << Grid::Zero(XDIM,YDIM).matrix().format(one_line_format);
  }
  if(ENERGY_OUT){
    out_file << "energy\n";
    out_file << energy.matrix().format(one_line_format);
  }
  if(VX_OUT){
    out_file << "vel_x\n";
    out_file << (mom_x/rho).matrix().format(one_line_format);
  }
  if(VY_OUT){
    out_file << "vel_y\n";
    out_file << (mom_y/rho).matrix().format(one_line_format);
  }

  for (int iter = 0; iter < NT; iter++){
    // Enforce rigid boundaries
    if(YBOUND1 == WALL || YBOUND2 == WALL){
      if(YBOUND1 == WALL) for(int i=0; i<XDIM; i++){
        mom_x(i,0) = 0.0;
        mom_y(i,0) = 0.0;
      }
      if(YBOUND2 == WALL) for(int i=0; i<XDIM; i++){
        mom_x(i,YDIM-1) = 0.0;
        mom_y(i,YDIM-1) = 0.0;
      }
    }
    if(XBOUND1 == WALL || XBOUND2 == WALL){
      if(XBOUND1 == WALL) for(int j=0; j<YDIM; j++){
        mom_x(0,j) = 0.0;
        mom_y(0,j) = 0.0;
      }
      if(XBOUND2 == WALL) for(int j=0; j<YDIM; j++){
        mom_x(XDIM-1,j) = 0.0;
        mom_y(XDIM-1,j) = 0.0;
      }
    }

    // Enforce constant chromospheric temperature
    for(int i=0; i<XDIM; i++){
      for(int j=0; j<CHROMOSPHERE_DEPTH; j++){
        temp(i,j) = TEMP_CHROMOSPHERE;
      }
    }

    //Compute values needed for time evolution
    Grid vx = mom_x/rho;
    Grid vy = mom_y/rho;
    press = 2.0*K_B*rho*temp/M_I;
    energy = press/(GAMMA - 1.0) + 0.5*(mom_x*vx + mom_y*vy) + mag_press;
    // Grid press = (GAMMA - 1.0)*(energy - 0.5*(mom_x*vx + mom_y*vy));

    //Compute time steps
    double dt_raw = recompute_dt(press, rho, vx, vy);
    e_viscous = EPSILON_VISCOUS*0.5*DX*DY/dt; 
    dt = EPSILON*dt_raw;
    dt_thermal = EPSILON_THERMAL*recompute_dt_thermal(rho, temp);
    Grid rad_loss_rate = radiative_losses(rho, temp, XDIM, YDIM);
    dt_radiative = EPSILON_RADIATIVE*recompute_dt_radiative(energy, rad_loss_rate);

    //Subcycle to simulate thermal diffusion
    Grid energy_relaxed = energy;
    int subcycles_conduct = (int)(dt/dt_thermal)+1;
    for(int subcycle = 0; subcycle < subcycles_conduct; subcycle++){
      Grid con_flux_x = conductive_flux(temp, KAPPA_0, 0);
      Grid con_flux_y = conductive_flux(temp, KAPPA_0, 1);
      energy_relaxed = energy_relaxed - (dt/(double)subcycles_conduct)*(derivative1D(con_flux_x,0)+derivative1D(con_flux_y,1));
      press = (GAMMA - 1.0)*(energy_relaxed - 0.5*(mom_x*vx + mom_y*vy) - mag_press);
      temp = M_I*press/(2.0*K_B*rho);
    }

    //Subcycle to simulate radiative losses
    int subcycles_radiate = (int)(dt/dt_radiative)+1;
    for(int subcycle = 0; subcycle < subcycles_radiate; subcycle++){
      // Enforce constant chromospheric temperature
      for(int i=0; i<XDIM; i++){
        for(int j=0; j<CHROMOSPHERE_DEPTH; j++){
          temp(i,j) = TEMP_CHROMOSPHERE;
        }
      }
      Grid losses = radiative_losses(rho, temp, XDIM, YDIM);
      if(subcycle>0) rad_loss_rate += losses;
      energy_relaxed = energy_relaxed - (dt/(double)subcycles_radiate)*losses;
      press = (GAMMA - 1.0)*(energy_relaxed - 0.5*(mom_x*vx + mom_y*vy) - mag_press);
      temp = M_I*press/(2.0*K_B*rho);
    }
    rad_loss_rate /= subcycles_radiate; //Average loss rate over all subcycles for plotting purposes


    //Advance time by dt
    Grid viscous_force_x = EPSILON_VISCOUS*(0.5*DX*DX/dt_raw)*laplacian(mom_x);
    Grid viscous_force_y = EPSILON_VISCOUS*(0.5*DY*DY/dt_raw)*laplacian(mom_y);
    Grid zero = Grid::Zero(1,1);
    Grid rho_next = rho - dt*divergence(rho,zero,zero,vx,vy);
    Grid mom_x_next = mom_x - dt*divergence(mom_x, press + mag_pxx, mag_pxy, vx, vy) + dt*viscous_force_x;
    Grid mom_y_next = mom_y - dt*divergence(mom_y, mag_pxy, press + mag_pyy, vx, vy) - dt*rho*grav + dt*viscous_force_y;
    // Grid energy_next = energy_relaxed - dt*divergence(energy_relaxed+press, mag_pxx*vx + mag_pxy*vy, mag_pxy*vx + mag_pyy*vy, vx, vy) 
    //                     - dt*rho*vy*grav + dt*(vx*viscous_force_x + vy*viscous_force_y) + dt*HEATING_RATE;
    Grid energy_next = energy_relaxed - dt*divergence(energy_relaxed+press, mag_pxx*vx + mag_pxy*vy, mag_pxy*vx + mag_pyy*vy, vx, vy) 
                        + dt*(vx*viscous_force_x + vy*viscous_force_y) + dt*HEATING_RATE;
    

    //Clamping wall boundary values
    if(YBOUND1 == WALL || YBOUND2 == WALL){
      if(YBOUND1 == WALL) for(int i=0; i<XDIM; i++){
        mom_x_next(i,0) = 0.0;
        mom_y_next(i,0) = 0.0;
        rho_next(i,0) = rho(i,0);
        energy_next(i,0) = energy(i,0);
      }
      if(YBOUND2 == WALL) for(int i=0; i<XDIM; i++){
        mom_x_next(i,YDIM-1) = 0.0;
        mom_y_next(i,YDIM-1) = 0.0;
        rho_next(i,YDIM-1) = rho(i,YDIM-1);
        energy_next(i,YDIM-1) = energy(i,YDIM-1);
      }
    }
    if(XBOUND1 == WALL || XBOUND2 == WALL){
      if(XBOUND1 == WALL) for(int j=0; j<YDIM; j++){
        mom_x_next(0,j) = 0.0;
        mom_y_next(0,j) = 0.0;
        rho_next(0,j) = rho(0,j);
        energy_next(0,j) = energy(0,j);
      }
      if(XBOUND2 == WALL) for(int j=0; j<YDIM; j++){
        mom_x_next(XDIM-1,j) = 0.0;
        mom_y_next(XDIM-1,j) = 0.0;
        rho_next(XDIM-1,j) = rho(XDIM-1,j);
        energy_next(XDIM-1,j) = energy(XDIM-1,j);
      }
    }

    //Sanity checks
    for(int i=0; i<XDIM; i++){
      for(int j=0; j<YDIM; j++){
        //Ensure no negative densities
        rho_next(i,j) = std::max(GRIDFLOOR, rho_next(i,j));
        //Ensure no negative pressure/temperature
        double bulk_kin_energy = 0.5*(mom_x_next(i,j)*mom_x_next(i,j)
                                    + mom_y_next(i,j)*mom_y_next(i,j))/rho_next(i,j);
        energy_next(i,j) = std::max(bulk_kin_energy + mag_press(i,j), energy_next(i,j));
      }
    }
    //Swap current and next pointers
    rho = rho_next;
    mom_x = mom_x_next;
    mom_y = mom_y_next;
    press = (GAMMA - 1.0)*(energy_next - 0.5*(mom_x*vx + mom_y*vy) - mag_press);
    temp = M_I*press/(2.0*K_B*rho);

    t = t + dt;
    // if(iter%10 == 0){
      // printf("\rIter: %i/%i|dt: %f|Cond. Subcyc: %i|Rad. Subcyc: %i\n", iter, NT,dt,subcycles_conduct,subcycles_radiate);
      // fflush(stdout);
    // }

    // std::cout << subcycles_conduct << std::endl;

    if(iter%OUTPUT_INTERVAL == 0){
      out_file << "t=" << t << std::endl;
      if(RHO_OUT){
        out_file << "rho\n";
        out_file << rho.matrix().format(one_line_format);
      }
      if(TEMP_OUT){
        out_file << "temp\n";
        out_file << temp.matrix().format(one_line_format);
      }
      if(PRESS_OUT){
        out_file << "press\n";
        out_file << press.matrix().format(one_line_format);
      }
      if(RAD_OUT){
        out_file << "rad\n";
        out_file << rad_loss_rate.matrix().format(one_line_format);
      }
      if(ENERGY_OUT){
        out_file << "energy\n";
        out_file << energy.matrix().format(one_line_format);
      }
      if(VX_OUT){
        out_file << "vel_x\n";
        out_file << (mom_x/rho).matrix().format(one_line_format);
      }
      if(VY_OUT){
        out_file << "vel_y\n";
        out_file << (mom_y/rho).matrix().format(one_line_format);
      }
    }
  }
  std::cout << "\rIterations: " << NT << "/" << NT << "\n";
  auto stop_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop_time - start_time);
  double minutes = (int)(duration.count()/60.0);
  double seconds = duration.count() - 60*minutes;
  out_file << "runtime=" << minutes << "min" << seconds << "sec";
  std::cout << "\rTotal runtime: " << minutes << " min " << seconds << " sec (approx. " 
    << (double)duration.count()/(double)NT << " sec per iteration)" << std::endl;
  out_file.close();
}