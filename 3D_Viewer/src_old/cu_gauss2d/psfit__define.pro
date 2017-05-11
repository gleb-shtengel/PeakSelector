;
;  file:  psfit__define.pro
;
;  Particle swarm curve fitting to a function of one or two
;  variables.
;
;  RTK, 29-Mar-2007
;  Last update:  14-Sep-2009
;
;  Requires:
;    ParticleSwarm class _or_ Particle_Swarm DLM
;
;  Use:
;    The user must supply x and y vectors and the name of a
;    fit function.  The fit function must accept *two* arguments:
;    the x values given and a vector of parameter values.
;
;    For example, to fit to p[0]*x^2 + p[1]*x + p[2] define
;    the function to be:
;
;    function f, x, p
;      return, p[0]*x^2 + p[1]*x + p[2]
;    end
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


;
;  Private:
;

;------------------------------------------------------
;  PSFIT::ReducedCHI2_XY
;+
;  The arguments are a set of parameters, use these to
;  call the user-supplied function on the given data
;  and then compute the reduced chi-square.
;
;@private
;-
function PSFIT::ReducedCHI2_XY, params
  compile_opt idl2, HIDDEN
  on_error, 2
  
  ;  Calculate the user's function
  zfit = call_function(self.f, *self.x, *self.y, params)
  
  ;  Return the reduced chi-square
  return, total((*self.cs)*(*self.z-zfit)^2) / double(n_elements(zfit) - self.n)
end


;------------------------------------------------------
;  PSFIT::ReducedCHI2
;+
;  The arguments are a set of parameters, use these to
;  call the user-supplied function on the given data
;  and then compute the reduced chi-square.
;
;@private
;-
function PSFIT::ReducedCHI2, params
  compile_opt idl2, HIDDEN
  on_error, 2
  
  ;  Calculate the user's function
  yfit = call_function(self.f, *self.x, params)
  
  ;  Return the reduced chi-square
  return, total((*self.cs)*(*self.y-yfit)^2) / double(n_elements(yfit) - self.n)
end


;------------------------------------------------------
;  PSFIT::HaveDLM
;+
;  Return true if the Particle_Swarm DLM has been
;  loaded.
;
;@private
;-
function PSFIT::HaveDLM
  compile_opt idl2, HIDDEN

  catch, err
  if (err) then begin
    catch, /cancel
    return, 0b
  endif
  
  dlm_load, 'Particle_Swarm'
  return, 1b
end


;------------------------------------------------------
;  PSFIT::StepFit
;+
;  Do the fit step by step tracking the parameter 
;  history.
;
;@private
;-
function PSFIT::StepFit
  compile_opt idl2, HIDDEN
  on_error, 2
  
  if (self.dlm) then begin
    PS_Initialize
    imax = PS_GetProperty('MAX_ITERATIONS')
  endif else begin
    self.oSwarm->Initialize
    self.oSwarm->GetProperty, IMAX=imax
  endelse
  
  params = dblarr(self.n, imax)
  
  if (self.dlm) then begin
    for i=0, imax-1 do begin
      PS_Step
      params[*,i] = PS_GetProperty('GLOBAL_BEST')
    endfor
  endif else begin
    for i=0, imax-1 do begin
      self.oSwarm->Step
      self.oSwarm->GetProperty, GLOBAL_BEST=g
      params[*,i] = g
    endfor
  endelse
  
  return, params
end


;------------------------------------------------------
;  PSFIT::MSE_XY
;+
;  Calculate the MSE between the fit and the original data.
;
;@private
;-
function PSFIT::MSE_XY, params
  compile_opt idl2, HIDDEN
  on_error, 2

  zfit = call_function(self.f, *self.x, *self.y, params)  
  return, sqrt(total((*self.z-zfit)^2) / double(n_elements(zfit)))
end


;------------------------------------------------------
;  PSFIT::MSE
;+
;  Calculate the MSE between the fit and the original data.
;
;@private
;-
function PSFIT::MSE, params
  compile_opt idl2, HIDDEN
  on_error, 2

  yfit = call_function(self.f, *self.x, params)  
  return, sqrt(total((*self.y-yfit)^2) / double(n_elements(yfit)))
end


;------------------------------------------------------
;  PSFIT::Fit_XY
;+
;  Do the fit for a function of two variables.
;
;  @keyword ZFIT {in}{type=boolean}{optional}
;    If set return a set of fit values at the given
;    x,y positions.  Otherwise, return the calculated parameters.
;
;  @keyword PARAMS {out}{type=vector|array}{optional}
;    If present, output the calculated parameters.  A vector
;    if history not set, otherwise an array of the parameters
;    found at each step.
;
;@private
;-
function PSFIT::Fit_XY, ZFIT=zfit, PARAMS=params, ERROR=error
  compile_opt idl2, HIDDEN
  on_error, 2
  
  ;  No error
  error = ''
  
  ;  Set the object
  if (self.dlm) then  $
    PS_SetProperty, 'OBJECT', self
  
  ;  Validate
  if self.f eq '' then  $
    message, 'No function name supplied.'
  if self.n eq 0  then  $
    message, 'The number of parameters is not set.'
  if ~ptr_valid(self.x) then  $
    message, 'No X data values set.'
  if ~ptr_valid(self.y) then  $
    message, 'No Y data values set.'
  if ~ptr_valid(self.z) then  $
    message, 'No Z data values set.'
  if (n_elements(*self.x) ne n_elements(*self.y)) then  $
    message, 'X, Y and Z vectors of unequal length.'
  if (n_elements(*self.x) ne n_elements(*self.z)) then  $
    message, 'X, Y and Z vectors of unequal length.'
    
  ;  Weighting
  if (self.eqw) then begin
    ptr_free, [self.s, self.cs]
    self.s = ptr_new(replicate(1.0d, n_elements(*self.z)))
    self.cs = ptr_new(replicate(1.0d, n_elements(*self.z)))
  endif else begin
    if ~ptr_valid(self.s) then  $
      message, 'Weights must be set first.'
    if n_elements(*self.s) ne n_elements(*self.z) then  $
      message, 'Weights and Z vectors of unequal lengths.'
  endelse
    
  ;  Fit
  e_min = 1e38
  self.icount = 0
  if (self.history) then begin
    for i=0, self.imax-1 do begin
      self.icount++
      params = self->StepFit()
      p = params[*, (size(params,/DIM))[1]-1]
      e = self->MSE_XY(p)
      if (e lt e_min) then begin
        e_min = e
        best = p
      endif
      if (e lt self.tol) then  $
        break
    endfor
  endif else begin
    for i=0, self.imax-1 do begin
      self.icount++
      params = (self.dlm) ? PS_Optimize(ERROR=err)               $
                          : self.oSwarm->Optimize(ERROR=err)
      e = self->MSE_XY(params)
      if (e lt e_min) then begin
        e_min = e
        best = params
      endif
      if (e lt self.tol) then  $
        break
    endfor
  endelse
  
  ;  Check for non-convergence
  if (i eq self.imax) then  $
    error = 'Failed to reach tolerance after ' + strtrim(self.imax,2) + ' iterations.'
  
  if ~keyword_set(zfit) then begin
    if (self.history) then      $
      return, params            $
    else                        $
      return, best
  endif else begin
    if (self.history) then begin
      if (self.dlm) then begin
        imax = PS_GetProperty('MAX_ITERATIONS')
      endif else begin
        self.oSwarm->GetProperty, IMAX=imax
      endelse
      return, call_function(self.f, *self.x, *self.y, best)
    endif else begin
      return, call_function(self.f, *self.x, *self.y, best)
    endelse
  endelse
end


;------------------------------------------------------
;  PSFIT::Fit_X
;+
;  Do the fit for a function of one variable.
;
;  @keyword YFIT {in}{type=boolean}{optional}
;    If set return a set of fit values at the given
;    x positions.  Otherwise, return the calculated parameters.
;
;  @keyword PARAMS {out}{type=vector|array}{optional}
;    If present, output the calculated parameters.  A vector
;    if history not set, otherwise an array of the parameters
;    found at each step.
;
;@private
;-
function PSFIT::Fit_X, YFIT=yfit, PARAMS=params, ERROR=error
  compile_opt idl2, HIDDEN
  on_error, 2
  
  ;  No error
  error = ''
  
  ;  Set the object
  if (self.dlm) then  $
    PS_SetProperty, 'OBJECT', self
  
  ;  Validate
  if self.f eq '' then  $
    message, 'No function name supplied.'
  if self.n eq 0  then  $
    message, 'The number of parameters is not set.'
  if ~ptr_valid(self.x) then  $
    message, 'No X data values set.'
  if ~ptr_valid(self.y) then  $
    message, 'No Y data values set.'
  if (n_elements(*self.x) ne n_elements(*self.y)) then  $
    message, 'X and Y vectors of unequal length.'
    
  ;  Weighting
  if (self.eqw) then begin
    ptr_free, [self.s, self.cs]
    self.s = ptr_new(replicate(1.0d, n_elements(*self.y)))
    self.cs = ptr_new(replicate(1.0d, n_elements(*self.y)))
  endif else begin
    if ~ptr_valid(self.s) then  $
      message, 'Weights must be set first.'
    if n_elements(*self.s) ne n_elements(*self.y) then  $
      message, 'Weights and Y vectors of unequal lengths.'
  endelse
    
  ;  Fit
  e_min = 1e38
  self.icount = 0
  if (self.history) then begin
    for i=0, self.imax-1 do begin
      self.icount++
      params = self->StepFit()
      p = params[*, (size(params,/DIM))[1]-1]
      e = self->MSE(p)
      if (e lt e_min) then begin
        e_min = e
        best = p
      endif
      if (e lt self.tol) then  $
        break
    endfor
  endif else begin
    for i=0, self.imax-1 do begin
      self.icount++
      params = (self.dlm) ? PS_Optimize(ERROR=err)               $
                          : self.oSwarm->Optimize(ERROR=err)
      e = self->MSE(params)
      if (e lt e_min) then begin
        e_min = e
        best = params
      endif
      if (e lt self.tol) then  $
        break
    endfor
  endelse
  
  if ~keyword_set(yfit) then begin
    if (self.history) then      $
      return, params            $
    else                        $
      return, best
  endif else begin
    if (self.history) then begin
      if (self.dlm) then begin
        imax = PS_GetProperty('MAX_ITERATIONS')
      endif else begin
        self.oSwarm->GetProperty, IMAX=imax
      endelse
      return, call_function(self.f, *self.x, best)
    endif else begin
      return, call_function(self.f, *self.x, best)
    endelse
  endelse
end


;--------------------------------------------------------------
;  PSFIT::CalculateSigmas
;+
;  Calculate the sigmas for the most recent fit.
;-
function PSFIT::CalculateSigmas, xb, gb
    compile_opt idl2, logical_predicate

    ;  Calculate the distance between the global best and each particle best
    dims = size(xb, /DIM)
    p = gb # replicate(1.0, dims[1])
    dd = sqrt(total((xb-p)^2,1))

    ;  Sort
    order = sort(dd)
    ii = (lindgen(dims[1]))[order]
    dd = dd[order]
    w = dd[1:11]  ;  distance to ten particle bests nearest to the global best
    i = ii[1:11]  ;  particle numbers
    p = reform(xb[*,i])  ;  actual particle positions
    s = dblarr(dims[0])
    for k=0L, dims[0]-1 do  $
        s[k] = stddev(p[k,*])
    return, s
end


;
;  Public:
;

;------------------------------------------------------
;  PSFIT::Fit
;+
;  Fit the function to the data.
;-
function PSFIT::Fit, YFIT=yfit, ZFIT=zfit, PARAMS=params, ERROR=error
  compile_opt idl2
  
  if (self.xy) then begin
    return, self->Fit_XY(ZFIT=zfit, PARAMS=params, ERROR=error)
  endif else begin
    return, self->Fit_X(YFIT=yfit, PARAMS=params, ERROR=error)
  endelse
end


;------------------------------------------------------
;  PSFIT::SetProperty
;+
;  Set properties.  Expose a subset of the ParticleSwarm
;  properties, all with a PS_ prefix.
;-
pro PSFIT::SetProperty, F=f, X=x, Y=y, HISTORY=h, EQUAL_WEIGHTS=eqw, STATISTICAL_WEIGHTS=sw, WEIGHTS=s,    $
                        PS_W=ps_w, PS_C1=ps_c1, PS_C2=ps_c2, IMAX=imax, TOLERANCE=tol, Z=z,                $
                        PS_C3=ps_c3, PS_REPULSIVE=ps_rep, PS_IMAX=ps_imax, PS_VMAX=ps_vmax, PS_DOMAIN=ps_domain,  $
                        PS_CONSTRAIN_TO_DOMAIN=ps_constrain
  compile_opt idl2, logical_predicate
  on_error, 2
 
  ;
  ;  PSFit properties
  ;
  if n_elements(imax) ne 0 then  self.imax = imax
  if n_elements(tol) ne 0 then  self.tol = tol
  if n_elements(h) ne 0 then  self.history = (h ne 0)
  if n_elements(f) ne 0 then  self.f = f
  if n_elements(x) ne 0 then begin
    ptr_free, self.x
    self.x = ptr_new(x)
  endif
  if n_elements(y) ne 0 then begin
    ptr_free, self.y
    self.y = ptr_new(y)
  endif
  if n_elements(z) ne 0 then begin
    ptr_free, self.z
    self.z = ptr_new(z)
  endif

  ;  Statistical weights - for counts (Poisson distributed values)  
  if n_elements(sw) ne 0 then begin
    if ~ptr_valid(self.y) then  $
      message, 'Y values must be set before statistical weights.'
    ptr_free, [self.s, self.cs]
    self.s = ptr_new(sqrt(*self.y))
    self.cs = ptr_new(1.0d/(*self.y))
    self.eqw = 0
  endif
  
  ;  No weighting
  if n_elements(eqw) ne 0 then $
    self.eqw = (eqw ne 0)
  
  ;  User-supplied weights
  if n_elements(s) ne 0 then begin
    ptr_free, [self.s, self.cs]
    self.s  = ptr_new(s)
    self.cs = ptr_new(1.0d/(s*s))
    self.eqw = 0
  endif

  ;
  ;  ParticleSwarm properties
  ;
  if n_elements(ps_w) then begin
    if (self.dlm) then begin
      PS_SetProperty, 'W', ps_w
    endif else begin
      self.oSwarm->SetProperty, W=ps_w
    endelse
  endif
  
  if n_elements(ps_c1) then begin
    if (self.dlm) then begin
      PS_SetProperty, 'C1', ps_c1
    endif else begin
      self.oSwarm->SetProperty, C1=ps_c1
    endelse
  endif
  
  if n_elements(ps_c2) then begin
    if (self.dlm) then begin
      PS_SetProperty, 'C2', ps_c2
    endif else begin
      self.oSwarm->SetProperty, C2=ps_c2
    endelse
  endif
  
  if n_elements(ps_c3) then begin
    if (self.dlm) then begin
      PS_SetProperty, 'C3', ps_c3
    endif else begin
      self.oSwarm->SetProperty, C3=ps_c3
    endelse
  endif
  
  if n_elements(ps_rep) then begin
    if (self.dlm) then begin
      PS_SetProperty, 'REPULSIVE', ps_rep
    endif else begin
      self.oSwarm->SetProperty, REPULSIVE=ps_rep
    endelse
  endif
  
  if n_elements(ps_imax) then begin
    if (self.dlm) then begin
      PS_SetProperty, 'MAX_ITERATIONS', ps_imax
      PS_SetProperty, 'TOLERANCE', ps_imax
    endif else begin
      self.oSwarm->SetProperty, IMAX=ps_imax, TOLERANCE=ps_imax
    endelse
  endif
  
  if n_elements(ps_vmax) then begin
    if (self.dlm) then begin
      PS_SetProperty, 'MAX_VELOCITY', ps_vmax
    endif else begin
      self.oSwarm->SetProperty, VMAX=ps_vmax
    endelse
  endif
  
  if n_elements(ps_domain) then begin
    if (self.dlm) then begin
      PS_SetProperty, 'DOMAIN', ps_domain
    endif else begin
      self.oSwarm->SetProperty, DOMAIN=ps_domain
    endelse
  endif

  if n_elements(ps_constrain) then begin
    if (self.dlm) then begin
      PS_SetProperty, 'CONSTRAIN_TO_DOMAIN', ps_constrain
    endif else begin
      self.oSwarm->SetProperty, CONSTRAIN_TO_DOMAIN=ps_constrain
    endelse
  endif
end


;------------------------------------------------------
;  PSFIT::GetProperty
;+
;  Return properties.
;-
pro PSFIT::GetProperty, F=f, X=x, Y=y, Z=z, N=n, HISTORY=h, EQUAL_WEIGHTS=eqw, WEIGHTS=w, TWO_D=xy,  $
                        USING_DLM=dlm, PARTICLES=p, IMAX=imax, TOLERANCE=tol, ITERATIONS_DONE=icount, $
                        SIGMAS=sigmas
  compile_opt idl2
  on_error, 2
  
  if arg_present(z) then  z = *self.z
  if arg_present(xy) then  xy = self.xy
  if arg_present(icount) then  icount = self.icount
  if arg_present(imax) then imax = self.imax
  if arg_present(tol) then  tol = self.tol
  if arg_present(p) then  p = self.p
  if arg_present(eqw) then  eqw = self.eqw
  if arg_present(w) then  w = *self.s
  if arg_present(h) then  h = self.history
  if arg_present(n) then  n = self.n
  if arg_present(f) then  f = self.f
  if arg_present(x) then  x = *self.x
  if arg_present(y) then  y = *self.y
  if arg_present(dlm) then  dlm = self.dlm

  if arg_present(sigmas) then begin
    if (self.dlm) then begin
      xb = PS_GetProperty('PARTICLE_BEST')
      gb = PS_GetProperty('GLOBAL_BEST')
      sigmas = self->CalculateSigmas(xb, gb)
    endif else begin
      sigmas = -1
    endelse
  endif
end


;------------------------------------------------------
;  PSFIT::Cleanup
;+
;  Destructor.
;-
pro PSFIT::Cleanup
  compile_opt idl2
  on_error, 2
  
  ptr_free, [self.x, self.y, self.s, self.cs, self.z]
  if obj_valid(self.oSwarm) then  $
    obj_destroy, self.oSwarm
end


;------------------------------------------------------
;  PSFIT::Init
;+
;  Constructor.
;-
function PSFIT::Init, N=n, PARTICLES=p, TWO_D=xy, _EXTRA=extra
  compile_opt idl2
  on_error, 2

  ;  Must be given the dimensions and number of particles to use
  if (n_elements(n) eq 0) then  $
    message, 'The number of fit parameters must be given in N.'
  if (n_elements(p) eq 0) then  $
    message, 'The number of particles to use must be given in PARTICLES.'
  if (n_elements(xy) ne 0) then begin
    self.xy = keyword_set(xy)  ; 2D
  endif else begin
    self.xy = 0 ; 1D
  endelse
  
  self.n = n
  self.p = p
    
  ;  Use the DLM if available, fall back to the class if not
  if self->HaveDLM() then begin
    PS_Configure, n, p
    PS_SetProperty, 'METHOD_NAME', (self.xy) ? 'ReducedCHI2_XY' : 'ReducedCHI2'
    PS_SetProperty, 'MAX_ITERATIONS', 250
    PS_SetProperty, 'MAX_VELOCITY', 10
    PS_SetProperty, 'TOLERANCE', 250
    PS_SetProperty, 'W', [0.9,0.4]
    self.dlm = 1b
  endif else begin  
    ;  Create a single swarm object.  Don't destroy it w/o setting OBJECT to obj_new() first.
    self.oSwarm = obj_new('ParticleSwarm', METHOD= (self.xy) ? 'ReducedCHI2_XY' : 'ReducedCHI2',    $
                          OBJECT=self, IMAX=250, VMAX=10,  $
                          N_PARTICLES=p, DIMENSIONS=n, TOLERANCE=250, W=[0.9,0.4])
    self.dlm = 0b
  endelse

    ;  Set properties
  self->SetProperty, HISTORY=0, /EQUAL_WEIGHTS, IMAX=1, TOLERANCE=1e-4
  self->SetProperty, _EXTRA=extra
  
  return, 1
end


;------------------------------------------------------
;  psfit__define
;+
;  Class definition.
;-
pro psfit__define
  compile_opt idl2
  
  class = {PSFIT,  $
            f      : '',         $  ;  name of the fit function
            x      : ptr_new(),  $  ;  independent variable
            y      : ptr_new(),  $  ;  dependent variable
            z      : ptr_new(),  $  ;  z data for 2D fits
            xy     : 0b,         $  ;  flag, 1=2D fit
            s      : ptr_new(),  $  ;  uncertainty in the dependent variable
            cs     : ptr_new(),  $  ;  1.0/s^2, precomputed
            eqw    : 0b,         $  ;  use equal weights if set (ignore s)
            tol    : 0.0d,       $  ;  tolerance (reduced chi-square value)
            imax   : 0L,         $  ;  number of iterations max
            icount : 0L,         $  ;  number of iterations performed
            n      : 0L,         $  ;  number of dimensions in the fit
            p      : 0L,         $  ;  number of particles
            history: 0b,         $  ;  if set, track the parameter history of the fit
            dlm    : 0b,         $  ;  true if Particle_Swarm DLM is available
            oSwarm : obj_new()   $  ;  swarm object, if not DLM available
          }
end

;
;  end psfit__define.pro
;
