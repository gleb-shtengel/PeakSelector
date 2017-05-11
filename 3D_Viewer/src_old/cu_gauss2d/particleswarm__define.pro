;
;  file: particleswarm__define.pro
;
;  Global & Repulsive Particle Swarm Optimization class.
;
;  This implements the canonical and repulsive particle swarm optimization 
;  algorithms for numerical problems.
;
;  RTK, 01-Mar-2007
;  Last update:  10-Mar-2008
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;
;  Private:
;

;--------------------------------------------------------------
;  ParticleSwarm::RandomVectorInDomain
;+
;  Return a random vector in the current domain.
;
;@private
;-
function ParticleSwarm::RandomVectorInDomain
  compile_opt idl2, HIDDEN
  on_error, 2

  d = *self.domain
  v = dblarr(self.m)
  for i=0, self.m-1 do $
    v[i] = d[0,i] + randomu(seed)*(d[1,i]-d[0,i])
  return, v
end


;--------------------------------------------------------------
;  ParticleSwarm::GlobalUpdate
;+
;  Do one iteration step.
;
;@private
;-
pro ParticleSwarm::GlobalUpdate
  compile_opt idl2, HIDDEN
  on_error, 2

  ;  Track if global position updated in this step
  update = 0b

  ;  Look at each particle
  for i=0, self.n-1 do begin

    ;  Update X and V
    x  = reform((*self.x)[*,i])      ;  current position
    v  = reform((*self.v)[*,i])      ;  current velocity
    b  = reform((*self.b)[*,i])      ;  best position for this particle
    g  = reform(*self.g)             ;  global best for all particles
    w  = self.wm*self.i+(self.w)[0]  ;  inertial constant, decreases over time
    c1 = self.c1                     ;  cognitive component weight
    c2 = self.c2                     ;  social component weight
    r1 = randomu(seed, self.m)       ;  random cognitive vector
    r2 = randomu(seed, self.m)       ;  random social vector

    ;  Update to the next position   
    x = x + v

    ;  Constrain to domain, if set
    if (self.const) then begin
      for j=0, n_elements(x)-1 do begin
        d = reform((*self.domain)[*,j])
        if (x[j] lt d[0]) then  x[j] = d[0]
        if (x[j] gt d[1]) then  x[j] = d[1]
      endfor
    endif

    ;  Update to the next velocity
    v = w*v + c1*r1*(b-x) + c2*r2*(g-x)

    ;  Clamp velocities to Vmax
    idx = where(v ge self.vmax, count)
    if (count ne 0) then  v[idx] = self.vmax
    idx = where(v le -self.vmax, count)
    if (count ne 0) then  v[idx] = -self.vmax

    (*self.x)[*,i] = x
    (*self.v)[*,i] = v

    ;  Update best position
    if (self.ftype) then begin
      b = call_function(self.func, x)  
    endif else begin
      b = call_method(self.meth, self.obj, x)
    endelse

    if (self.maximize) then begin
      if (b gt (*self.bf)[i]) then begin
        (*self.bf)[i]  = b
        (*self.b)[*,i] = x
      endif
      if ((*self.bf)[i] gt self.gf) then begin
        self.gf = (*self.bf)[i]
        ptr_free, self.g
        self.g = ptr_new(reform((*self.b)[*,i]))
        update = 1b
      endif
    endif else begin
      if (b lt (*self.bf)[i]) then begin
        (*self.bf)[i]  = b
        (*self.b)[*,i] = x
      endif
      if ((*self.bf)[i] lt self.gf) then begin
        self.gf = (*self.bf)[i]
        ptr_free, self.g
        self.g = ptr_new(reform((*self.b)[*,i]))
        update = 1b
      endif
    endelse

  endfor

  ;  Count this iteration
  self.i = self.i + 1
end


;--------------------------------------------------------------
;  ParticleSwarm::RepulsiveUpdate
;+
;  Do one iteration step.
;
;@private
;-
pro ParticleSwarm::RepulsiveUpdate
  compile_opt idl2, HIDDEN
  on_error, 2

  ;  Track if global position updated in this step
  update = 0b

  ;  Look at each particle
  for i=0, self.n-1 do begin

    ;  Update X and V
    x  = reform((*self.x)[*,i])      ;  current position
    v  = reform((*self.v)[*,i])      ;  current velocity
    b  = reform((*self.b)[*,i])      ;  best position for this particle
    g  = reform(*self.g)             ;  global best for all particles
    w  = self.wm*self.i+(self.w)[0]  ;  inertial constant, decreases over time
    c1 = self.c1                     ;  cognitive component weight
    c2 = self.c2                     ;  social component weight
    c3 = self.c3                     ;  random velocity weight
    r1 = randomu(seed, self.m)       ;  random cognitive vector
    r2 = randomu(seed, self.m)       ;  random social vector
    r3 = randomu(seed, self.m)       ;  random vector

    ;  Update to the next position   
    x = x + v

    ;  Constrain to domain, if set
    if (self.const) then begin
      for j=0, n_elements(x)-1 do begin
        d = reform((*self.domain)[*,j])
        if (x[j] lt d[0]) then  x[j] = d[0]
        if (x[j] gt d[1]) then  x[j] = d[1]
      endfor
    endif

    ;  A randomly chosen best position
    repeat begin
      n = fix(self.n * randomu(seed))
    endrep until (n ne i)
    y = reform((*self.b)[*,n])

    ;  A randomly chosen velocity
    repeat begin
      n = fix(self.n * randomu(seed))
    endrep until (n ne i)
    z = reform((*self.v)[*,n])

    ;  Update the velocity
    v = w*v + c1*r1*(b-x) + c2*r2*w*(y-x) + c3*r3*w*z

    ;  Clamp velocities to Vmax
    idx = where(v ge self.vmax, count)
    if (count ne 0) then  v[idx] = self.vmax
    idx = where(v le -self.vmax, count)
    if (count ne 0) then  v[idx] = -self.vmax

    (*self.x)[*,i] = x
    (*self.v)[*,i] = v

    ;  Update best position
    if (self.ftype) then begin
      b = call_function(self.func, x)  
    endif else begin
      b = call_method(self.meth, self.obj, x)
    endelse
    
    if (self.maximize) then begin
      if (b gt (*self.bf)[i]) then begin
        (*self.bf)[i]  = b
        (*self.b)[*,i] = x
      endif
      if ((*self.bf)[i] gt self.gf) then begin
        self.gf = (*self.bf)[i]
        ptr_free, self.g
        self.g = ptr_new(reform((*self.b)[*,i]))
        update = 1b
      endif
    endif else begin
      if (b lt (*self.bf)[i]) then begin
        (*self.bf)[i]  = b
        (*self.b)[*,i] = x
      endif
      if ((*self.bf)[i] lt self.gf) then begin
        self.gf = (*self.bf)[i]
        ptr_free, self.g
        self.g = ptr_new(reform((*self.b)[*,i]))
        update = 1b
      endif
    endelse

  endfor

  ;  Count this iteration
  self.i = self.i + 1
end


;--------------------------------------------------------------
;  ParticleSwarm::SwarmHasConverged
;+
;  Return true if the RMS error between the global position
;  and the top closest 25% is below the tolerance value.
;
;@private
;-
function ParticleSwarm::SwarmHasConverged
  compile_opt idl2, HIDDEN
  on_error, 2

  ;  Find the distance between the current particle positions
  ;  and the global best
  d = ((*self.x) - ((*self.g) # replicate(1.0d, self.n)))^2
  d = sqrt(total(temporary(d),1))

  ;  Get the best 25%
  idx = n_elements(d)/4+1
  t = total((d[sort(d)])[0:idx])/(idx+1)
  return, (t le self.gtol)
end


;
;  Public:
;

;--------------------------------------------------------------
;  ParticleSwarm::GetProperty
;+
;  Get current object properties.
;-
pro ParticleSwarm::GetProperty, W=w, C1=c1, C2=c2,                               $
                                FUNC=func, METHOD=method, OBJECT=obj,            $
                                N_PARTICLES=npart, DOMAIN=domain, IMAX=imax,     $
                                TOLERANCE=gtol, POSITIONS=x, VELOCITIES=v,       $
                                PARTICLE_BEST=b, GLOBAL_BEST=g, DIMENSIONS=m,    $
                                INITIAL_POSITIONS=xi, INITIAL_VELOCITIES=vi,     $
                                ITERATIONS=i, VMAX=vmax,     $
                                CONSTRAIN_TO_DOMAIN=const, C3=c3, REPULSIVE=rep, $
                                MAXIMIZE=mmax
  compile_opt idl2
  on_error, 2

  if arg_present(mmax)   then  mmax   = self.maximize
  if arg_present(rep)    then  rep    = self.repulsive
  if arg_present(c3)     then  c3     = self.c3
  if arg_present(vmax)   then  vmax   = self.vmax
  if arg_present(w)      then  w      = self.w
  if arg_present(c1)     then  c1     = self.c1
  if arg_present(c2)     then  c2     = self.c2
  if arg_present(func)   then  func   = self.func
  if arg_present(method) then  method = self.meth
  if arg_present(obj)    then  obj    = self.obj
  if arg_present(npart)  then  npart  = self.n
  if arg_present(imax)   then  imax   = self.imax
  if arg_present(gtol)   then  gtol   = self.gtol
  if arg_present(m)      then  m      = self.m
  if arg_present(i)      then  i      = self.i
  if arg_present(const)  then  const  = self.const

  if arg_present(domain) then  domain = ptr_valid(self.domain) ? *self.domain : -1
  if arg_present(x)      then  x = ptr_valid(self.x) ? *self.x : -1
  if arg_present(v)      then  v = ptr_valid(self.v) ? *self.v : -1
  if arg_present(b)      then  b = ptr_valid(self.b) ? *self.b : -1
  if arg_present(g)      then  g = ptr_valid(self.g) ? *self.g : -1
  if arg_present(xi)     then  xi = ptr_valid(self.xi) ? *self.xi : -1
  if arg_present(vi)     then  vi = ptr_valid(self.vi) ? *self.vi : -1
end


;--------------------------------------------------------------
;  ParticleSwarm::SetProperty
;+
;  Set current object properties.
;-
pro ParticleSwarm::SetProperty, W=w, C1=c1, C2=c2,                               $
                                FUNC=func, METHOD=method, OBJECT=obj,            $
                                N_PARTICLES=npart, DOMAIN=domain, IMAX=imax,     $
                                TOLERANCE=gtol, INITIAL_POSITIONS=x,             $
                                INITIAL_VELOCITIES=v, DIMENSIONS=m, VMAX=vmax,   $
                                CONSTRAIN_TO_DOMAIN=constrain, C3=c3,            $
                                REPULSIVE=rep, MAXIMIZE=mmax
  compile_opt idl2
  on_error, 2

  if (n_elements(mmax) ne 0) then  self.maximize = (mmax ne 0)
  if (n_elements(c3) ne 0) then  self.c3 = c3
  if (n_elements(rep) ne 0) then  self.repulsive = (rep ne 0)

  if (n_elements(constrain) ne 0) then  $
    self.const = (constrain ne 0)

  if (n_elements(w) ne 0) then begin
    if (n_elements(w) eq 1) then begin
      self.w = [w,w]
      self.wm = 0
    endif else begin
      self.w = (w[0] gt w[1]) ? [w[0],w[1]] : [w[1],w[0]]
      self.wm = ((self.w)[1] - (self.w)[0]) / double(self.imax)
    endelse
  endif

  if (n_elements(m) ne 0)     then  self.m     = m
  if (n_elements(c1) ne 0)    then  self.c1    = c1
  if (n_elements(c2) ne 0)    then  self.c2    = c2
  if (n_elements(npart) ne 0) then  self.n     = npart

  if (n_elements(imax) ne 0)  then begin
    self.imax  = imax
    self.wm = ((self.w)[1] - (self.w)[0]) / double(self.imax)
  endif

  if (n_elements(vmax) ne 0)  then  self.vmax  = abs(vmax)
  if (n_elements(gtol) ne 0)  then  self.gtol  = gtol

  if (n_elements(func) ne 0) then begin
    if (size(func,/TYPE) ne 7) then  $
      message, 'A string is required.'
    if (func eq '') then begin
      self.ftype = 0 
      self.func = ''
    endif else begin
      self.ftype = 1
      self.func = func
    endelse
  endif

  if (n_elements(method) ne 0) then begin
    if (size(method,/TYPE) ne 7) then  $
      message, 'A string is required.'
    if (method eq '') then begin
      self.ftype = 1 
      self.meth = ''
    endif else begin
      self.ftype = 0
      self.meth = method
    endelse    
  endif

  if (n_elements(obj) ne 0) then begin
    if (size(obj,/TYPE) ne 11) then  $
      message, 'An object reference is required.'
    obj_destroy, self.obj
    self.obj = obj
    self.ftype = 0
  endif

  if (n_elements(domain) ne 0) then begin
    if (n_elements(domain) eq 2) then begin
      if (self.m eq 0) then  $
        message, 'DIMENSIONS must be set before setting the DOMAIN.'
      d = dblarr(2,self.m)
      for i=0, self.m-1 do begin
        d[*,i] = domain
      endfor
      ptr_free, self.domain
      self.domain = ptr_new(d)
    endif else begin
      msg = 'A 2 x M matrix of lower/upper limits per dimension is required.'
      if size(domain,/N_DIM) ne 2 then  message, msg
      dim = size(domain,/DIM)
      if dim[0] ne 2 then  message, msg
      if dim[1] ne self.m then  message, msg
      ptr_free, self.domain
      self.domain = ptr_new(domain)
    endelse
  endif

  if (n_elements(x) ne 0) then begin
    msg = 'An M x N matrix of initial particle positions is required.'
    if size(x,/N_DIM) ne 2 then  message,msg
    dim = size(x,/DIM)
    if dim[0] ne self.m then  message,msg
    if dim[1] ne self.n then  message,msg
    ptr_free, self.xi
    self.xi = ptr_new(x)
  endif

  if (n_elements(v) ne 0) then begin
    msg = 'An M x N matrix of initial particle velocities is required.'
    if size(v,/N_DIM) ne 2 then  message,msg
    dim = size(v,/DIM)
    if dim[0] ne self.m then  message,msg
    if dim[1] ne self.n then  message,msg
    ptr_free, self.vi
    self.vi = ptr_new(v)
  endif
end


;--------------------------------------------------------------
;  ParticleSwarm::Initialize
;+
;  Initialize x and v.
;-
pro ParticleSwarm::Initialize
  compile_opt idl2

  ;  Reset iteration count & g count
  self.i = 0L

  ;
  ;  If the user supplied initial positions, use them
  ;  otherwise, randomly choose vectors in the given
  ;  domain, or, [0,1) per dimension if the domain is not
  ;  specified.
  ;
  if ptr_valid(self.xi) then begin
    ;
    ;  User-supplied initial positions
    ;
    ptr_free, self.x
    self.x = ptr_new(*self.xi)
  endif else begin
    ;
    ;  Randomly selected
    ;
    if ~ptr_valid(self.domain) then begin
      ;
      ;  No domain specified
      ;
      xi = randomu(seed, self.m, self.n)
    endif else begin
      ;
      ;  Use the given domain
      ;
      xi = dblarr(self.m, self.n)
      for i=0, self.n-1 do begin
        xi[*,i] = self->RandomVectorInDomain()
      endfor
    endelse
    ptr_free, self.x
    self.x = ptr_new(xi)
  endelse

  ;
  ;  Set the initial velocities.
  ;
  if ptr_valid(self.vi) then begin
    ptr_free, self.v
    self.v = ptr_new(*self.vi)
  endif else begin
    ;  Set all to zero.
    ptr_free, self.v
    self.v = ptr_new(dblarr(self.m, self.n))
  endelse

  ;
  ;  Set up the initial particle best positions and function
  ;  values.
  ;
  ptr_free, self.b
  ptr_free, self.bf
  self.b = ptr_new(*self.x)
  self.bf = ptr_new(dblarr(self.n))
  for i=0, self.n-1 do begin
    if (self.ftype) then begin
      (*self.bf)[i] = call_function(self.func, (*self.x)[*,i])  
    endif else begin
      (*self.bf)[i] = call_method(self.meth, self.obj, (*self.x)[*,i])
    endelse
  endfor

  ;
  ;  Set up the initial global best
  ;
  ptr_free, self.g

  g = (*self.b)[*,0]
  gf = (*self.bf)[0]

  for i=1, self.n-1 do begin
    if (*self.bf)[i] lt gf then begin
      g = (*self.b)[*,i]
      gf = (*self.bf)[i]
    endif
  endfor

  self.g = ptr_new(g)
  self.gf = gf
end


;--------------------------------------------------------------
;  ParticleSwarm::Done
;+
;  Return true if convergence or maximum number if iterations.
;-
function ParticleSwarm::Done
  compile_opt idl2

  ;  Did we run out of iterations?
  if (self.i eq self.imax) then  $
    return, 1b

  ;  Are we within the tolerance?
  return, self->SwarmHasConverged()
end


;--------------------------------------------------------------
;  ParticleSwarm::Step
;+
;  Do one iteration.
;-
pro ParticleSwarm::Step
  compile_opt idl2
  on_error, 2
  
  if (self.repulsive) then begin
    self->RepulsiveUpdate
  endif else begin
    self->GlobalUpdate
  endelse
end


;--------------------------------------------------------------
;  ParticleSwarm::Optimize
;+
;  Do the optimization using the current properties.
;-
function ParticleSwarm::Optimize, ERROR=error
  compile_opt idl2
  on_error, 2

  error = ''

  ;  Initialize x, v, b and g
  self->Initialize

  ;  Iterate and check if done
  if self.repulsive then begin
    repeat begin
      self->RepulsiveUpdate
    endrep until self->Done()
  endif else begin
    repeat begin
      self->GlobalUpdate
    endrep until self->Done()
  endelse

  ;  Didn't converge
  if (self.i ge self.imax) then begin
    error = 'Failed to meet tolerance after '+strtrim(self.imax,2)+' iterations.'
  endif

  ;  Return best position so far
  return, *self.g
end


;
;  Constructor/destructor:
;

;--------------------------------------------------------------
;  ParticleSwarm::Destroy
;+
;  Actual destructor.
;-
pro ParticleSwarm::Destroy
  compile_opt idl2
  on_error, 2

  ptr_free, [self.x, self.v, self.b, self.bf, self.g,     $
             self.domain, self.xi, self.vi]
  if obj_valid(self.obj) then  $
    obj_destroy, self.obj
end


;--------------------------------------------------------------
;  ParticleSwarm::Cleanup
;+
;  Class destructor.
;-
pro ParticleSwarm::Cleanup
  compile_opt idl2
  on_error, 2

  self->Destroy
end


;--------------------------------------------------------------
;  ParticleSwarm::Init
;+
;  Class constructor.
;-
function ParticleSwarm::Init, _EXTRA=_extra
  compile_opt idl2
  on_error, 2

  ;  Set defaults
  self.w  = [0.9,0.9]  ;  slightly less than 1
  self.wm = 0
  self.c1 = 2          ;  "standard" values
  self.c2 = 2   
  self.c3 = 2       
  self.imax = 25       ;  maximum iterations
  self.gtol = 1e-4     ;  global tolerance 
  self.n    = 10       ;  number of particles
  self.vmax = 1e8      ;  Vmax, per dimension
  self.const= 0b       ;  do not constrain motion
  self.repulsive = 0b  ;  global by default
  self.maximize = 0b   ;  minimize by default

  ;  Process any given properties
  self->SetProperty, _EXTRA=_extra

  return, 1
end


;--------------------------------------------------------------
;  ParticleSwarm__define
;+
;  Class definition.
;-
pro ParticleSwarm__define
  compile_opt idl2
  on_error, 2
 
  class = {ParticleSwarm,        $
            m      : 0L,         $  ;  dimensionality of the problem
            w      : dblarr(2),  $  ;  inertial constant - decreases with each step, [start,end]
            wm     : 0.0d,       $  ;  slope of w line
            c1     : 0.0d,       $  ;  cognitive constant
            c2     : 0.0d,       $  ;  social constant
            c3     : 0.0d,       $  ;  repulsive constant
            n      : 0L,         $  ;  number of particles
            domain : ptr_new(),  $  ;  domain, low, high, (2 x m matrix)
            const  : 0b,         $  ;  if set, constrain search to the given domain
            repulsive : 0b,      $  ;  if set, use repulsive optimization, otherwise use global
            maximize : 0b,       $  ;  if set, maximize not minimize
            xi     : ptr_new(),  $  ;  particle initial positions (m x n matrix)
            x      : ptr_new(),  $  ;  particle positions (m x n matrix)
            vi     : ptr_new(),  $  ;  particle initial velocities (m x n matrix)
            v      : ptr_new(),  $  ;  particle velocities (m x n matrix)
            b      : ptr_new(),  $  ;  particle personal best positions (m x n matrix)
            bf     : ptr_new(),  $  ;  particle personal best function values (n-element vector)
            g      : ptr_new(),  $  ;  global best position (m-element vector)
            gf     : 0.0d,       $  ;  global best function value (scalar)
            vmax   : 0.0d,       $  ;  maximum velocity, per component
            imax   : 0L,         $  ;  maximum number of iterations
            i      : 0L,         $  ;  number of iterations performed
            gtol   : 0.0d,       $  ;  global best tolerance
            ftype  : 0b,         $  ;  1=function, 0=method call
            func   : '',         $  ;  name of IDL objective function
            meth   : '',         $  ;  method name of IDL objective function
            obj    : obj_new()   $  ;  an IDL object reference which supports the method
          }
end


;
;  end particleswarm__define.pro
;
