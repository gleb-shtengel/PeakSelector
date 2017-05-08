;
;  file:  psfit.pro
;
;  IDL wrapper for GPU particle swarm fit.
;
;  RTK, 08-Oct-2008
;  Last update:  08-Oct-2008
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;--------------------------------------------------------------
;  psfit
;+
;  Fit a function using a particle swarm
;-
function psfit, x0, y0, sigma, N_PARTICLES=npart0, IMAX=imax0, TOLERANCE=tol0, $
                STATISTICAL=stats, N_PARAMS=nparams
  compile_opt idl2

  ;  Must have the number of parameters to fit
  if (n_elements(nparams) eq 0) then  $
    message, 'The number of parameters is required.'
  nparams = nparams[0]

  ;  Validate X and Y, convert to float
  x = float(x0)
  y = float(y0)
  if (n_elements(x) ne n_elements(y)) then  $
    message, 'X and Y must be the same length'

  ;  Calculate the maximum number of samples that can be in the 
  ;  x, y and w vectors and still fit in the constant memory of
  ;  the device (<64k)...
  if (n_elements(x) gt (5459 - round(nparams/3.0))) then  $
    message, 'Too many samples in the arrays to fit into device memory.'

  ;  Set parameters
  npart = (n_elements(npart0) eq 0) ? 100000L : npart0
  imax  = (n_elements(imax0) eq 0) ? 100 : imax0
  tol   = (n_elements(tol0) eq 0) ? 1e-4 : tol0

  ;  Calculate weights from sigmas
  if (n_elements(sigma) eq 0) then begin
    ;  No sigma = no weighting
    w = replicate(1.0, n_elements(y))
  endif else begin
    if keyword_set(stats) then begin
      ;  Statistical weighting
      w = float(1.0/y)
    endif else begin
      ;  User supplied weights
      if (n_elements(sigma) ne n_elements(y)) then  $
        message, 'Sigma and y must be of the same length'
      w = float(1.0 / sigma)
    endelse
  endelse

  ;  Call the DLM
  params = particle_swarm_fit(x, y, w, nparams, npart, imax, tol)

  ;  Return the fit parameters
  return, params
end

