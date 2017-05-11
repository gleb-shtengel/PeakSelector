; docformat = 'rst'

;+
; Individuals are elements of the problem that encode a solution to the problem,
; calculate how good the solution is (i.e. the fitness), mutate themselves, and
; create a new solution from themselves and another individual.
; 
; This class is intended to be subclassed, not instantiated.
;
; :Properties:
;    name
;       optional name of the individual
;    fitness
;       fitness of the individual
;-


;+
; Calculates the fitness function.
;
; :Abstract:
; :Returns: float
;-
function ga_individual::fitness
  compile_opt strictarr
  
  return, 0.0
end


;+
; Mutate this individual's solution slightly.
;
; :Abstract:
;-
pro ga_individual::mutate
  compile_opt strictarr
  
end


;+
; Create an offspring between this individual and the given individual.
;
; :Abstract:
; :Returns: 
;    individual object representing the offspring between this individual and
;    the specified individual
;-
function ga_individual::reproduce, individual
  compile_opt strictarr
  
  return, obj_new()
end


pro ga_individual::getProperty
  compile_opt strictarr
  
end


pro ga_individual::setProperty
  compile_opt strictarr
  
end


;+
; Free resources.
;-
pro ga_individual::cleanup
  compile_opt strictarr
  
end


;+
; Create a randomly generated individual.
;
; :Returns: 1 for success, 0 for failure
;-
function ga_individual::init
  compile_opt strictarr
  
  return, 1
end


;+
; Define instance variables.
;
; :Fields:
;    name
;       optional name to refer to the individual by
;    fitness
;       result of fitness function
;-
pro ga_individual__define
  compile_opt strictarr
  
  define = { ga_individual, $
             name: '', $
             fitness: 0.0 $
           }
end
