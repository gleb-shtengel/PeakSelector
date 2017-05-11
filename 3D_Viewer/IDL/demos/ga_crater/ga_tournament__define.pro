; docformat = 'rst'

;+
; Represents a genetic algorithm competition.
;
; :Examples:
;    To run a tournament::
;    
;       IDL> tournament = obj_new('GA_Tournament', classname='my_solution', $
;                                 n_rounds=1000, population_size=1000)
;       IDL> winner = tournament->run()
;       IDL> winner->getProperty, fitness=fitness
;       IDL> print, fitness
;       IDL> obj_destroy, tournament
;
; :Properties:
;    classname
;       the classname of the individuals in the tournament
;    n_rounds
;       the number of rounds in the tournament
;    population_size
;       the number of individuals alive at a given time
;-


;+
; Run the tournament and return the winner object(s).
;
; :Returns: object reference or array of object references
;
; :Keywords:
;    _extra : in, optional, type=keywords
;       keywords to individual's init method
;-
function ga_tournament::run, time=time, _extra=e
  compile_opt strictarr
  
  t0 = systime(/seconds)
  
  ; get an initial population of individuals
  print, 'Populating tournament...'
  self->_populate, _strict_extra=e
  
  for r = 0L, self.nRounds - 1L do begin
    self->_calculateFitnesses
    
    maxF = max(*self.fitnesses, min=minF)
    print, format='(%"Round: %d; fitnesses: %0.2f-%0.2f")', r, minF, maxF
    
    rank = sort(*self.fitnesses)
    
    self->_cull, rank[0:self.toCull - 1L]
    self->_reproduce, rank[0:self.toCull - 1L], $
                      rank[self.populationSize - self.toReproduce:self.populationSize - 1L]
    self->_mutate, rank[self.toCull:self.populationSize - self.toReproduce - 1L]
  endfor
  
  self->_calculateFitnesses
  maxFitness = max(*self.fitnesses, maxIndex)
  
  time = systime(/seconds) - t0
  
  return, (*self.population)[maxIndex]
end


;+
; Create an initial random population of individuals. 
;
; :Keywords:
;    _extra : in, optional, type=keywords
;       keywords to individual's init method
;-
pro ga_tournament::_populate, _extra=e
  compile_opt strictarr
  
  self.population = ptr_new(objarr(self.populationSize))
  
  for i = 0L, self.populationSize - 1L do begin
    (*self.population)[i] = obj_new(self.classname, _extra=e)
  endfor
end


;+
; Calculate the fitness (and store it) for each individual in the population.
;-
pro ga_tournament::_calculateFitnesses
  compile_opt strictarr

  for i = 0L, self.populationSize - 1L do begin
    (*self.fitnesses)[i] = (*self.population)[i]->fitness()
  endfor
end


;+
; Eliminate the individuals with lower fitnesses. Return the indices of the 
; culled individuals.
;
; :Returns: indices of culled individuals
;-
pro ga_tournament::_cull, indToCull
  compile_opt strictarr
  
  obj_destroy, (*self.population)[indToCull]
end


;+
; Mutate the individuals that aren't parents or children.
;
; :Params:
;-
pro ga_tournament::_mutate, indToMutate
  compile_opt strictarr
  
  ; mutate each individual in mutated set
  for m = 0L, self.toMutate - 1L do begin
    (*self.population)[indToMutate[m]]->mutate
  endfor
end


;+
; Produce a new generation of solutions.
; 
; :Returns: indices of all parent and children individuals
; :Params:
;-
pro ga_tournament::_reproduce, indForChildren, indForParents
  compile_opt strictarr
  
  ; for each new child
  for c = 0L, self.toCull - 1L do begin
    ; pick out two parents
    momAndDadInd = mg_sample(self.toReproduce, 2)
    
    ; call child = mom->reproduce(dad)
    momAndDad = (*self.population)[indForParents[momAndDadInd]]
    (*self.population)[indForChildren[c]] = momAndDad[0]->reproduce(momAndDad[1])
  endfor
end


;+
; Get properties of the tournament.
;-
pro ga_tournament::getProperty, classname=classname, n_rounds=nRounds, $
                                population_size=populationSize
  compile_opt strictarr
  
  if (arg_present(classname)) then classname = self.classname                                
  if (arg_present(nRounds)) then nRounds = self.nRounds                                
  if (arg_present(populationSize)) then populationSize = self.populationSize                                
end


pro ga_tournament::setProperty
  compile_opt strictarr
  
end


;+
; Free tournament resources (including individuals).
;-
pro ga_tournament::cleanup
  compile_opt strictarr
  
  obj_destroy, *self.population
  ptr_free, self.population, self.fitnesses
end


;+
; Create a tournament.
;
; :Returns: 1 for success, 0 for failure
;-
function ga_tournament::init, classname=classname, n_rounds=nRounds, $
                              population_size=populationSize, $
                              mutate_fraction=mutateFraction, $
                              cull_fraction=cullFraction
  compile_opt strictarr
  on_error, 2
  
  if (n_elements(classname) eq 0) then begin
    message, 'classname for individuals is required'
  endif
  
  self.classname = classname
  
  self.nRounds = n_elements(nRounds) eq 0 ? 100 : nRounds  
  self.populationSize = populationSize
  self.fitnesses = ptr_new(fltarr(self.populationSize))
  
  self.mutateFraction = n_elements(mutateFraction) eq 0 ? 0.1 : mutateFraction
  self.cullFraction = n_elements(cullFraction) eq 0 ? 0.5 : cullFraction
  
  self.toCull = self.populationSize * self.cullFraction > 1L
  self.toMutate = self.populationSize * self.mutateFraction > 1L
  self.toReproduce = self.populationSize - self.toCull - self.toMutate
  
  return, 1
end


;+
; Define the instance variables.
;
; :Fields:
;    classname
;       the classname of the individuals in the tournament
;    nRounds
;       the number of rounds in the tournament
;    populationSize
;       the number of individuals alive at a given time
;    population
;       pointer to array of individual objects
;-
pro ga_tournament__define
  compile_opt strictarr
  
  define = { ga_tournament, $
             classname: '', $  
             nRounds: 0L, $
             populationSize: 0L, $
             population: ptr_new(), $
             fitnesses: ptr_new(), $
             mutateFraction: 0.0, $
             cullFraction: 0.0, $
             toCull: 0L, $
             toMutate: 0L, $
             toReproduce: 0L $
           }
end
