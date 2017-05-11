;+
; Get nIndices random indices for an array of size nValues (do not repeat an 
; index).
;
; @returns lonarr(nIndices)
;
; @param nValues {in}{required}{type=long}
;        size of array to choose indices from
;
; @param nIndices {in}{required}{type=long}
;        number of indices needed
;
; @keyword seed {in}{out}{optional}{type=integer or lonarr(36)}
;          seed to use for random number generation, new seed will be output
;-
function mg_sample, nValues, nIndices, seed=seed
  compile_opt strictarr
  
  ; get random nIndices by finding the indices of the smallest nIndices in a 
  ; array of random values
  values = randomu(seed, nValues)
  
  ; our random values are uniformly distributed, so ideally the nIndices 
  ; smallest values are in the first bin of the below histogram
  nBins = nValues / nIndices
  h = histogram(values, nbins=nBins, reverse_indices=ri)

  ; the candidates for being in the first nIndices will live in bins 0..bin
  nCandidates = 0L
  for bin = 0L, nBins - 1L do begin
    nCandidates += h[bin]
    if (nCandidates ge nIndices) then break    
  endfor

  ; get the candidates and sort them
  candidates = ri[ri[0] : ri[bin + 1L] - 1L]
  sortedCandidates = sort(values[candidates])

  ; return the first nIndices of them
  return, (candidates[sortedCandidates])[0:nIndices-1L]
end
