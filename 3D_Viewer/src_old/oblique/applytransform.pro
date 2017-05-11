function ApplyTransform, tx, data
  compile_opt idl2, logical_predicate
  
  dims = [size(data, /DIMENSIONS),1]
  if (dims[0] lt 4) then begin
    zero = (make_array([1], TYPE=size(data, /TYPE)))[0]
    one = zero + 1b
    in_data = replicate(one, [4, dims[1]])
    if (dims[0] lt 3) then in_data[2,*] = zero
    in_data[0,0] = arg_present(data) ? data : temporary(data)
  endif else in_data = arg_present(data) ? data : temporary(data)
  
  res = matrix_multiply(tx, in_data, /ATRANSPOSE)
  ; @TODO should reduce to 3xN first
  return, res
end
    
    