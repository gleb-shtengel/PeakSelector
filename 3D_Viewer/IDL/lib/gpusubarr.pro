; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpusubarr.pro
;
; Extracts a sub-array from a GPU array.
;
; Copyright (C) 2008 Tech-X Corporation. All rights reserved.
;
; This file is part of GPULib.
;
; This file may be distributed under the terms of the GNU Affero General Public
; License (AGPL). This file may be distributed and/or modified under the
; terms of the GNU Affero General Public License version 3 as published by the
; Free Software Foundation.
;
; This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
; WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
;
; Licensees holding valid Tech-X commercial licenses may use this file
; in accordance with the Tech-X Commercial License Agreement provided
; with the Software.
;
; See http://gpulib.txcorp.com/ or email sales@txcorp.com for more information.
;
; This work was in part funded by NASA SBIR Phase II Grant #NNG06CA13C.
;
;-----------------------------------------------------------------------------

;+
;
; Extract a subset of an array and store it in a subset of the result array.
;
; :Params:
;    p1 : in, required, type={ GPUHANDLE }
;       src_gpu: GPU array to extract subarray from.
;    p2 : in, required, type=int or intarr
;       src_x: index[array] of src_gpu x dimension. When src_x is an int, this 
;       is the x-index to be extracted. When src_x is a 2-element int-arr, this 
;       is array is of the form [lower, upper] and lower and upper index of the 
;       x-subrange. src_x, lower or upper can be -1, specifying a the full range 
;       (equivalent to IDL's '*'.
;    p3 : in, type=int or intarr
;       src_y: index[array] of src_gpu y dimension. For more details see src_x;
;       required if x_gpu is 2D
;    p4 : in, required, type = { GPUHANDLE }
;       dest_gpu: GPU array to store subarray into
;    p5 : in, required, type=int or intarr
;       dest_x: index[array] of dest_gpu x dimension. For more details see 
;       src_x.
;    p6 : in, type=int or intarr
;       dest_y: index[array] of dest_gpu y dimension. For more details see
;       src_x; required if dest is 2D
;
;-
pro gpuSubArr, p1, p2, p3, p4, p5, p6
 on_error, 2

 src = p1

 ; clean up input arguments
 if src.n_dimensions eq 1 then begin ; 2nd parameter is range
     sxrange = long(p2)
     syrange = 0L
     dest = p3
     dxrange = long(p4)
     dyrange = (dest.n_dimensions eq 2) ? long(p5) : 0L
 end else begin                 ; we have a 2d object
     sxrange = long(p2)
     syrange = long(p3)
     dest = p4
     dxrange = long(p5)
     dyrange = (dest.n_dimensions eq 2) ? long(p6) : 0L
 end


;-- source range

; compute offset
 sxoff = sxrange[0]
 if sxrange[0] eq -1 then sxoff = 0L

 syoff = syrange[0]
 if syrange[0] eq -1 then syoff = 0L

 soff = sxoff  + syoff * src.dimensions[0]

; compute width
 if n_elements(sxrange) eq 1 then begin
     swidth = (sxrange[0] eq -1) ? src.dimensions[0] : 1L
 end else begin
     top = (sxrange[1] eq -1) ? src.dimensions[0] : sxrange[1]+1
     swidth = (top - sxrange[0])
 end

; compute height
 if n_elements(syrange) eq 1 then begin
     sheight = (syrange[0] eq -1) ? src.dimensions[1] : 1L
 end else begin
     top = (syrange[1] eq -1)? src.dimensions[1] : syrange[1]+1
     sheight = (top - syrange[0])
 end


;-- destination range

; compute offset
 dxoff = dxrange[0]
 if dxrange[0] eq -1 then dxoff = 0L

 dyoff = dyrange[0]
 if dyrange[0] eq -1 then dyoff = 0L

 doff = dxoff + dyoff * dest.dimensions[0]

; compute width
 if n_elements(dxrange) eq 1 then begin
     dwidth = (dxrange[0] eq -1) ? dest.dimensions[0] : 1L
 end else begin
     top = (dxrange[1] eq -1) ? dest.dimensions[0] : dxrange[1]+1
     dwidth = (top - dxrange[0])
 end

; compute height
 if n_elements(dyrange) eq 1 then begin
     dheight = (dyrange[0] eq -1) ? dest.dimensions[1] : 1L
 end else begin
     top = (dyrange[1] eq -1) ? dest.dimensions[1] : dyrange[1]+1L
     dheight = (top - dyrange[0])
 end

; different cases for copying:
; 1D src, 1D dest
; 1D src, 2D dest
; 2D src, 1D dest
; 2D src, 2D dest


; 1D src, 1D dest
 if (sheight eq 1) and (dheight eq 1) then begin
     if swidth ne dwidth then begin
         message, level=-1, 'gpuSubArr: invalid vector sizes'
     end

	 if (!gpu.mode eq 0) then begin
          (*dest.data)[doff:doff+swidth-1] = (*src.data)[soff:soff+swidth-1]
     end else begin
          err = cudaMemcpy(dest.handle+4L*doff, src.handle+4L*soff, swidth*4L, 3L)
     end
     return
 end

; 1D src, 2D dest
 if (sheight eq 1) and (dwidth eq 1) then begin
     if swidth ne dheight then begin
         message, level=-1, 'gpuSubArr: invalid vector sizes'
     end

	 if (!gpu.mode eq 0) then begin
          (*dest.data)[dxoff, dyoff:dyoff+dheight-1] = (*src.data)[soff:soff+dheight-1]
     end else begin
          err = cudaMemcpy2D(dest.handle+4L*doff, dest.dimensions[0]*4L, $
                        src.handle+4L*soff, 4L, 4L, dheight, 3L)
     end
     return
 end

; 2D src, 1D dest
 if (swidth eq 1) and (dheight eq 1) then begin
     if sheight ne dwidth then begin
         message, level=-1, 'gpuSubArr: invalid vector size'
     end
     if (!gpu.mode eq 0) then begin
          (*dest.data)[doff:doff+sheight-1] = (*src.data)[sxoff, syoff:syoff+sheight-1]
     end else begin
          err = cudaMemcpy2D(dest.handle+4L*doff, 4L, $
                       src.handle+4L*soff, src.dimensions[0]*4L, $
                       4L, sheight, 3L)
     end
     return
 end

; 2D src, 2D dest
 if swidth * sheight ne dwidth * dheight then begin
     message, level=1, 'gpuSubArr: invalid number of elements'
 end

; no gap means no break in x direction. If height eq 1 then anything,
; if height ne 1, then width has to be dimension[0]*4L

 snogap = (sheight eq 1) or (swidth eq src.dimensions[0])
 dnogap = (dheight eq 1) or (dwidth eq dest.dimensions[0])
 
; both source and destionation are contiguous blocks
 if snogap and dnogap then begin

   if (!gpu.mode eq 0) then begin
       (*dest.data)[doff:doff+swidth*sheight-1] = $
                  (*src.data)[soff:soff+swidth*sheight-1]
   end else begin
        err = cudaMemcpy(dest.handle+4L*doff, src.handle+4L*soff, $
                  swidth*sheight*4L, 3L)
   end
   return
 end

; src is contiguous block, dest has gaps
 if snogap and ~dnogap then begin
   if (!gpu.mode eq 0) then begin
       (*dest.data)[dxoff:dxoff+dwidth-1, dyoff:dyoff+dheight-1] =    $
             (*src.data)[soff:soff+dwidth*dheight-1]
   end else begin
       err = cudaMemcpy2D(dest.handle+4L*doff, dest.dimensions[0]*4L, $
                     src.handle+4L*soff, 4L*dwidth,                $
                     4L*dwidth, dheight, 3L)
   end
   return
 end

 ; src has gap, dest is contiguous
 if ~snogap and dnogap then begin
  if (!gpu.mode eq 0) then begin
       (*dest.data)[doff:doff+swidth*sheight-1] = $
                (*src.data)[sxoff:sxoff+swidth-1, syoff:syoff+sheight-1]
  end else begin
       err = cudaMemcpy2D(dest.handle+4L*doff, 4L*swidth, $
                     src.handle+4L*soff, src.dimensions[0] * 4L ,$
                     4L*swidth, sheight, 3L)
  end
  return
 end

 if (!gpu.mode eq 0) then begin
       (*dest.data)[dxoff+dxoff+dwidth-1, dyoff:dyoff+dheight-1] = $
               (*src.data)[sxoff:sxoff+swidth-1, syoff:syoff+sheight-1]
 end else begin
    err = cudaMemcpy2D(dest.handle+4L*doff, dest.dimensions[0]*4L, $
                    src.handle+4L*soff, src.dimensions[0]*4L, $
                    4L*swidth, sheight, 3L)
 end
end


