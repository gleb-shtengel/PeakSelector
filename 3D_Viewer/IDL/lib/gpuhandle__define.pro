; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuhandel__define.pro
;
; Definition of a GPU data type in IDL
;
; Copyright (C) 2008 Tech-X Corporation. All rights reserved.
;
; This file is part of GPULib.
;
; This file may be distributed under the terms of the GNU Affero General Public
; License (AGPL). This file may be distributed and/or modified under the
; terms of the GNU Affero  General Public License version 3 as published by the
; Free Software Foundation.
;
; This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
; WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
;
; Licensees holding valid Tech-X commercial licenses may use this file
; in accordance with the Tech-X Commercial License Agreement provided
; with the Software.
;
; See http://gpulib.txcorp.com/ or email sales@txcorp.com for more 
; information.
; 
; This work was in part funded by NASA SBIR Phase II Grant #NNG06CA13C.
;
;-----------------------------------------------------------------------------

;+
; Define GPU variable's fields.
;
; :Fields:
;    type
;       IDL type code for variable
;    isTemporary
;       true if the GPU variable is a temporary variable i.e. that it was 
;       created as a return value
;    n_elements
;       total number of elements in the variable
;    n_dimensions
;       number of dimensions: 0, 1, 2
;    dimensions
;       size of each dimension
;    handle
;       used in hardware or emulation mode
;    data
;       used in IDL mode
;-
pro gpuhandle__define
  
  define = { GPUHANDLE, $
             type         : 0L, $
             isTemporary  : 0B, $
             n_elements   : 0L, $
             n_dimensions : 0L, $
             dimensions   : [0L, 0L], $
             handle       : 0LL, $ ; used in hardware or emulator mode
             ;handle       : 0L, $ ; used in hardware or emulator mode
             data         : ptr_new() $  ; used in IDL emulation mode
           }
end
