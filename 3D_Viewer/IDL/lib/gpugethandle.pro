; docformat = 'rst'
;
;-----------------------------------------------------------------------------
;
; gpuGetHandle.pro
;
; creates an empty gpuhandle struct
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
; See http://gpulib.txcorp.com/ or email sales@txcorp.com for more information.
;
; This work was in part funded by NASA SBIR Phase II Grant #NNG06CA13C.
;
;-----------------------------------------------------------------------------

;+
; Creates a variable on the GPU that can be manipulated by the FastGPU library
; routines.
;
; :Returns: 
;    a blank { GPUHANDLE } structure
;-
function gpuGetHandle
  return, { GPUHANDLE }
end


