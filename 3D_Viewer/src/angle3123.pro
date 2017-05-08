; -------------------------------------------------------------
;+
;  @returns The 3 angles of a space three 1-2-3 given a 3 x 3 
;           cosine direction matrix else -1 on failure.
;
;  Definition :  Given 2 sets of dextral orthogonal unit vectors
;                (a1, a2, a3) and (b1, b2, b3), the cosine direction matrix
;                C (3 x 3) is defined as the dot product of:
;
;                C(i,j) = ai . bi  where i = 1,2,3
;
;                A column vector X (3 x 1) becomes X' (3 x 1)
;                after the rotation as defined as :
;
;                X' = C X
;
;                The space three 1-2-3 means that the x rotation is first,
;                followed by the y rotation, then the z.
;
;  @param cosMat {in}{required}{type=3x3 matrix}
;    Cosine direction matrix.
;-
;--------------------------------------------------------------
function angle3123, $
    cosMat           ; IN: cosine direction matrix (3 x 3)
On_Error, 2
    ;  Verify the input parameters
    ;
    if (N_PARAMS() ne 1) then begin
        Message,'Error in angle3123: 1 parameters must be passed.', /Continue
        RETURN, -1
    endif
    sizec = size(cosMat, /Dimensions)
    if (N_elements(sizec) ne 2) then begin
        Message,'Error, the input matrix must be of dimension 2', /Continue
        RETURN, -1
    endif
    if ((sizec[0] ne 3) or (sizec[1] ne 3)) then begin
        PRINT,'Error, the input matrix must be 3 by 3'
        RETURN, -1
    endif

    ;  Compute the 3 angles (in degrees)
    ;
    cosMat = TRANSPOSE(cosMat)
    angle = DBLARR(3)
    angle[1] = -cosMat[2,0]
    angle[1] = ASIN(angle[1])
    c2 = COS(angle[1])
;    if (ABS(angle[1]) lt 1.0e-6) then begin
    if (c2 lt 1.0e-6) then begin
        angle[0] = ATAN(-cosMat[1,2], cosMat[1,1])
        angle[2] = 0.0
    endif else begin
        angle[0] = ATAN( cosMat[2,1], cosMat[2,2])
        angle[2] = ATAN( cosMat[1,0], cosMat[0,0])
    endelse
    angle = angle * (180.0/!DPI)

    RETURN, angle

end    ;   of angle3123
