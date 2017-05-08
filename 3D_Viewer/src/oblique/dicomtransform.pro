;+
; This function computes a transformation matrix based on DICOM
; position, DICOM orientation, pixel spacing, image dimensions, and
; adjacent slice spacing.
;
; @Returns
;  Returns a 4x4 transformation matrix between patient space and voxel
;  space.
;
; @Param dicomPosition {in}{required}{type=double}
;  Set this to a 3 element array containing locations of the first
;  pixel of the last row of the image (default) or the first pixel of
;  the first row (/ORDER).
;
; @Param dicomOrientation {in}{required}{type=double}
;  Set this to a 6 element array where the first 3 represent a
;  direction vector along the pixel rows, and the last 3 represent a
;  direction vector across the columns. When the direction vectors are
;  placed at the dicomPosition, they point towards the other pixels on
;  the same row and column respectively.
;
; @Param pixelSpacing {in}{required}{type=double}
;  Set this to a 2 element array specifying the width and height of a
;  pixel.
;
; @Param imageDimensions {in}{required}{type=long}
;  Set this to the image dimensions in number of pixels.
;
; @Param zSpacing {in}{optional}{type=double}
;  Set this to the spacing between slices. Positive spacing is along
;  the positive normal vector given by the cross product of the two
;  vectors given by the "dicomOrientation" parameter. If /ORDER is
;  specified, then the two vectors are used directly, otherwise the
;  the second vector is first multiplied by -1. A negative zSpacing
;  value goes in the opposite direction. The default value is +1.
;
; @History
;  Jul-2006, AB, ITT - Original <br>
;-
function DicomTransform, dicomPosition, dicomOrientation, $
                            pixelSpacing, zSpacing
    compile_opt idl2, logical_predicate

    if N_ELEMENTS(zSpacing) eq 0 then zSpacing = 1.0

    transform = DIAG_MATRIX(REPLICATE(1d,4))

    xVec = dicomOrientation[0:2]
    yVec = dicomOrientation[3:5]
    zVec = CROSSP(xVec, yVec)

    xVec /= pixelSpacing[0]
    yVec /= pixelSpacing[1]
    zVec /= zSpacing

    transform[0,0] = [[xVec], [yVec], [zVec]]
    firstPixelCenter = dicomPosition
    transform[3,0:2] = matrix_multiply(transform[0:2,0:2], $
      -firstPixelCenter, /ATRANSPOSE)

    return, transform
end
