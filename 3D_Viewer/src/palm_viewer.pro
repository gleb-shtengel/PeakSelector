;------------------------------------------------------------------------------
;+
; Standalone 3D Viewer for PALM data.
;
; @Author
;   R. Kneusel, ITT Visual Information Solutions Global Services Group
;
; @History
;   Jan 2009 : RTK, ITT VIS GSG
;-
;------------------------------------------------------------------------------
pro PALM_Viewer
    compile_opt idl2, logical_predicate

    ;  Default color table
    oPalette = obj_new('IDLgrPalette')
    oPalette -> Loadct, 3
    oPalette -> GetProperty, RED=r, GREEN=g, BLUE=b
    obj_destroy, oPalette
    ColorTable = [[r],[g],[b]]

    ;  Default values
    CompositeFunction = 1 ; 1=MIP
    DataRange = [0.0,100.0] ; [v0,v1] such that 0.0 <= v0 < v1 and v0 < v1 <= 1.0
    MaxScale = 1.0 ; 0.01 <= v <= 1.0
    MaxVolumeDimension = 800 ; 100 <= v <= 1000
    SubVolumeWidth = 100 ; [20,40,60,80,100,120,140,160]
    doVerbose = 1 ; 0=off 1=on
    UseHue = 0; 0=off, 1=on
    UseEDM = 1
    useShear = 0
    shearAmount = 0
    zScaleFactor = 4.0 ; 0.1 <= v <= 10.0

    ; Initialize the main GUI
    oMainGUI = obj_new('PALM_MainGUI', $
        BACKGROUND_COLOR=[0,0,0], $
        COLOR_TABLE=ColorTable, $
        COMPOSITE_FUNCTION=CompositeFunction, $
        DATA_RANGE=DataRange, $
        ERROR_MESSAGE=errMsg, $
        HIDE=0, $
        MAXIMUM_SCALE=MaxScale, $
        MAXIMUM_VOLUME_DIMENSION=maxVolumeDimension, $
        SUBVOLUME_WIDTH=SubVolumeWidth, $
        VERBOSE=doVerbose, $
        Z_SCALE_FACTOR=zScaleFactor,  $
        /STANDALONE)
end

