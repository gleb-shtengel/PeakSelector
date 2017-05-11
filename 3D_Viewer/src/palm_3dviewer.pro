;------------------------------------------------------------------------------
;+
; Main launch routine for the 3-dimensional viewer
;
; @History
;   April, 2008 : Daryl Atencio, ITT VIS GSG - Original version
;
; @param FilterIndex {in}{type=vector}{required}
;   Peak filter vector.
;
; @param FunctionIndex {in}{type=vector}{required}
;   Peak function vector.
;
; @keyword ACCUMULATION {in}{type=vector}{required}
;   Sum or envelope?
;
; @keyword GROUP_LEADER {in}{type=vector}{required}
;   Widget ID of the main PeakSelector window.
;
; @keyword GROUP_PARAMS_POINTER {in}{type=vector}{required}
;   Unused
;
; @keyword GROUP_SIZE {in}{type=vector}{required}
;   Number of groups.
;
; @keyword LABEL {in}{type=vector}{required}
;   Unused
;
; @keyword NANOS_PER_CCD_PIXEL {in}{type=vector}{required}
;   CCD pixel dimensions in nm (assumed square)
;
; @keyword PARAMETER_LIMITS {in}{type=vector}{required}
;   CGroupParams parameter limits.
;
; @keyword VERBOSE {in}{type=vector}{required}
;   Output messages if set.
;-
;------------------------------------------------------------------------------
pro PALM_3DViewer, FilterIndex, FunctionIndex, $
    ACCUMULATION=Accumulation, $
    GROUP_LEADER=GroupLeader, $
    GROUP_PARAMS_POINTER=pGroupParams, $
    GROUP_SIZE=GroupSize, $
    LABEL=label, $
    NANOS_PER_CCD_PIXEL=nanos, $
    PARAMETER_LIMITS=ParameterLimits, $
    VERBOSE=verbose

    !except = 0

    doVerbose = keyword_set(verbose)

    oPalette = obj_new('IDLgrPalette')
; default color table
    oPalette -> Loadct, 3
    oPalette -> GetProperty, $
        RED=r, $
        GREEN=g, $
        BLUE=b
    obj_destroy, oPalette
    ColorTable = [[r],[g],[b]]

; ******************** Defaults ********************
    if (getenv('PALM') ne '') then begin
        print, 'Using development settings'
        CompositeFunction = 1 ; 0=alpha blending, 1=MIP, 2=alpha sum, 3=average intensity, 4=Hue display
        DataRange = [0.0,100.0] ; [v0,v1] such that 0.0 <= v0 < v1 and v0 < v1 <= 1.0
        MaxScale = 1.0 ; 0.01 <= v <= 1.0
        MaxVolumeDimension = 400 ; 100 <= v <= 1000
        SubVolumeWidth = 100 ; [20,40,60,80,100,120,140,160]
        doVerbose = 1 ; 0=off 1=on
        UseHue = 0; 0=off, 1=on
        UseEDM = 1
        const = 1 ; 0=off, 1=on
        gammaRender = 0.7
        zScaleFactor = 3.0 ; 0.1 <= v <= 10.0
        brightness = 1.0
    endif else begin
        CompositeFunction = 1 ; 0=alpha blending, 1=MIP, 2=alpha sum, 3=average intensity, 4=Hue display
        DataRange = [0.0,100.0] ; [v0,v1] such that 0.0 <= v0 < v1 and v0 < v1 <= 1.0
        MaxScale = 1.0 ; 0.01 <= v <= 1.0
        MaxVolumeDimension = 800 ; 100 <= v <= 1000
        SubVolumeWidth = 100 ; [20,40,60,80,100,120,140,160]
        doVerbose = 1 ; 0=off 1=on
        UseHue = 0; 0=off, 1=on
        UseEDM = 1
        const = 1
        gammaRender = 0.7
        zScaleFactor = 1.0 ; 0.1 <= v <= 10.0
        brightness = 1.0
    endelse
; **************************************************

; Initialize the main GUI
    oMainGUI = obj_new('PALM_MainGUI', $
        BACKGROUND_COLOR=[0,0,0], $
        COLOR_TABLE=ColorTable, $
        COMPOSITE_FUNCTION=CompositeFunction, $
        DATA_RANGE=DataRange, $
        ERROR_MESSAGE=errMsg, $
        /HIDE, $
        MAXIMUM_SCALE=MaxScale, $
        MAXIMUM_VOLUME_DIMENSION=maxVolumeDimension, $
        SUBVOLUME_WIDTH=SubVolumeWidth, $
        VERBOSE=doVerbose, $
        Z_SCALE_FACTOR=zScaleFactor)

; Initialize the PALM display model
    oModel = obj_new('PALMgr3DModel', $
        ACCUMULATION=Accumulation, $
        AUTO_FILTER=0, $
        LOG_FILTER=0, $
        COLOR_TABLE=ColorTable, $
        COMPOSITE_FUNCTION=CompositeFunction, $
        DATA_RANGE=DataRange*MaxScale, $
        FILTER_INDEX=FilterIndex, $
        FUNCTION_INDEX=FunctionIndex, $
        GAMMA_RENDER=gammaRender, $
        MAXIMUM_VOLUME_DIMENSION=MaxVolumeDimension, $
        NANOS_PER_CCD_PIXEL=nanos, $
        PARAMETER_LIMITS=ParameterLimits, $
        SUBVOLUME_WIDTH=SubVolumeWidth, $
        USE_HUE=UseHue, $
        USE_EDM=UseEDM, $
        CONSTANT=const, $
        VERBOSE=doVerbose, $
        Z_SCALE_FACTOR=zScaleFactor,  $
        BRIGHTNESS=brightness,  $
        MAIN_GUI=oMainGUI)

    if ~obj_valid(oModel) then $
        return

    oMainGUI -> UpdateModel, oModel

    if strupcase(!version.OS_Family) EQ 'UNIX' then $
        oMainGUI -> BaseEvent ; No expose event on UNIX
    oMainGUI -> SetProperty, HIDE=0

end
