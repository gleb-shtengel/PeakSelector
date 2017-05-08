pro PALMgrModel::Cleanup

    ptr_free, self.pVolume
    obj_destroy, self.oVolume
    self -> IDLgrModel::Cleanup

end

pro PALMgrModel::ConvertCoords

    self.oVolume -> GetProperty, XRANGE=xRange, $
                                 YRANGE=yRange, $
                                 ZRANGE=zRange
    ranges = [[xRange], [yRange], [zRange]]
    max_range = max(ranges[1,*]-ranges[0,*], maxSub)
    refRange = ranges[*,maxSub]
    xs = [-(xRange[0])+((refRange[1]-refRange[0])/2-(xRange[1]-xRange[0])/2),1]/(refRange[1]-refRange[0])
    ys = [-(yRange[0])+((refRange[1]-refRange[0])/2-(yRange[1]-yRange[0])/2),1]/(refRange[1]-refRange[0])
    zs = [-(zRange[0])+((refRange[1]-refRange[0])/2-(zRange[1]-zRange[0])/2),1]/(refRange[1]-refRange[0])
    xs[0] = xs[0] + (-0.5)
    ys[0] = ys[0] + (-0.5)
    zs[0] = zs[0] + (-0.5)
    self.oVolume->SetProperty, XCOORD_CONV=xs, YCOORD_CONV=ys, ZCOORD_CONV=zs

end


pro PALMgrModel::GetProperty, $
    REFERENCE_RANGE=refRange, $
    X_RANGE=xRange, $
    XCOORD_CONV=xs, $
    Y_RANGE=yRange, $
    YCOORD_CONV=ys, $
    Z_RANGE=zRange, $
    ZCOORD_CONV=zs, $
    _REF_EXTRA=_extra

    if arg_present(refRange) then begin
        self.oVolume -> GetProperty, XRANGE=xRange, $
                                   YRANGE=yRange, $
                                   ZRANGE=zRange
        ranges = [[xRange], [yRange], [zRange]]
        max_range = max(ranges[1,*]-ranges[0,*], maxSub)
        refRange = ranges[*,maxSub]
    endif
    if arg_present(xs) then $
        self.oVolume -> GetProperty, XCOORD_CONV=xs
    if arg_present(ys) then $
        self.oVolume -> GetProperty, YCOORD_CONV=ys
    if arg_present(zs) then $
        self.oVolume -> GetProperty, ZCOORD_CONV=zs
    if arg_present(xRange) then $
        self.oVolume -> GetProperty, XRANGE=xRange
    if arg_present(yRange) then $
        self.oVolume -> GetProperty, YRANGE=yRange
    if arg_present(zRange) then $
        self.oVolume -> GetProperty, ZRANGE=zRange
    if n_elements(_extra) GT 0 then begin
        self -> IDLgrModel::GetProperty, _EXTRA=_extra
    endif

end


function PALMgrModel::GetVolPtr

    if ~ptr_valid(self.pVolume) then $
        return, 0

    return, self.pVolume

end


function PALMgrModel::Init, volData

    void = self -> IDLgrModel::Init()
    volDims = [10,13,5]
    volData = randomu(seed,volDims)
    volData = bytscl(volData)
    self.pVolume = ptr_new(volData, /NO_COPY)
    self.oVolume = obj_new('IDLgrVolume', *self.pVolume)
    self -> Add, self.oVolume
    self -> ConvertCoords

    return, 1

end


pro PALMgrModel__Define
    void = {PALMgrModel, $
            inherits IDLgrModel, $
            pVolume : ptr_new(), $
            oVolume : obj_new()}
end