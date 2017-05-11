
   pro aviris, just_gpu = just_gpu, nodisplay = nodisplay, _extra=e

      ; demonstrates matrix operations for GPULIB
      ; Tech-X corporation, 2008
      ; fillmore@txcorp.com

      ; gpuinit, _extra=e

      nwvl = 224
      nx = 614
      ny = 512

      data = intarr(nwvl, nx, ny)

      file = '../aviris_data/f970620t01p02_r03_sc03.a.rfl'
      openr, lun, file, /get_lun
      readu, lun, data
      close, lun
      data = data / 1.0e4

      data_blue = total(data(0:20, *, *), 1)
      data_green = total(data(10:30, *, *), 1)
      data_red = total(data(30:40, *, *), 1)

      image = bytarr(nx, ny, 3)
      image(*, *, 0) = bytscl(data_red)
      image(*, *, 1) = bytscl(data_green)
      image(*, *, 2) = bytscl(data_blue)

      window, xsize = nx, ysize = ny

      for kx = 2, nx - 3 do begin
      for ky = 2, ny - 3 do begin

         spectrum = data(*, kx, ky)

         spectrum_norm = sqrt(transpose(spectrum) # spectrum)

         data_matrix = reform(data, [nwvl, long(nx) * ny])

         for k = long(0), long(nx) * ny - 1 do begin
            s_norm = sqrt(transpose(data_matrix[*, k]) # data_matrix[*, k])
            s_norm_inv = 1.0 / s_norm[0]
            data_matrix[*, k] =  s_norm_inv * data_matrix[*, k]
         endfor

         data_product = spectrum # data_matrix / spectrum_norm[0]

         r = max(data_product, idx)

         mask = where((1.0 - data_product) lt 0.5)

         iy = idx / nx
         ix = idx mod nx

         tmp = image(*, *, 0)
         tmp(mask) = 0
         image(*, *, 0) = tmp
         tmp = image(*, *, 1)
         tmp(mask) = 1
         image(*, *, 1) = tmp
         tmp = image(*, *, 2)
         tmp(mask) = 2
         image(*, *, 2) = tmp

         image(kx-2:kx+2, ky-2:ky+2, *) = 255

         tv, image, true = 3

      endfor
      endfor

   end

