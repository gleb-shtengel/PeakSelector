pro plot_fit, x, p
  y = p[0]*sin(p[1]*x) + p[2]*exp(-(x-p[3])^2/p[4]) + p[5]
  oplot, x,y, COLOR=2551234
end

