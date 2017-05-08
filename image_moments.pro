PRO image_moments, image, order, moments
;
;	M00=total(image)
;	Xave=sum[um(x*image[y,x])]
;	Yave=
;
size=sqrt(n_elements(image))
coord=indgen(size)
moments=fltarr(order+1,order+1)

M00=total(image)
xave=total(image##coord)/total(image)
yave=total(image#coord)/total(image)

for i=0,order do begin
	for j=0,order	do begin
		moments[i,j]=(image##(coord-xave)^i)#(coord-yave)^j
	endfor
endfor
moments=moments/moments[0,0]
end