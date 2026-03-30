from netCDF4 import Dataset
import numpy as np

class netCDFwriter(object):

    def __init__(self, name, flow) -> None:

        # the data set that we use to store data
        self.data = Dataset(name+'.nc','w','NETCDF4') # using netCDF4 for output format
        
        # two space dimension and a time dimension
        self.data.createDimension('x',flow.nx)
        self.data.createDimension('y',flow.ny)
        self.data.createDimension('t',None)

        # fill-in the coordinates
        self.x = self.data.createVariable('x','float32',('x'))
        self.x[:] = flow.x
        self.y = self.data.createVariable('y','float32',('y'))
        self.y[:] = flow.y
        self.t = self.data.createVariable('t','float32',('t'))

        # set up the vorticity
        self.w = self.data.createVariable('w','float32',('t','y','x'))
        self.w.setncattr('units','1/s')

        # set up the velocity
        self.u = self.data.createVariable('u', 'float32', ('t', 'y', 'x'))
        self.u.setncattr('units', 'L/s')
        self.v = self.data.createVariable('v', 'float32', ('t', 'y', 'x'))
        self.v.setncattr('units', 'L/s')

        # Laplacian of vorticity
        self.lapw = self.data.createVariable('lapw', 'float32', ('t', 'y', 'x'))
        self.lapw.setncattr('units', '1/s^3')

        # forcing coefficient alpha (time dependent scalar)
        self.alpha = self.data.createVariable('alpha', 'float32', ('t',))
        self.alpha.setncattr('description', 'forcing coefficient')

        # counter
        self.c = 0

        # add initial flow data
        self.add(flow)


    def add(self, flow) -> None:

        # set the time
        self.t[self.c] = flow.time

        # write vorticity
        self.w[self.c,:,:] = np.fft.irfft2(flow.wh, axes=(-2,-1))

        # write velocity
        flow.get_u()
        flow.get_v()
        self.u[self.c, :, :] = flow.u
        self.v[self.c, :, :] = flow.v

        # write Laplacian of vorticity
        lapw = flow.get_laplace_w()
        self.lapw[self.c, :, :] = lapw

        # write forcing coefficient
        self.alpha[self.c] = flow.alpha

        # update counter
        self.c += 1


    def close(self) -> None:
        # close the Dataset, not mandatory
        self.data.close()