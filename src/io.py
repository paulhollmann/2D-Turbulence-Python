from netCDF4 import Dataset
import numpy as np

class netCDFwriter(object):

    def __init__(self, name, flow, downsample=1) -> None:

        self.ds = downsample  # downsampling factor

        # compute output resolution
        self.nx_out = flow.nx // self.ds
        self.ny_out = flow.ny // self.ds

        # dataset
        self.data = Dataset(name + '.nc', 'w', 'NETCDF4')

        # dimensions
        self.data.createDimension('x', self.nx_out)
        self.data.createDimension('y', self.ny_out)
        self.data.createDimension('t', None)

        # coordinates (downsampled)
        self.x = self.data.createVariable('x', 'float32', ('x',))
        self.x[:] = flow.x[::self.ds]

        self.y = self.data.createVariable('y', 'float32', ('y',))
        self.y[:] = flow.y[::self.ds]

        self.t = self.data.createVariable('t', 'float32', ('t',))

        # fields
        self.w = self.data.createVariable('w', 'float32', ('t', 'y', 'x'))
        self.w.setncattr('units', '1/s')

        #self.u = self.data.createVariable('u', 'float32', ('t', 'y', 'x'))
        #self.u.setncattr('units', 'L/s')

        #self.v = self.data.createVariable('v', 'float32', ('t', 'y', 'x'))
        #self.v.setncattr('units', 'L/s')

        self.lapw = self.data.createVariable('lapw', 'float32', ('t', 'y', 'x'))
        self.lapw.setncattr('units', '1/s^3')

        # forcing variables
        self.forced = flow.forced
        if self.forced:
            self.alpha = self.data.createVariable('alpha', 'float32', ('t',))
            self.alpha.setncattr('description', 'forcing coefficient')

            self.phif = self.data.createVariable('phif', 'float32', ('t', 'y', 'x'))
            self.phif.setncattr('description', 'physical forcing field')

        # damping variables
        self.dragged = flow.dragged
        if self.dragged:
            self.phid = self.data.createVariable('phid', 'float32', ('t', 'y', 'x'))
            self.phid.setncattr('description', 'physical damping field')

        self.c = 0

        self.add(flow)

    def _ds2(self, arr):
        """Helper: 2D downsampling"""
        return arr[::self.ds, ::self.ds]

    def add(self, flow) -> None:

        self.t[self.c] = flow.time

        # vorticity
        w_full = np.fft.irfft2(flow.wh, axes=(-2, -1))
        self.w[self.c, :, :] = self._ds2(w_full)

        # velocity
        #flow.get_u()
        #flow.get_v()
        #self.u[self.c, :, :] = self._ds2(flow.u)
        #self.v[self.c, :, :] = self._ds2(flow.v)

        # laplacian
        lapw = flow.get_laplace_w()
        self.lapw[self.c, :, :] = self._ds2(lapw)

        # scalar (forcing coefficient)
        if self.forced:
            self.alpha[self.c] = flow.alpha


        # --- forcing field ---
        if self.forced:
            phif_h = np.zeros_like(flow.wh)
            phif_h[flow.forcing_mask] = flow.alpha * flow.wh[flow.forcing_mask]
            phif = np.fft.irfft2(phif_h, axes=(-2, -1))
            self.phif[self.c, :, :] = self._ds2(phif)


        # --- damping field ---
        if flow.dragged:
            phid_h = np.zeros_like(flow.wh)
            phid_h[flow.damping_mask] = -flow.drag_coeff * flow.wh[flow.damping_mask]
            phid = np.fft.irfft2(phid_h, axes=(-2, -1))
            self.phid[self.c, :, :] = self._ds2(phid)

        self.c += 1

    def close(self) -> None:
        self.data.close()