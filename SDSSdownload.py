import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb


class Image:
    def __init__(self, ra, dec, search_radius=5*u.arcsec, bands=['i','r','g'], image_dir='./images/', data_dir='./data/'):
        # If no units given for coords, assume degrees
        if type(ra) is u.quantity.Quantity:
            self.ra = ra
        else:
            self.ra = ra*u.deg
        if type(dec) is u.quantity.Quantity:
            self.dec = dec
        else:
            self.dec = dec*u.deg
        self.pos = coords.SkyCoord(ra=self.ra, dec=self.dec)
        self.id = SDSS.query_region(self.pos, radius=5*u.arcsec)
        self.objid = self.id[0]['objid']

        # Set filter bands
        # Ensure that there are 3 bands
        # and that they are ugriz
        ugriz = ['u','g','r','i','z']
        self.bands = bands
        if len(self.bands) != 3:
            print('Must select 3 bands.\n')
            print('Using default bands (i, r, g).')
            self.bands = ['i', 'r', 'g']
        valid = True
        for band in self.bands:
            if band not in ugriz:
                valid = False
        if valid == False:
            print('Must select only ugriz filter bands.')
            print('Using default bands (i, r, g).')
            self.bands = ['i', 'r', 'g']
        self.dir = image_dir
        self.data_dir = data_dir
        
        self.hdus = []
        image_list = []
        for band in self.bands:
            hdu = SDSS.get_images(matches=self.id, band=band)
            self.hdus.append(hdu)
            image_list.append(hdu[0][0].data)
            
        xx = len(image_list[0])
        yy = len(image_list[0][0])
        
        image = np.zeros((xx,yy,3))
        for k in range(3):
            for i in range(len(image_list[k])):
                for j in range(len(image_list[k][i])):
                    image[i,j,k] = image_list[k][i,j]
                    
        self.image_list = image_list
        self.image = image
        self.cutout = []
        self.image_size = 0

    def data(self, image_size=151):
        if image_size != self.image_size:
            self.image_size=image_size
            cutout = []
            for i in range(len(self.bands)):
                wcs = WCS(self.hdus[i][0][0])
                cut = Cutout2D(self.image_list[i],self.pos,image_size*u.pixel,wcs)
                cutout.append(cut.data)
            a,b,c = cutout.shape
            cutout_reshaped = np.zeros((b, c, a))
            for k in range(a):
                for i in range(b):
                    for j in range(c):
                        cutout_reshaped[i,j,k] = cutout[k][i,j]
            self.cutout = cutout_reshaped
        return self.cutout

    def plot(self, filter='r'):
        if len(self.cutout) == 0:
            self.cutout = self.data()
        if filter==self.bands[0]:
            i = 0
        if filter==self.bands[1]:
            i = 1
        if filter==self.bands[2]:
            i = 2
        if len(self.cutout) == 0:
            self.cutout = self.data()
        image = self.cutout[:,:,i]
        plt.imshow(image, norm="log", cmap="binary", origin="lower")
        
    def plot_color(self, Q=10, stretch=0.5):
        if len(self.cutout) == 0:
            self.cutout = self.data()
        lupton = make_lupton_rgb(self.cutout[:,:,0], self.cutout[:,:,1], self.cutout[:,:,2], Q=Q, stretch=stretch)
        plt.imshow(lupton, origin='lower')
          
    def region_data(self):
        return self.image

    def plot_region(self, filter='r', box=False):
        if filter==self.bands[0]:
            i = 0
        if filter==self.bands[1]:
            i = 1
        if filter==self.bands[2]:
            i = 2
        plt.imshow(self.image[:,:,i], norm="log", cmap="binary", origin="lower")
        if box:
            wcs = WCS(self.hdus[i][0][0])
            x,y = wcs.world_to_pixel(self.pos)
            plt.gca().add_patch(Rectangle((x-75,y-75),151,151,edgecolor='r', fill=False))

    def plot_region_color(self, Q=10, strech=0.5, box=False):
        lupton = make_lupton_rgb(self.image[:,:,0], self.image[:,:,1], self.image[:,:,2], Q=Q, stretch=strech)
        plt.imshow(lupton, origin='lower')
        if box:
            wcs = WCS(self.hdus[0][0][0])
            x,y = wcs.world_to_pixel(self.pos)
            plt.gca().add_patch(Rectangle((x-75,y-75),151,151,edgecolor='r', fill=False))

    def save_plot(self, filter='r', dir=None):
        if len(self.cutout) == 0:
            self.cutout = self.data()
        if dir:
            self.dir = dir
        self.plot(filter=filter)
        plt.axis("off")
        plt.savefig(self.dir + f'{self.objid}.png')
        plt.close()

    def save_plot_region(self, filter='r', dir=None):
        if dir:
            self.dir = dir
        self.plot_region(filter=filter)
        plt.axis("off")
        plt.savefig(self.dir + f'{self.objid}.png')
        plt.close()

    def save_plot_color(self, cutout=True, Q=10, stretch=0.5, dir=None):
        if len(self.cutout) == 0:
            self.cutout = self.data()
        if dir:
            self.dir = dir
        self.plot_color(Q=Q, stretch=stretch)
        plt.axis("off")
        plt.box(False)
        plt.savefig(self.dir + f'{self.objid}.png')
        plt.close()

    def save_plot_region_color(self, cutout=True, Q=10, stretch=0.5, dir=None):
        if dir:
            self.dir = dir
        self.plot_region_color(Q=Q, stretch=stretch)
        plt.axis("off")
        plt.savefig(self.dir + f'{self.objid}_region.png')
        plt.close()

    def save_data(self, dir=None):
        if dir:
            self.data_dir = dir
        if len(self.cutout) == 0:
            self.cutout = self.data()
        np.save(self.data_dir + f'{self.objid}', self.cutout)

    def save_data_region(self, dir=None):
        if dir:
            self.data_dir = dir
        np.save(self.data_dir + f'{self.objid}_region', self.image)
        