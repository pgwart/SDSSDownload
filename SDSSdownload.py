import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import astropy.units as u
import astropy.cosmology.units as cu
u.add_enabled_units(cu)  
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import make_lupton_rgb
import pandas as pd
import os

def size_from_z(p90,z,box_scale=2.5):
    """
    Finds image size for a galaxy.
    p90: Petrosian R90
    z: Redshift
    box_scale: Size of box relative to angular diameter
    """
    if p90 == 999999.0:
        return 151*u.pixel
    z = z*cu.redshift
    d = z.to(u.parsec, cu.with_redshift())
    s = 1000*p90*u.parsec
    radius = (s/d)*u.rad
    diameter = 2*radius
    return diameter*box_scale
    
class Image:
    def __init__(self, ra, dec, df_path, search_radius=5*u.arcsec, bands=['r'], data_dir='./data/', show_progress=False):
        # If no units given for coords, assume degrees
        if type(ra) is u.quantity.Quantity:
            self.ra = ra
        else:
            self.ra = ra*u.deg
        if type(dec) is u.quantity.Quantity:
            self.dec = dec
        else:
            self.dec = dec*u.deg

        # Default image size = 151 pixels
        self.size = 151

        # Flag for whether the image is on the plate
        self.on_plate = 1

        # Get coordinates and SDSS ID
        self.pos = coords.SkyCoord(ra=self.ra, dec=self.dec)
        self.id = SDSS.query_region(self.pos, radius=5*u.arcsec)
        
        # Find search result that matches one of our SDSS galaxies
        # else, use the first result
        self.index = 0
        df = pd.read_csv(df_path)
        IDs = [id for id in df['objID']]
        for i,objid in enumerate(self.id['objid']):
            if objid in IDs:
                self.index = i
        self.objid = self.id[self.index]['objid']
        df_cut = df[df['objID'] == self.objid]

        # Get rid of DataFrame to save memory (hopefully?)
        del df

        # Set filter bands
        # Ensure that there are 3 bands
        # and that they are ugriz
        ugriz = ['u','g','r','i','z']
        self.bands = bands
        valid = True
        if bands == 'all':
            self.bands = ugriz
        else:
            for band in self.bands:
                if band not in ugriz:
                    valid = False
            if valid == False:
                print('Must select only ugriz filter bands.')
                print('Using default band (r).')
                self.bands = ['r']
            
        self.data_dir = data_dir
        self.gal_dir = data_dir + f'{self.objid}/'
        
        if not os.path.exists(self.gal_dir):
            os.makedirs(self.gal_dir)
        
        for band in self.bands:
            hdu = SDSS.get_images(matches=self.id, band=band, show_progress=show_progress)
            hdu[0].writeto(self.gal_dir + 'plate_' + band + '.fits', overwrite=True)

    def cutout(self, size=None):
        if size == None:
            size = self.size
        if type(size) == int:
            if size%2 == 0:
                print('For best results, use odd pixel size.')
            size = size*u.pixel
        for band in self.bands:
            filename = self.gal_dir + 'plate_' + band + '.fits'
            #cut = astrocut.fits_cut(self.gal_dir + f'plate_' + band + '.fits', self.pos, cutout_size=size, memory_only=True)
            #cut[0].writeto(self.gal_dir + 'cutout_' + band + '.fits', overwrite=True)

            hdu = fits.open(filename)[0]
            wcs = WCS(hdu.header)
                
            try:
                cutout = Cutout2D(hdu.data, position=self.pos, size=size, wcs=wcs, mode='strict')
            except:
                cutout = Cutout2D(hdu.data, position=self.pos, size=size, wcs=wcs, mode='partial', fill_value=0.0)
                self.on_plate = 0
                
            hdu.data = cutout.data
        
            hdu.header.update(cutout.wcs.to_header())
            cutout_filename = self.gal_dir + 'cutout_' + band + '.fits'
            hdu.writeto(cutout_filename, overwrite=True)

    def plot_cutout(self, band=None, save=False):
        if band == None:
            band = self.bands[0]
        with fits.open(self.gal_dir + 'cutout_' + band + '.fits') as hdu:
            norm = LogNorm(np.min(hdu[0].data), np.max(hdu[0].data))
            plt.axis('off')
            fig = plt.imshow(hdu[0].data, norm="log", cmap="binary", origin="lower")
            if save:
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.savefig(self.gal_dir + 'cutout_' + band + '.png', bbox_inches='tight', pad_inches=0)
                
    def save_plot(self, band=None):
        if band == None:
            band = self.bands[0]
        with fits.open(self.gal_dir + 'cutout_' + band + '.fits') as hdu:
            norm = LogNorm(np.min(hdu[0].data), np.max(hdu[0].data))
            plt.axis('off')
            fig = plt.imshow(hdu[0].data, norm="log", cmap="binary", origin="lower")
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig(self.gal_dir + 'cutout_' + band + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()

    def plot_cutout_color(self, red='r', green='g', blue='u', Q=10, stretch=0.5, save=False):
        with fits.open(self.gal_dir + 'cutout_' + red + '.fits') as red_hdu:
            with fits.open(self.gal_dir + 'cutout_' + green + '.fits') as green_hdu:
                with fits.open(self.gal_dir + 'cutout_' + blue + '.fits') as blue_hdu:
                    data = [red_hdu[0].data, green_hdu[0].data, blue_hdu[0].data]
                    axes = []
                    for array in data:
                        i,j = np.shape(array)
                        axes.append(i)
                        axes.append(j)
                    length = min(axes)
                    reshaped = np.zeros((length, length, 3))
                    for k in range(3):
                        for i in range(length):
                            for j in range(length):
                                reshaped[i,j,k] = data[k][i,j]
                    image = make_lupton_rgb(reshaped[:,:,0],reshaped[:,:,1],reshaped[:,:,2], Q=Q, stretch=stretch)

                    plt.axis('off')
                    fig = plt.imshow(image, origin='lower')
                    if save:
                        fig.axes.get_xaxis().set_visible(False)
                        fig.axes.get_yaxis().set_visible(False)
                        plt.savefig(self.gal_dir + 'cutout_color.png', bbox_inches='tight', pad_inches=0)

    def save_cutout_color(self, red='r', green='g', blue='u', Q=10, stretch=0.5, save=False):
        with fits.open(self.gal_dir + 'cutout_' + red + '.fits') as red_hdu:
            with fits.open(self.gal_dir + 'cutout_' + green + '.fits') as green_hdu:
                with fits.open(self.gal_dir + 'cutout_' + blue + '.fits') as blue_hdu:
                    data = [red_hdu[0].data, green_hdu[0].data, blue_hdu[0].data]
                    axes = []
                    for array in data:
                        i,j = np.shape(array)
                        axes.append(i)
                        axes.append(j)
                    length = min(axes)
                    reshaped = np.zeros((length, length, 3))
                    for k in range(3):
                        for i in range(length):
                            for j in range(length):
                                reshaped[i,j,k] = data[k][i,j]
                    image = make_lupton_rgb(reshaped[:,:,0],reshaped[:,:,1],reshaped[:,:,2], Q=Q, stretch=stretch)

                    plt.axis('off')
                    fig = plt.imshow(image, origin='lower')
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
                    plt.savefig(self.gal_dir + 'cutout_color.png', bbox_inches='tight', pad_inches=0)
                    plt.close()
                    

    def save_all(self):
        for band in self.bands:
            self.save_plot(band=band)
            