from PIL import Image
import skimage
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

y_crop = 950

def load_img( fp, convert_gray=True, channel=-1):
    # Load RGB Image
    img_rgb = imread(fp)
    #print( img_rgb.dtype )
    #print( np.shape(img_rgb) )
    # Convert to Gray Image
    img_gray = skimage.color.rgb2gray( img_rgb )
    #print( img_gray.dtype )
    #print( np.shape(img_gray) )
    
    if convert_gray:
        return img_gray
    elif channel==0 or channel==1 or channel==2:
        return img_rgb[:,:,channel]
    else:
        return img_rgb

def plot_hist( img, bins=30, exclude_under=0, exclude_over=9999999):
    img_flat = []
    img = img[0:y_crop,:]
    
    for i in img.flatten():
        if i >= exclude_under and i <= exclude_over:
            img_flat.append(i)

    plt.hist(img_flat, bins=bins)
    plt.ylabel('Pixel count')
    plt.xlabel('Relative intensity')
    
# Takes in coordinates as strings, coordinates are seperated by commas
def get_dist(coor1, coor2):
    y1, x1 = coor1
    y2, x2 = coor2
    dist = ( (int(x1) - int(x2))**2 + (int(y1) - int(y2))**2 )**0.5
    return dist

# Returns
def get_stats_gray( img, exclude_under=0 ):
    img = img[0:y_crop,:]
    img_flat = img.flatten()
    
    if exclude_under > 0:
        img_flat_tmp = img_flat.copy()
        img_flat = []
        for p in img_flat_tmp:
            if p > exclude_under:
                img_flat.append(p)
    
    mean = np.mean(img_flat)
    std = np.std(img_flat)
    var = np.var(img_flat)
    minI = np.amin(img_flat)
    maxI = np.amax(img_flat)
    
    print( 'Mean:', round(mean,4) )
    print( 'Std:', round(std,4) )
    print( 'Var:', round(var,4) )
    print( 'Min:', round(minI,4) )
    print( 'Max:', round(maxI,4) )
    
    return (mean, std, var, minI, maxI)

def get_avg_pixel_val(img_array, y, x, radius=3):
    total_val = 0.0
    num_pixels = 0
    
    for x_offset in range( 1-radius,radius):
        for y_offset in range( 1-radius,radius):
            if abs(x_offset)+abs(y_offset)>radius:
                continue
            try:
                total_val += img_array[y+y_offset,x+x_offset]
                num_pixels +=1
            except Exception as e:
                continue
    return total_val/num_pixels

def get_ROI(img_array, background='light'): # background='light'/'dark'
    """
    background="light": 3 channels. blue=green=0 red=255
    background="dark" : 1 channel. brightness over 240
    """
    ROI_locations = []
    ROI_min_separation = 25
    
    # If Image is white convert to gray and find dark areas
    if background=='light':
        y_len, x_len, _ = np.shape(img_array)
        
        for y in range(y_len):
            for x in range(x_len):
                p_r, p_g, p_b = img_array[y,x]
                # Check if is red dot
                if p_b==0 and p_g==0 and p_r==255:
                    too_close = False
                    # Check if any ROI is too close
                    for ROI in ROI_locations:
                        if get_dist( ROI, [y,x]) < ROI_min_separation:
                            too_close = True
                    if not too_close:
                            ROI_locations.append([y,x])
    
    # If Image is dark take the red channel and find white areas
    if background=='dark':
        y_len, x_len = np.shape(img_array)
        
        for y in range(y_len):
            for x in range(x_len):
                pixel = img_array[y,x]
                # Skip pixels that are not bright enough
                if pixel > 235:
                    dot_avg = get_avg_pixel_val( img_array, y, x, radius=10)
                    # Bright enough to be considered the red dot
                    if dot_avg > 238:
                        too_close = False
                        # Check if any ROI is too close
                        for ROI in ROI_locations:
                            if get_dist( ROI, [y,x]) < ROI_min_separation:
                                too_close = True
                        if not too_close:
                            ROI_locations.append([y,x])
    return ROI_locations

def img_light_or_dark( img_array ):
    r,g,b = img_array[0,0,:]
    if r==255 and g==255 and b==255:
        return 'light'
    else:
        return 'dark'

def image_to_heatmap(img_array):
    # CROP
    img_array = img_array[0:y_crop,:]

    y_len, x_len = np.shape(img_array)
    new_img_array = np.zeros((y_len, x_len))

    for y in range(y_len):
        for x in range(x_len):
            pixel = img_array[y,x]
            new_img_array[y,x] = get_avg_pixel_val(img_array,y,x)
            
    return new_img_array

def get_brightest_coors(img_array, \
                        brightness_threshold=999, \
                        minimum_distance=20, \
                        radius=1, \
                        plot_result=False, \
                        convert_gray=True, \
                        channel=-1):
    # CROP
    img_array = img_array[0:y_crop,:]

    y_len, x_len = np.shape(img_array)
    bright_coordinates = []
    for y in range(y_len):
        for x in range(x_len):
            if radius==1:
                pixel = img_array[y,x]
            else:
                pixel = get_avg_pixel_val(img_array, y, x, radius=radius)
                
            if pixel > brightness_threshold:
                points_too_close_together = False
                coordinates = [y,x]
                # Do NOT add coordinate if it is too close to another coordinate
                for coor_temp in bright_coordinates:
                    if get_dist(coor_temp,coordinates) < minimum_distance:
                        points_too_close_together = True
                        break
                    if len(bright_coordinates) > 1000:
                        print('TOO MANY POINTS')
                        return ''
                if not points_too_close_together:
                    bright_coordinates.append(coordinates)

    print(len(bright_coordinates),'bright points found')

    
    
    x_coors = []
    y_coors = []
    for coor in bright_coordinates:
        y, x = coor
        x_coors.append( int(x) )
        y_coors.append( int(y) )
   
    if plot_result:
        if convert_gray:
            plt.imshow(img_array, cmap='gray')
        else:
            plt.imshow(img_array)    

        plt.scatter(x=x_coors, y=y_coors, c='r', s=.5)
        plt.show()
    
    return bright_coordinates