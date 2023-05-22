import cv2 
from scipy.interpolate import interp2d
import numpy as np 
import time
from PIL import Image


def read_with_mask(img_path, mask):
    img_source = cv2.imread(img_path,cv2.IMREAD_COLOR)
    img_mosaic = np.zeros(img_source.shape)
    filter_size = len(mask)
    for each_line in range(0, len(img_source)):
        for each_pixel in range(0, len(img_source[each_line])):
            img_mosaic[each_line][each_pixel][mask[each_line%filter_size][each_pixel%filter_size]] = img_source[each_line][each_pixel][mask[each_line%filter_size][each_pixel%filter_size]]
           
    cv2.imwrite("mozaik.png", img_mosaic)
    return np.split(img_mosaic, 3, axis=2)


def find_in_line(line, no_varaible, start_point):
    list_of_varaible = []
    
    for each_pixel in range(start_point, len(line)):
        if line[each_pixel] > 0:
            list_of_varaible.append(each_pixel)
        if len(list_of_varaible) == no_varaible:
            return list_of_varaible
    return None

def find_in_column(matrix, column , no_varaible, start_point):
    list_of_varaible = []
    
    for each_pixel in range(start_point, len(matrix)):
        if matrix[each_pixel][column] > 0:
            list_of_varaible.append(each_pixel)
        if len(list_of_varaible) == no_varaible:
            return list_of_varaible
    return None

def linear(colors):
    red = np.copy(colors[2])
    green = np.copy(colors[1])
    blue = np.copy(colors[0])
    
    for each_line in range(1, len(red[0]) -1):
        start = 1
        while start < len(red[0]) -1 :
            points =  find_in_line(red[each_line], 2, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
                if red[each_line][pixel] == 0:
                    
                    red[each_line][pixel] = red[each_line][points[0]] + (pixel - points[0])*(red[each_line][points[0]] -red[each_line][points[1]])/(points[1]- points[0])
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(red) -1):
        start = 1
        while start < len(red[0]) -1 :
            points =  find_in_column(red, each_column, 2, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
               # print(red[each_column][points[0]])
                if red[pixel][each_column] == 0:
                    red[pixel][each_column] = red[points[0]][each_column] + (pixel - points[0])*(red[points[0]][each_column] -red[points[1]][each_column])/(points[1]- points[0])
            start = points[-1]
            
    for each_line in range(1, len(blue[0]) -1):
        start = 1
        while start < len(blue[0]) -1 :
            points =  find_in_line(blue[each_line], 2, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
                if blue[each_line][pixel] == 0:
                    
                    blue[each_line][pixel] = blue[each_line][points[0]] + (pixel - points[0])*(blue[each_line][points[0]] -blue[each_line][points[1]])/(points[1]- points[0])
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(blue) -1):
        start = 1
        while start < len(blue[0]) -1 :
            points =  find_in_column(blue, each_column, 2, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
               # print(blue[each_column][points[0]])
                if blue[pixel][each_column] == 0:
                    blue[pixel][each_column] = blue[points[0]][each_column] + (pixel - points[0])*(blue[points[0]][each_column] -blue[points[1]][each_column])/(points[1]- points[0])
            start = points[-1]
            
    for each_line in range(1, len(green[0]) -1):
        start = 1
        while start < len(green[0]) -1 :
            points =  find_in_line(green[each_line], 2, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
                if green[each_line][pixel] == 0:
                    
                    green[each_line][pixel] = green[each_line][points[0]] + (pixel - points[0])*(green[each_line][points[0]] -green[each_line][points[1]])/(points[1]- points[0])
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(green) -1):
        start = 1
        while start < len(green[0]) -1 :
            points =  find_in_column(green, each_column, 2, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
               # print(green[each_column][points[0]])
                if green[pixel][each_column] == 0:
                    green[pixel][each_column] = green[points[0]][each_column] + (pixel - points[0])*(green[points[0]][each_column] -green[points[1]][each_column])/(points[1]- points[0])
            start = points[-1]
            
            
            
    img = np.concatenate((blue, green, red), axis=2)
    cv2.imwrite("renew69.png", img)

def get_square_func(values, positions):
    
    a = (values[2] - values[1]) / ((positions[2] -positions[1]) * (positions[2] - positions[0])) - (values[1] - values[0]) / ((positions[1] - positions[0]) * (positions[2] - positions[0]))
    b = (values[2] - values[0]) / (positions[1] - positions[0]) - a * (positions[0] + positions[1])
    c = values[0] - a * positions[0]**2 - b*positions[0]
    return (c ,b, a)

def square(colors):
    
    red = np.copy(colors[2])
    green = np.copy(colors[1])
    blue = np.copy(colors[0])
    
    for each_line in range(1, len(red[0]) -1):
        start = 1
        while start < len(red[0]) -1 :
            points =  find_in_line(red[each_line], 3, start)
            if points == None:
                break
            func_points = get_square_func([red[each_line][points[0]],red[each_line][points[1]],red[each_line][points[2]]],points)
            for pixel in range(start, points[-1]):
                if red[each_line][pixel] == 0:
                    red[each_line][pixel] = func_points[0] + func_points[1] * pixel + func_points[2] * pixel * pixel
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(red) -1):
        start = 1
        while start < len(red[0]) -1 :
            points =  find_in_column(red, each_column, 3, start)
            if points == None:
                break
            func_points = get_square_func([red[points[0]][each_column],red[points[1]][each_column],red[points[2]][each_column]],points)
            for pixel in range(start, points[-1]):
               # print(red[each_column][points[0]])
                if red[pixel][each_column] == 0:
                    red[pixel][each_column] = func_points[0] + func_points[1] * pixel + func_points[2] * pixel * pixel
            start = points[-1]
            
    for each_line in range(1, len(blue[0]) -1):
        start = 1
        while start < len(blue[0]) -1 :
            points =  find_in_line(blue[each_line], 3, start)
            if points == None:
                break
            func_points = get_square_func([blue[each_line][points[0]],blue[each_line][points[1]],blue[each_line][points[2]]],points)
            for pixel in range(start, points[-1]):
                if blue[each_line][pixel] == 0:
                    blue[each_line][pixel] = func_points[0] + func_points[1] * pixel + func_points[2] * pixel * pixel
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(blue) -1):
        start = 1
        while start < len(blue[0]) -1 :
            points =  find_in_column(blue, each_column, 3, start)
            if points == None:
                break
            func_points = get_square_func([blue[points[0]][each_column],blue[points[1]][each_column],blue[points[2]][each_column]],points)
            for pixel in range(start, points[-1]):
               # print(blue[each_column][points[0]])
                if blue[pixel][each_column] == 0:
                    blue[pixel][each_column] = func_points[0] + func_points[1] * pixel + func_points[2] * pixel * pixel
            start = points[-1]
            
    for each_line in range(1, len(green[0]) -1):
        start = 1
        while start < len(green[0]) -1 :
            points =  find_in_line(green[each_line], 3, start)
            if points == None:
                break
            func_points = get_square_func([green[each_line][points[0]],green[each_line][points[1]],green[each_line][points[2]]],points)
            for pixel in range(start, points[-1]):
                if green[each_line][pixel] == 0:
                    green[each_line][pixel] = func_points[0] + func_points[1] * pixel + func_points[2] * pixel * pixel
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(green) -1):
        start = 1
        while start < len(green[0]) -1 :
            points =  find_in_column(green, each_column, 3, start)
            if points == None:
                break
            func_points = get_square_func([green[points[0]][each_column],green[points[1]][each_column],green[points[2]][each_column]],points)
            for pixel in range(start, points[-1]):
               # print(green[each_column][points[0]])
                if green[pixel][each_column] == 0:
                    green[pixel][each_column] = func_points[0] + func_points[1] * pixel + func_points[2] * pixel * pixel
            start = points[-1]
            
    img = np.concatenate((blue, green, red), axis=2)
    cv2.imwrite("renew969.png", img)

def get_cubic_func(values, positions, point):
    
    for i in range(1, len(positions) -1):
        if positions[i] < point and positions[i + 1] >point:
            h = positions[i+1] - positions[i]
            a = (positions[i+1] - point) / h
            b = (point - positions[i]) / h
            c = (a**3 - a) * (h**2)
            d = (b**3 - b) * (h**2)
            return a * values[i] + b * values[i+1] + c * (values[i] - values[i-1]) + d * (values[i+1] - values[i])

def cubic(colors):
    
    red = np.copy(colors[2])
    green = np.copy(colors[1])
    blue = np.copy(colors[0])
    
    for each_line in range(1, len(red[0]) -1):
        start = 1
        while start < len(red[0]) -1 :
            points =  find_in_line(red[each_line], 4, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
                if red[each_line][pixel] == 0:
                    red[each_line][pixel] = get_cubic_func([red[each_line][points[0]-1],red[each_line][points[0]],red[each_line][points[1]],red[each_line][points[2]],red[each_line][points[3]]],[points[0] -1 ] + points,pixel)
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(red) -1):
        start = 1
        while start < len(red[0]) -1 :
            points =  find_in_column(red, each_column, 4, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
               # print(red[each_column][points[0]])
                if red[pixel][each_column] == 0:
                    red[pixel][each_column] = get_cubic_func([red[points[0]-1][each_column],red[points[0]][each_column],red[points[1]][each_column],red[points[2]][each_column],red[points[3]][each_column]],[points[0] -1 ]+ points,pixel)
            start = points[-1]
            
            
    for each_line in range(1, len(blue[0]) -1):
        start = 1
        while start < len(blue[0]) -1 :
            points =  find_in_line(blue[each_line], 4, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
                if blue[each_line][pixel] == 0:
                    blue[each_line][pixel] = get_cubic_func([blue[each_line][points[0]-1],blue[each_line][points[0]],blue[each_line][points[1]],blue[each_line][points[2]],blue[each_line][points[3]]],[points[0] -1 ]+ points,pixel)
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(blue) -1):
        start = 1
        while start < len(blue[0]) -1 :
            points =  find_in_column(blue, each_column, 4, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
               # print(blue[each_column][points[0]])
                if blue[pixel][each_column] == 0:
                    blue[pixel][each_column] = get_cubic_func([blue[points[0]-1][each_column],blue[points[0]][each_column],blue[points[1]][each_column],blue[points[2]][each_column],blue[points[3]][each_column]],[points[0] -1 ]+ points,pixel)
            start = points[-1]
            
            
            
            
    for each_line in range(1, len(green[0]) -1):
        start = 1
        while start < len(green[0]) -1 :
            points =  find_in_line(green[each_line], 4, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
                if green[each_line][pixel] == 0:
                    green[each_line][pixel] = get_cubic_func([green[each_line][points[0]-1],green[each_line][points[0]],green[each_line][points[1]],green[each_line][points[2]],green[each_line][points[3]]],[points[0] -1 ]+ points,pixel)
                       
            start = points[-1]
    
    start = 1
    for each_column in range(1, len(green) -1):
        start = 1
        while start < len(green[0]) -1 :
            points =  find_in_column(green, each_column, 4, start)
            if points == None:
                break
            for pixel in range(start, points[-1]):
               # print(green[each_column][points[0]])
                if green[pixel][each_column] == 0:
                    green[pixel][each_column] = get_cubic_func([green[points[0]-1][each_column],green[points[0]][each_column],green[points[1]][each_column],green[points[2]][each_column],green[points[3]][each_column]],[points[0] -1 ]+ points,pixel)
            start = points[-1]
            
            
            
     
    img = np.concatenate((blue, green, red), axis=2)
    cv2.imwrite("renew96996.png", img)

def compare(img_path):
    orginal_img = cv2.imread("Lenna.png",cv2.IMREAD_COLOR)
    img = cv2.imread(img_path,cv2.IMREAD_COLOR)
    
    return np.square(np.subtract(orginal_img,img)).mean()
    
if __name__ == "__main__":
    mask1 = [[1,2],
             [0,1]]
    mask2= [[1,0,2,1,2,0],
            [2,1,1,0,1,1],
            [0,1,1,2,1,1],
            [1,2,0,1,0,2],
            [0,1,1,2,1,1],
            [2,1,1,0,1,1]]
    colors_bayer = read_with_mask("Lenna.png", mask1)
    start_time = time.time()
    linear(colors_bayer)
    print(f"Linear: {time.time() - start_time}, MSE: { compare('Bayer_linear.png') } ")
    start_time = time.time()
    square(colors_bayer)
    print(f"Square: {time.time() - start_time}, MSE: { compare('Bayer_Square.png') }")
    start_time = time.time()
    cubic(colors_bayer)
    print(f"Cubic: {time.time() - start_time}, MSE: { compare('Bayer_Cubic.png') }")
    
    colors_bayer = read_with_mask("Lenna.png", mask2)
    start_time = time.time()
    linear(colors_bayer)
    print(f"Linear: {time.time() - start_time}, MSE: { compare('Cmos_linear.png') }")
    start_time = time.time()
    square(colors_bayer)
    print(f"Square: {time.time() - start_time}, MSE: { compare('Cmos_square.png') }")
    start_time = time.time()
    cubic(colors_bayer)
    print(f"Cubic: {time.time() - start_time}, MSE: { compare('Cmos_cubic.png') }")
    
    