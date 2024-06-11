import cv2 as cv

img = cv.imread('/Users/nicholasrethans/Documents/Photos/yus.jpg')

#Grayscale img
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
                            #must be odd numbers (controls amt of blur)

#Edge Cascade (Find the edges present in an image)
canny = cv.Canny(blur, 125, 175)
"""
You can play with feeding the Canny function a crisp or more blurred image for it to detect more or less edges
"""
#Dillating the image
dilated = cv.dilate(canny, (5,5), iterations=5)

#Eroding
eroded = cv.erode(dilated, (5,5), iterations = 5)

#resize
resized = cv.resize(img, (500,500))
    #Interpolation args:
    """
    Interpolation is the process of estimating the values of pixels in the resized image based on the values of surrounding pixels in the original image. This is necessary because when you resize an image, you change the number of pixels, and the new pixel locations may not align perfectly with the old pixel locations.
OpenCV provides several interpolation methods, each with its own tradeoff between speed and quality:

cv2.INTER_NEAREST: This is the fastest interpolation method, but it produces the lowest quality results. It simply takes the value of the nearest pixel in the original image and assigns it to the new pixel in the resized image.
cv2.INTER_LINEAR: This method uses bilinear interpolation, which takes into account the values of the four nearest pixels in the original image and performs a weighted average to determine the value of the new pixel.
cv2.INTER_AREA: This method is recommended for shrinking (downscaling) images. It uses pixel area relation to calculate the new pixel values, which helps to avoid moiré patterns and other artifacts.
cv2.INTER_CUBIC: This method uses bicubic interpolation, which considers the values of 16 neighboring pixels in the original image. It produces the highest quality results but is also the slowest method.
cv2.INTER_LANCZOS4: This method uses the Lanczos filter, which is a more complex interpolation algorithm that produces very high-quality results, especially for upscaling images. However, it is computationally expensive.

In general, for most image resizing tasks, cv2.INTER_AREA is a good choice as it provides a balance between quality and speed. For shrinking images, it is recommended over other methods like cv2.INTER_LINEAR as it helps to avoid aliasing artifacts.
For upscaling images, cv2.INTER_CUBIC or cv2.INTER_LANCZOS4 can produce better results, but they are slower and may not be suitable for real-time applications or when processing large batches of images.
    """
#Cropping
cropped = img[50:200, 200:400]

cv.imshow('Test',cropped)
cv.imshow('Frame',img)
cv.waitKey(0)